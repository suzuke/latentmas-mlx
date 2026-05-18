"""
MLX Communicating Activations — activation grafting between two LLMs.

Core idea: Run Model A to layer k, run Model B to layer j, replace B's last-token
activation with A's, then continue B's forward pass to generate output.

Usage:
  python mlx_comm_activations.py --task gsm8k --max_samples 50
"""
import argparse, json, re, time
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.generate import generate_step, cache as mlx_cache
from mlx_lm.models.base import create_attention_mask
from datasets import load_dataset


SYSTEM = "You are a helpful assistant."

# ── Cross-LoRA-style SVD subspace alignment ──────────────

def compute_random_alignment_matrix(d_a, d_b, seed=42):
    """Random orthogonal matrix for control / ablation.

    Generates a random orthogonal-style projection matrix of shape [d_a, d_b]
    via SVD of a random Gaussian. Same structural properties as the SVD-derived
    alignment (orthogonal-like) but with NO model-derived information — useful
    to test whether SVD alignment is doing meaningful work or whether any
    rotation suffices.
    """
    d = max(d_a, d_b)
    key = mx.random.key(seed)
    M = mx.random.normal((d, d), key=key)
    U, _, Vh = mx.linalg.svd(M, stream=mx.cpu)
    R = mx.matmul(U, Vh)  # orthogonal d×d
    T = R[:d_a, :d_b]
    mx.eval(T)
    return T


def compute_svd_alignment_matrix(model_a, model_b, n_sample=8192, rank=None):
    """Compute T (d_a × d_b) such that h_a @ T approximately maps Model A's
    activation space onto Model B's, via Cross-LoRA-style SVD subspace
    alignment of the embedding matrices.

    For row-vector convention:
        h_b ≈ h_a @ T
    where T = V_a @ V_b^T, V_a/V_b are right singular vectors of E_a/E_b.

    Assumption: top-k PCs of two LLMs' embedding spaces encode similar
    concepts in rank order. This is the simplest possible alignment with
    NO training data; it may not always hold.

    Args:
        model_a, model_b: MLX models with .model.embed_tokens or
            .language_model.model.embed_tokens
        n_sample: number of vocab rows to sample (use min over both vocabs)
        rank: truncation rank; None = full hidden_size

    Returns:
        T: mx.array of shape [d_a, d_b], float32
    """
    inner_a = model_a.model if hasattr(model_a, 'model') else model_a.language_model.model
    inner_b = model_b.model if hasattr(model_b, 'model') else model_b.language_model.model
    embed_a = inner_a.embed_tokens
    embed_b = inner_b.embed_tokens

    # Sample-based to handle quantized embeddings cleanly
    vocab_a, vocab_b = embed_a.weight.shape[0], embed_b.weight.shape[0]
    n = min(n_sample, vocab_a, vocab_b)
    ids = mx.arange(n)
    E_a = embed_a(ids).astype(mx.float32)   # [n, d_a]
    E_b = embed_b(ids).astype(mx.float32)   # [n, d_b]

    # SVD on CPU stream (large matrices, deterministic)
    _, _, Vh_a = mx.linalg.svd(E_a, stream=mx.cpu)   # Vh_a is [d_a, d_a]
    _, _, Vh_b = mx.linalg.svd(E_b, stream=mx.cpu)   # Vh_b is [d_b, d_b]
    mx.eval(Vh_a, Vh_b)

    d_a, d_b = Vh_a.shape[0], Vh_b.shape[0]
    if rank is None:
        rank = min(d_a, d_b)
    # Truncate to rank
    Vh_a_top = Vh_a[:rank, :]   # [rank, d_a]
    Vh_b_top = Vh_b[:rank, :]   # [rank, d_b]

    # Row-vector form: h_b = h_a @ (Vh_a^T @ Vh_b)
    # i.e. T = Vh_a_top^T @ Vh_b_top, shape [d_a, d_b]
    T = mx.matmul(mx.transpose(Vh_a_top), Vh_b_top)
    mx.eval(T)
    return T


# ── Activation grafting core ─────────────────────────────

def get_activation_at_layer(model, input_ids, layer_idx):
    """Run model up to layer_idx, return the hidden state at that layer for the last token."""
    inner = model.model
    h = inner.embed_tokens(input_ids[None])

    cache = [None] * len(inner.layers)
    # Causal mask is required for prefill (N>1 tokens); without it every
    # prompt token attends to every other token including future ones,
    # which is OOD for the causally-trained model.
    mask = create_attention_mask(h, cache[0])

    for i, layer in enumerate(inner.layers):
        if i > layer_idx:
            break
        h = layer(h, mask, cache[i])

    return h[0, -1]  # [d_h] — last token's activation at layer_idx


def generate_with_grafted_activation(
    model, tokenizer, input_ids, graft_layer,
    max_tokens=2048, temp=0.6,
):
    """
    Communicating Activations (same-model variant):
    1. Run model A (= model with temp=0.7) to generate a completion, get layer-k activation
    2. Run model B (= same model) up to layer j, replace last-token activation, continue
    
    Since A=B, we use the completion's activation as the "communication".
    """
    inner = model.model
    n_layers = len(inner.layers)

    # Step 1: Run A — generate a short completion to get its "thinking"
    # Use the full model forward to get activation at graft_layer.
    # Causal mask required for prefill (otherwise tokens see future = OOD).
    h = inner.embed_tokens(input_ids[None])
    cache_a = [None] * n_layers
    mask_a = create_attention_mask(h, cache_a[0])
    activations = {}
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask_a, cache_a[i])
        if i == graft_layer:
            activations[i] = h[0, -1]  # [d_h] — A's activation at graft layer

    grafted = activations[graft_layer]

    # Step 2: Run B with the grafted activation
    # Fresh forward, but at graft_layer replace last-token activation.
    # Same causal mask requirement applies.
    h = inner.embed_tokens(input_ids[None])
    kv_cache = mlx_cache.make_prompt_cache(model)
    mask_b = create_attention_mask(h, kv_cache[0])

    for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
        h = layer(h, mask_b, c)
        if i == graft_layer:
            # Replace last token activation: h[:, -1, :] = grafted
            # MLX doesn't have .at indexing, use concatenation
            h = mx.concatenate([h[:, :-1, :], grafted.reshape(1, 1, -1)], axis=1)

    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])

    # Generate from the modified state
    h_norm = inner.norm(h)
    if model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm)

    logits = logits[0, -1]
    first_token = mx.random.categorical(logits * (1.0 / temp))

    tokens = [first_token.item() if hasattr(first_token, 'item') else int(first_token)]
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]

    for tok_val, _ in generate_step(
        mx.array([tokens[-1]]), model,
        max_tokens=max_tokens - 1,
        prompt_cache=kv_cache,
        sampler=lambda x: mx.random.categorical(x * (1.0 / temp)),
    ):
        t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
        if t in eos_ids:
            break
        tokens.append(t)

    return tokenizer.decode(tokens)


def generate_with_externally_grafted_activation(
    model, tokenizer, input_ids, graft_layer, grafted_activation,
    max_tokens=2048, temp=0.6,
):
    """
    Cross-model variant: takes a pre-computed grafted_activation (typically
    from Model A via get_activation_at_layer) and injects it into THIS model
    (Model B) at graft_layer's last-token position.

    grafted_activation: 1-D array of shape (hidden_size,) matching this
    model's hidden_size. For cross-arch grafts where hidden sizes differ,
    pass an externally-projected activation (projection not handled here).
    """
    inner = model.model if hasattr(model, 'model') else model.language_model.model
    n_layers = len(inner.layers)

    # Run forward with injection
    h = inner.embed_tokens(input_ids[None])
    target_d = h.shape[-1]
    if grafted_activation.shape[-1] != target_d:
        raise ValueError(
            f"Grafted activation dim {grafted_activation.shape[-1]} != "
            f"target model hidden dim {target_d}. Cross-arch with different "
            f"hidden sizes requires an external projection adapter."
        )
    if not (0 <= graft_layer < n_layers):
        raise ValueError(
            f"graft_layer={graft_layer} out of range for model with {n_layers} layers"
        )

    kv_cache = mlx_cache.make_prompt_cache(model)
    mask = create_attention_mask(h, kv_cache[0])

    for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
        h = layer(h, mask, c)
        if i == graft_layer:
            h = mx.concatenate(
                [h[:, :-1, :], grafted_activation.reshape(1, 1, -1).astype(h.dtype)],
                axis=1,
            )

    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])

    # Generate first token from modified state
    h_norm = inner.norm(h)
    if hasattr(model, 'args') and getattr(model.args, 'tie_word_embeddings', False):
        logits = inner.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm) if hasattr(model, 'lm_head') else inner.embed_tokens.as_linear(h_norm)

    logits = logits[0, -1]
    first_token = mx.random.categorical(logits * (1.0 / temp))

    tokens = [first_token.item() if hasattr(first_token, 'item') else int(first_token)]
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]

    if tokens[0] in eos_ids:
        return tokenizer.decode([])

    for tok_val, _ in generate_step(
        mx.array([tokens[-1]]), model,
        max_tokens=max_tokens - 1,
        prompt_cache=kv_cache,
        sampler=lambda x: mx.random.categorical(x * (1.0 / temp)),
    ):
        t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
        if t in eos_ids:
            break
        tokens.append(t)

    return tokenizer.decode(tokens)


def simple_generate(model, tokenizer, input_ids, max_tokens=2048, temp=0.6):
    """Standard generation without grafting."""
    tokens = []
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]
    for tok_val, _ in generate_step(
        input_ids, model,
        max_tokens=max_tokens,
        sampler=lambda x: mx.random.categorical(x * (1.0 / temp)),
    ):
        t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
        if t in eos_ids:
            break
        tokens.append(t)
    return tokenizer.decode(tokens)


# ── Prompt & answer extraction ───────────────────────────

def build_prompt(tokenizer, question, task):
    suffix = "\nYour final answer must be selected from A,B,C,D." if task in ("arc_challenge", "gpqa") else ""
    fmt = "\\boxed{A/B/C/D}" if task in ("arc_challenge", "gpqa") else "\\boxed{YOUR_FINAL_ANSWER}"
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Target Question: {question}\n\nSolve step-by-step.{suffix}\nPut final answer in {fmt}."},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return mx.array(tokenizer.encode(text, add_special_tokens=False))


def extract_answer(text, task):
    if task in ("arc_challenge", "gpqa"):
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxes:
            return boxes[-1].strip().upper()
        for letter in ["D", "C", "B", "A"]:
            if letter in text.upper():
                return letter
        return ""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", boxes[-1])
        return nums[0] if nums else boxes[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def extract_gold(answer_text):
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    return m.group(1) if m else ""


def load_data(task, max_samples):
    if task == "gsm8k":
        ds = list(load_dataset("gsm8k", "main", split="test"))[:max_samples]
        return [{"question": d["question"], "gold": extract_gold(d["answer"])} for d in ds]
    elif task == "arc_challenge":
        ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))[:max_samples]
        items = []
        for d in ds:
            labels = d["choices"]["label"]
            texts = d["choices"]["text"]
            choices = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
            items.append({"question": f"{d['question']}\n{choices}", "gold": d["answerKey"].upper()})
        return items
    raise ValueError(f"Unknown task: {task}")


# ── Main ─────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", default="mlx-community/Qwen3-8B-4bit", help="Sender model")
    p.add_argument("--model_b", default="mlx-community/Qwen3-8B-4bit", help="Receiver model")
    p.add_argument("--task", choices=["gsm8k", "arc_challenge"], default="gsm8k")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--graft_layer", type=int, default=26, help="Layer at which to graft (same-model)")
    p.add_argument("--graft_layer_a", type=int, default=None,
                   help="Layer in Model A to capture activation (cross-model). Defaults to --graft_layer")
    p.add_argument("--graft_layer_b", type=int, default=None,
                   help="Layer in Model B to inject activation (cross-model). Defaults to --graft_layer")
    p.add_argument("--align", choices=["none", "svd", "random"], default="none",
                   help="Cross-model alignment method (only used when models differ). "
                        "'random' is a control / ablation using a random orthogonal matrix.")
    p.add_argument("--svd_rank", type=int, default=None,
                   help="Truncation rank for SVD alignment (default: full hidden_size)")
    p.add_argument("--temp", type=float, default=0.6)
    args = p.parse_args()

    print(f"Loading Model A: {args.model_a}...")
    model_a, tok_a = mlx_lm.load(args.model_a)
    is_cross_model = args.model_a != args.model_b
    if not is_cross_model:
        model_b, tok_b = model_a, tok_a
        print("Model A = Model B (same-model activation communication)")
    else:
        print(f"Loading Model B: {args.model_b}...")
        model_b, tok_b = mlx_lm.load(args.model_b)

    # Hidden-size sanity check for cross-model graft
    inner_a = model_a.model if hasattr(model_a, 'model') else model_a.language_model.model
    inner_b = model_b.model if hasattr(model_b, 'model') else model_b.language_model.model
    h_test_a = inner_a.embed_tokens(mx.array([[0]]))
    h_test_b = inner_b.embed_tokens(mx.array([[0]]))
    hidden_a = h_test_a.shape[-1]
    hidden_b = h_test_b.shape[-1]
    n_layers_a = len(inner_a.layers)
    n_layers_b = len(inner_b.layers)
    print(f"Model A: hidden={hidden_a}, n_layers={n_layers_a}")
    print(f"Model B: hidden={hidden_b}, n_layers={n_layers_b}")

    graft_layer_a = args.graft_layer_a if args.graft_layer_a is not None else args.graft_layer
    graft_layer_b = args.graft_layer_b if args.graft_layer_b is not None else args.graft_layer

    # Compute cross-model alignment matrix if requested
    align_matrix = None
    if is_cross_model and args.align == "svd":
        print(f"Computing SVD alignment matrix (rank={args.svd_rank or 'full'})...")
        align_matrix = compute_svd_alignment_matrix(model_a, model_b, rank=args.svd_rank)
        print(f"  Alignment T shape: {align_matrix.shape}")
    elif is_cross_model and args.align == "random":
        print(f"Computing RANDOM orthogonal alignment matrix (control / ablation)...")
        align_matrix = compute_random_alignment_matrix(hidden_a, hidden_b)
        print(f"  Random T shape: {align_matrix.shape}")

    # Decide which methods to run
    methods = ["model_a_only", "model_b_only"]
    if is_cross_model:
        # Cross-model graft path:
        # - same hidden size → graft works with or without alignment
        # - different hidden size → SVD alignment provides the projection
        if hidden_a == hidden_b:
            methods.append("cross_model_graft")
            method_suffix = f" (align={args.align})" if args.align != "none" else ""
            print(f"Will run cross_model_graft{method_suffix}: A.layer[{graft_layer_a}] -> B.layer[{graft_layer_b}]")
        elif args.align in ("svd", "random"):
            methods.append("cross_model_graft")
            print(f"Will run cross_model_graft (align={args.align}, cross-arch {hidden_a}->{hidden_b}): A.layer[{graft_layer_a}] -> B.layer[{graft_layer_b}]")
        else:
            print(f"⚠ Skipping cross_model_graft: hidden sizes differ ({hidden_a} vs {hidden_b}) and --align=none")
            print("  Use --align svd|random to enable cross-arch projection.")
    else:
        methods.append("activation_graft")
        print(f"Will run same-model activation_graft at layer {args.graft_layer}")

    data = load_data(args.task, args.max_samples)

    for method in methods:
        correct = 0
        total_time = 0

        for i, item in enumerate(data):
            t0 = time.time()

            if method == "model_a_only":
                ids = build_prompt(tok_a, item["question"], args.task)
                resp = simple_generate(model_a, tok_a, ids, args.max_tokens, args.temp)
            elif method == "model_b_only":
                ids = build_prompt(tok_b, item["question"], args.task)
                resp = simple_generate(model_b, tok_b, ids, args.max_tokens, args.temp)
            elif method == "activation_graft":
                # Same-model graft (A=B): self-graft on B
                ids_b = build_prompt(tok_b, item["question"], args.task)
                resp = generate_with_grafted_activation(
                    model_b, tok_b, ids_b, args.graft_layer,
                    args.max_tokens, args.temp,
                )
            elif method == "cross_model_graft":
                # True cross-model graft: capture from A, optionally align, inject into B
                ids_a = build_prompt(tok_a, item["question"], args.task)
                ids_b = build_prompt(tok_b, item["question"], args.task)
                grafted = get_activation_at_layer(model_a, ids_a, graft_layer_a)
                if align_matrix is not None:
                    # Apply T: h_b = h_a @ T  (row-vector convention)
                    grafted = mx.matmul(grafted.astype(mx.float32), align_matrix).astype(grafted.dtype)
                resp = generate_with_externally_grafted_activation(
                    model_b, tok_b, ids_b, graft_layer_b, grafted,
                    args.max_tokens, args.temp,
                )

            elapsed = time.time() - t0
            total_time += elapsed

            pred = extract_answer(resp, args.task)
            gold = item["gold"]
            ok = pred.strip().lower() == gold.strip().lower()
            correct += ok

            if (i + 1) % 10 == 0 or i == len(data) - 1:
                print(f"  [{method}] {i+1}/{len(data)} acc={correct}/{i+1} time={elapsed:.1f}s")

        acc = correct / len(data)
        avg_time = total_time / len(data)
        print(json.dumps({
            "method": method, "task": args.task, "samples": len(data),
            "accuracy": round(acc, 4), "correct": correct,
            "avg_time_sec": round(avg_time, 2), "total_time_sec": round(total_time, 2),
        }))


if __name__ == "__main__":
    main()
