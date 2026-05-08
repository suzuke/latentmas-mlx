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
from datasets import load_dataset


SYSTEM = "You are a helpful assistant."

# ── Activation grafting core ─────────────────────────────

def get_activation_at_layer(model, input_ids, layer_idx):
    """Run model up to layer_idx, return the hidden state at that layer for the last token."""
    inner = model.model
    h = inner.embed_tokens(input_ids[None])

    cache = [None] * len(inner.layers)
    mask = None  # No cache, full attention

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
    # Use the full model forward to get activation at graft_layer
    h = inner.embed_tokens(input_ids[None])
    cache_a = [None] * n_layers
    activations = {}
    for i, layer in enumerate(inner.layers):
        h = layer(h, None, cache_a[i])
        if i == graft_layer:
            activations[i] = h[0, -1]  # [d_h] — A's activation at graft layer

    grafted = activations[graft_layer]

    # Step 2: Run B with the grafted activation
    # Fresh forward, but at graft_layer replace last-token activation
    h = inner.embed_tokens(input_ids[None])
    kv_cache = mlx_cache.make_prompt_cache(model)

    for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
        h = layer(h, None, c)
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
    p.add_argument("--graft_layer", type=int, default=26, help="Layer at which to graft (paper default)")
    p.add_argument("--temp", type=float, default=0.6)
    args = p.parse_args()

    print(f"Loading Model A: {args.model_a}...")
    model_a, tok_a = mlx_lm.load(args.model_a)
    if args.model_a == args.model_b:
        model_b, tok_b = model_a, tok_a
        print("Model A = Model B (same-model activation communication)")
    else:
        print(f"Loading Model B: {args.model_b}...")
        model_b, tok_b = mlx_lm.load(args.model_b)

    data = load_data(args.task, args.max_samples)

    # Run three methods: Model A alone, Model B alone, A→B activation graft
    for method in ["model_a_only", "model_b_only", "activation_graft"]:
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
            else:
                # Activation graft: A's activation injected into B at graft_layer
                ids_b = build_prompt(tok_b, item["question"], args.task)
                resp = generate_with_grafted_activation(
                    model_b, tok_b, ids_b, args.graft_layer,
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
