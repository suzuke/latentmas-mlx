"""
MILESTONE-13: Layer Swap Sensitivity Experiment

Sweeps graft_layer_b across receiver model depth to test whether
alignment quality (SVD vs random T) matters at different injection points.

Pre-registered hypotheses:
  H1: Early-layer random T drops >10pp below SVD (>5pp with Bonferroni)
  H2: cos_sim(pre, post) decreases monotonically as graft moves earlier
  H3: There exists a sensitivity threshold layer

Usage:
  python layer_sweep.py --max_samples 100 --temp 0.0
  python layer_sweep.py --max_samples 10 --temp 0.0  # quick sanity check
"""
import argparse, json, math, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.generate import generate_step, cache as mlx_cache
from mlx_lm.models.base import create_attention_mask
from datasets import load_dataset

import numpy as np


def norm_cdf(x):
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


SYSTEM = "You are a helpful assistant."


# ── Core functions ───────────────────────────────────────

def get_activation_at_layer(model, input_ids, layer_idx):
    """Run model up to layer_idx, return last-token hidden state."""
    inner = model.model if hasattr(model, 'model') else model.language_model.model
    h = inner.embed_tokens(input_ids[None])
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])
    for i, layer in enumerate(inner.layers):
        if i > layer_idx:
            break
        h = layer(h, mask, cache[i])
    return h[0, -1]


def generate_with_graft_and_measure(
    model, tokenizer, input_ids, graft_layer, grafted_activation,
    max_tokens=2048, temp=0.0,
):
    """
    Inject grafted_activation at graft_layer, measure cos_sim and L2_ratio,
    then generate. Returns (text, cos_sim, l2_ratio).
    """
    inner = model.model if hasattr(model, 'model') else model.language_model.model
    n_layers = len(inner.layers)

    h = inner.embed_tokens(input_ids[None])
    kv_cache = mlx_cache.make_prompt_cache(model)
    mask = create_attention_mask(h, kv_cache[0])

    cos_sim = 0.0
    l2_ratio = 0.0

    for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
        h = layer(h, mask, c)
        if i == graft_layer:
            # Measure before replacement
            h_pre = h[0, -1].astype(mx.float32)
            g = grafted_activation.astype(mx.float32)

            # cos_sim(pre, grafted)
            dot = mx.sum(h_pre * g)
            norm_pre = mx.sqrt(mx.sum(h_pre * h_pre))
            norm_g = mx.sqrt(mx.sum(g * g))
            cos_sim = (dot / (norm_pre * norm_g + 1e-8)).item()

            # L2 ratio: ||grafted|| / ||residual_at_layer||
            l2_ratio = (norm_g / (norm_pre + 1e-8)).item()

            # Replace last-token activation
            h = mx.concatenate(
                [h[:, :-1, :], g.reshape(1, 1, -1).astype(h.dtype)],
                axis=1,
            )

    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])

    # Generate
    h_norm = inner.norm(h)
    if hasattr(model, 'args') and getattr(model.args, 'tie_word_embeddings', False):
        logits = inner.embed_tokens.as_linear(h_norm)
    else:
        logits = model.lm_head(h_norm) if hasattr(model, 'lm_head') else inner.embed_tokens.as_linear(h_norm)

    logits = logits[0, -1]
    if temp == 0.0:
        first_token = mx.argmax(logits)
    else:
        first_token = mx.random.categorical(logits * (1.0 / temp))

    tokens = [first_token.item() if hasattr(first_token, 'item') else int(first_token)]
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]

    if tokens[0] in eos_ids:
        return tokenizer.decode([]), cos_sim, l2_ratio

    sampler = (lambda x: mx.argmax(x, axis=-1)) if temp == 0.0 else (lambda x: mx.random.categorical(x * (1.0 / temp)))

    for tok_val, _ in generate_step(
        mx.array([tokens[-1]]), model,
        max_tokens=max_tokens - 1,
        prompt_cache=kv_cache,
        sampler=sampler,
    ):
        t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
        if t in eos_ids:
            break
        tokens.append(t)

    return tokenizer.decode(tokens), cos_sim, l2_ratio


def simple_generate(model, tokenizer, input_ids, max_tokens=2048, temp=0.0):
    """Standard generation without grafting."""
    tokens = []
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]
    sampler = (lambda x: mx.argmax(x, axis=-1)) if temp == 0.0 else (lambda x: mx.random.categorical(x * (1.0 / temp)))
    for tok_val, _ in generate_step(
        input_ids, model,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
        if t in eos_ids:
            break
        tokens.append(t)
    return tokenizer.decode(tokens)


# ── Alignment matrices ───────────────────────────────────

def compute_svd_alignment(model_a, model_b, n_sample=8192):
    """SVD-based alignment T: h_b ≈ h_a @ T."""
    inner_a = model_a.model if hasattr(model_a, 'model') else model_a.language_model.model
    inner_b = model_b.model if hasattr(model_b, 'model') else model_b.language_model.model
    vocab_a = inner_a.embed_tokens.weight.shape[0]
    vocab_b = inner_b.embed_tokens.weight.shape[0]
    n = min(n_sample, vocab_a, vocab_b)
    ids = mx.arange(n)
    E_a = inner_a.embed_tokens(ids).astype(mx.float32)
    E_b = inner_b.embed_tokens(ids).astype(mx.float32)
    _, _, Vh_a = mx.linalg.svd(E_a, stream=mx.cpu)
    _, _, Vh_b = mx.linalg.svd(E_b, stream=mx.cpu)
    mx.eval(Vh_a, Vh_b)
    d_a, d_b = Vh_a.shape[0], Vh_b.shape[0]
    rank = min(d_a, d_b)
    T = mx.matmul(mx.transpose(Vh_a[:rank, :]), Vh_b[:rank, :])
    mx.eval(T)
    return T


def compute_random_alignment(d_a, d_b, seed=42):
    """Random orthogonal matrix (control)."""
    d = max(d_a, d_b)
    key = mx.random.key(seed)
    M = mx.random.normal((d, d), key=key)
    U, _, Vh = mx.linalg.svd(M, stream=mx.cpu)
    R = mx.matmul(U, Vh)
    T = R[:d_a, :d_b]
    mx.eval(T)
    return T


# ── Data loading ─────────────────────────────────────────

def build_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Target Question: {question}\n\nSolve step-by-step.\nPut final answer in \\boxed{{YOUR_FINAL_ANSWER}}."},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return mx.array(tokenizer.encode(text, add_special_tokens=False))


def extract_answer(text):
    import re
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", boxes[-1])
        return nums[0] if nums else boxes[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def load_math500(max_samples, min_level=3, max_level=5):
    """Load MATH-500 level 3-5 (harder subset)."""
    import re
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    filtered = []
    for d in ds:
        level_str = d.get("level", "Level 1")
        # Extract numeric level
        m = re.search(r"(\d+)", str(level_str))
        level = int(m.group(1)) if m else 1
        if min_level <= level <= max_level:
            filtered.append(d)

    data = []
    for d in filtered[:max_samples]:
        q = d.get("problem", d.get("question", ""))
        sol = d.get("solution", d.get("answer", ""))
        boxes = re.findall(r"\\boxed\{([^}]*)\}", sol)
        gold = boxes[-1] if boxes else sol.strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", gold)
        gold_norm = nums[0] if nums else gold
        data.append({"question": q, "gold": gold_norm})

    print(f"Loaded {len(data)} MATH-500 problems (level {min_level}-{max_level})")
    return data


# ── Main experiment ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MILESTONE-13 Layer Sweep")
    p.add_argument("--model_a", default="mlx-community/Qwen3-1.7B-4bit", help="Sender")
    p.add_argument("--model_b", default="mlx-community/Qwen3-4B-4bit", help="Receiver")
    p.add_argument("--graft_layer_a_pct", type=float, default=0.5, help="Source layer as %% of depth")
    p.add_argument("--sweep_pcts", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 0.90],
                   help="Receiver layer percentages to sweep")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--min_level", type=int, default=3)
    p.add_argument("--max_level", type=int, default=5)
    p.add_argument("--output", type=str, default=None, help="Output JSONL path")
    p.add_argument("--conditions", nargs="+", default=["baseline", "svd", "random"],
                   choices=["baseline", "svd", "random"])
    args = p.parse_args()

    # Load models
    print(f"Loading Model A: {args.model_a}...")
    model_a, tok_a = mlx_lm.load(args.model_a)
    print(f"Loading Model B: {args.model_b}...")
    model_b, tok_b = mlx_lm.load(args.model_b)

    inner_a = model_a.model if hasattr(model_a, 'model') else model_a.language_model.model
    inner_b = model_b.model if hasattr(model_b, 'model') else model_b.language_model.model
    n_layers_a = len(inner_a.layers)
    n_layers_b = len(inner_b.layers)
    hidden_a = inner_a.embed_tokens(mx.array([[0]])).shape[-1]
    hidden_b = inner_b.embed_tokens(mx.array([[0]])).shape[-1]

    graft_layer_a = int(n_layers_a * args.graft_layer_a_pct)
    sweep_layers_b = [int(n_layers_b * pct) for pct in args.sweep_pcts]
    # Clamp to valid range
    sweep_layers_b = [min(max(l, 0), n_layers_b - 2) for l in sweep_layers_b]

    print(f"Model A: {args.model_a}, layers={n_layers_a}, hidden={hidden_a}")
    print(f"Model B: {args.model_b}, layers={n_layers_b}, hidden={hidden_b}")
    print(f"Source layer: {graft_layer_a} ({args.graft_layer_a_pct*100:.0f}%)")
    print(f"Sweep layers B: {list(zip(args.sweep_pcts, sweep_layers_b))}")
    print(f"Conditions: {args.conditions}")
    print(f"temp={args.temp}, n={args.max_samples}")

    # Compute alignment matrices
    T_svd, T_random = None, None
    if "svd" in args.conditions:
        print("Computing SVD alignment...")
        T_svd = compute_svd_alignment(model_a, model_b)
    if "random" in args.conditions:
        print("Computing random orthogonal alignment...")
        T_random = compute_random_alignment(hidden_a, hidden_b)

    # Load data
    data = load_math500(args.max_samples, args.min_level, args.max_level)
    if not data:
        print("ERROR: No data loaded. Exiting.")
        return

    # Output file
    out_path = args.output or f"audit-results/layer-sweep-{int(time.time())}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    results = []  # [{layer_pct, condition, accuracy, avg_cos_sim, avg_l2_ratio, n}]

    # Run baseline (receiver only, no graft) — once
    if "baseline" in args.conditions:
        print("\n=== BASELINE (Model B only) ===")
        correct = 0
        for i, item in enumerate(data):
            ids = build_prompt(tok_b, item["question"])
            resp = simple_generate(model_b, tok_b, ids, args.max_tokens, args.temp)
            pred = extract_answer(resp)
            ok = pred.strip().lower() == item["gold"].strip().lower()
            correct += ok
            if (i + 1) % 20 == 0 or i == len(data) - 1:
                print(f"  [baseline] {i+1}/{len(data)} acc={correct}/{i+1} ({correct/(i+1)*100:.1f}%)")

        baseline_acc = correct / len(data)
        results.append({
            "layer_pct": "N/A", "layer_idx": "N/A",
            "condition": "baseline", "accuracy": baseline_acc,
            "correct": correct, "n": len(data),
            "avg_cos_sim": None, "avg_l2_ratio": None,
        })
        print(f"  BASELINE: {correct}/{len(data)} = {baseline_acc*100:.1f}%")

    # Run graft conditions at each layer
    for pct, layer_b in zip(args.sweep_pcts, sweep_layers_b):
        for cond in [c for c in args.conditions if c != "baseline"]:
            T = T_svd if cond == "svd" else T_random
            print(f"\n=== Layer {layer_b} ({pct*100:.0f}%), condition={cond} ===")

            correct = 0
            cos_sims = []
            l2_ratios = []

            for i, item in enumerate(data):
                # Capture activation from A
                ids_a = build_prompt(tok_a, item["question"])
                act_a = get_activation_at_layer(model_a, ids_a, graft_layer_a)

                # Apply alignment
                if T is not None:
                    act_aligned = mx.matmul(act_a.astype(mx.float32), T).astype(act_a.dtype)
                else:
                    act_aligned = act_a

                # Inject into B and generate
                ids_b = build_prompt(tok_b, item["question"])
                resp, cos_sim, l2_ratio = generate_with_graft_and_measure(
                    model_b, tok_b, ids_b, layer_b, act_aligned,
                    args.max_tokens, args.temp,
                )

                pred = extract_answer(resp)
                ok = pred.strip().lower() == item["gold"].strip().lower()
                correct += ok
                cos_sims.append(cos_sim)
                l2_ratios.append(l2_ratio)

                if (i + 1) % 20 == 0 or i == len(data) - 1:
                    print(f"  [{cond}@{pct*100:.0f}%] {i+1}/{len(data)} acc={correct}/{i+1} ({correct/(i+1)*100:.1f}%) cos={np.mean(cos_sims):.4f} l2r={np.mean(l2_ratios):.4f}")

            acc = correct / len(data)
            row = {
                "layer_pct": pct, "layer_idx": layer_b,
                "condition": cond, "accuracy": acc,
                "correct": correct, "n": len(data),
                "avg_cos_sim": float(np.mean(cos_sims)),
                "avg_l2_ratio": float(np.mean(l2_ratios)),
                "std_cos_sim": float(np.std(cos_sims)),
                "std_l2_ratio": float(np.std(l2_ratios)),
            }
            results.append(row)

    # Write results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Layer%':<8} {'Condition':<10} {'Acc':<8} {'cos_sim':<10} {'L2_ratio':<10} {'n'}")
    print("-"*60)
    for r in results:
        cs = f"{r['avg_cos_sim']:.4f}" if r['avg_cos_sim'] is not None else "N/A"
        lr = f"{r['avg_l2_ratio']:.4f}" if r['avg_l2_ratio'] is not None else "N/A"
        lp = f"{r['layer_pct']*100:.0f}%" if isinstance(r['layer_pct'], float) else r['layer_pct']
        print(f"{lp:<8} {r['condition']:<10} {r['accuracy']*100:.1f}%   {cs:<10} {lr:<10} {r['n']}")

    # Statistical tests (H1: SVD > random at early layers)
    print("\n" + "="*80)
    print("HYPOTHESIS TESTS (H1: SVD_acc - random_acc > 5pp at early layers)")
    print("="*80)
    svd_results = {r['layer_pct']: r for r in results if r['condition'] == 'svd'}
    rand_results = {r['layer_pct']: r for r in results if r['condition'] == 'random'}
    n_tests = len(args.sweep_pcts)
    alpha_bonf = 0.05 / n_tests

    for pct in args.sweep_pcts:
        if pct in svd_results and pct in rand_results:
            s = svd_results[pct]
            r = rand_results[pct]
            n = s['n']
            diff = s['accuracy'] - r['accuracy']
            # Fisher exact approximation via normal test for proportions
            p_pool = (s['correct'] + r['correct']) / (2 * n)
            se = math.sqrt(2 * p_pool * (1 - p_pool) / n) if p_pool > 0 and p_pool < 1 else 1
            z = diff / se if se > 0 else 0
            p_val = 1 - norm_cdf(z)  # one-sided
            sig = "***" if p_val < alpha_bonf else ""
            print(f"  Layer {pct*100:.0f}%: SVD={s['accuracy']*100:.1f}% random={r['accuracy']*100:.1f}% diff={diff*100:+.1f}pp p={p_val:.4f} {sig}")
            if p_val < alpha_bonf and diff > 0.05:
                print(f"    → H1 SUPPORTED at layer {pct*100:.0f}% (Bonferroni α={alpha_bonf:.4f})")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
