"""Compare baseline vs latent_mas JSONL outputs.

Tabulates:
  - accuracy
  - truncation rate (out_tokens == max_tokens)
  - failed-and-truncated vs failed-and-reasoning rates
  - per-index head-to-head (same question, both methods)

Usage:  uv run python audit-results/notes/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE = Path(__file__).parent.parent / "baseline-gsm8k-50-bf16.jsonl"
LATENT = Path(__file__).parent.parent / "latent_mas-gsm8k-50-bf16.jsonl"


def load(path: Path) -> tuple[dict, list[dict]]:
    meta = None
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("_meta"):
                meta = d
                continue
            samples.append(d)
    return meta, samples


def main() -> None:
    if not BASELINE.exists() or not LATENT.exists():
        print(f"Missing data: {BASELINE.exists()=}, {LATENT.exists()=}")
        sys.exit(1)

    b_meta, b_samples = load(BASELINE)
    l_meta, l_samples = load(LATENT)

    max_tokens_b = b_meta["max_tokens"]
    max_tokens_l = l_meta["max_tokens"]
    assert max_tokens_b == max_tokens_l, "Comparing different token budgets"

    def stats(label: str, samples: list[dict], max_t: int) -> dict:
        n = len(samples)
        correct = sum(1 for s in samples if s["correct"])
        trunc = sum(1 for s in samples if s["out_tokens"] >= max_t)
        trunc_wrong = sum(1 for s in samples if s["out_tokens"] >= max_t and not s["correct"])
        non_trunc_wrong = sum(1 for s in samples if s["out_tokens"] < max_t and not s["correct"])
        return {
            "label": label,
            "n": n,
            "correct": correct,
            "acc": correct / n,
            "trunc_total": trunc,
            "trunc_wrong": trunc_wrong,
            "non_trunc_wrong": non_trunc_wrong,
        }

    b = stats("Baseline", b_samples, max_tokens_b)
    l = stats("LatentMAS", l_samples, max_tokens_l)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'LatentMAS':>12}")
    print("-" * 51)
    print(f"{'samples':<25} {b['n']:>12} {l['n']:>12}")
    print(f"{'accuracy':<25} {b['acc']:>11.1%} {l['acc']:>11.1%}")
    print(f"{'truncated (>= max)':<25} {b['trunc_total']:>12} {l['trunc_total']:>12}")
    print(f"{'    of which wrong':<25} {b['trunc_wrong']:>12} {l['trunc_wrong']:>12}")
    print(f"{'non-truncated wrong':<25} {b['non_trunc_wrong']:>12} {l['non_trunc_wrong']:>12}")

    gap_total = l['acc'] - b['acc']
    # "Reasoning-only" accuracy: exclude truncations from both denominator AND count
    def reasoning_acc(samples, max_t):
        non_trunc = [s for s in samples if s["out_tokens"] < max_t]
        if not non_trunc:
            return None
        return sum(1 for s in non_trunc if s["correct"]) / len(non_trunc)

    b_rea = reasoning_acc(b_samples, max_tokens_b)
    l_rea = reasoning_acc(l_samples, max_tokens_l)
    print()
    print(f"{'reasoning-only acc':<25} {b_rea:>11.1%} {l_rea:>11.1%}  (excluding truncations)")
    print()
    print(f"Raw gap (L - B):       {gap_total*100:+.1f} pp")
    print(f"Reasoning-only gap:    {(l_rea - b_rea)*100:+.1f} pp")
    print()

    # Head-to-head per index
    b_by_idx = {s["index"]: s for s in b_samples}
    l_by_idx = {s["index"]: s for s in l_samples}
    shared = sorted(set(b_by_idx) & set(l_by_idx))

    both_ok = 0
    both_wrong = 0
    only_b = 0
    only_l = 0
    for idx in shared:
        bo = b_by_idx[idx]["correct"]
        lo = l_by_idx[idx]["correct"]
        if bo and lo:
            both_ok += 1
        elif not bo and not lo:
            both_wrong += 1
        elif bo and not lo:
            only_b += 1
        elif lo and not bo:
            only_l += 1

    print(f"Head-to-head ({len(shared)} shared indices):")
    print(f"  Both correct:           {both_ok}")
    print(f"  Both wrong:             {both_wrong}")
    print(f"  Only baseline correct:  {only_b}")
    print(f"  Only LatentMAS correct: {only_l}")
    print()

    # List samples where they diverge
    print("=== Samples where baseline correct but LatentMAS wrong ===")
    for idx in shared:
        bo = b_by_idx[idx]
        lo = l_by_idx[idx]
        if bo["correct"] and not lo["correct"]:
            l_trunc = lo["out_tokens"] >= max_tokens_l
            print(f"  idx={idx} gold={bo['gold']} | baseline=✓({bo['extracted_pred']}, {bo['out_tokens']}t) | latent=✗({lo['extracted_pred']}, {lo['out_tokens']}t){'[TRUNC]' if l_trunc else ''}")

    print()
    print("=== Samples where LatentMAS correct but baseline wrong ===")
    for idx in shared:
        bo = b_by_idx[idx]
        lo = l_by_idx[idx]
        if not bo["correct"] and lo["correct"]:
            b_trunc = bo["out_tokens"] >= max_tokens_b
            print(f"  idx={idx} gold={bo['gold']} | baseline=✗({bo['extracted_pred']}, {bo['out_tokens']}t){'[TRUNC]' if b_trunc else ''} | latent=✓({lo['extracted_pred']}, {lo['out_tokens']}t)")


if __name__ == "__main__":
    main()
