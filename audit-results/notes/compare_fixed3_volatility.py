"""Compare fixed-3 collapse trajectories against fixed-50 baseline.

Both on same branch (audit/fix-recursivemas-norm, rescaling-enabled),
so this is the apples-to-apples comparison the broken-baseline lacked.

Tests two signatures:
  (1) Mean ||h|| — does fixed-3 differ from fixed-50?
  (2) Within-sample std(||h||) — is fixed-3 more volatile?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FIXED50_NPZ = ROOT / "audit-results" / "activations-fixed-50"
FIXED3_NPZ = ROOT / "audit-results" / "activations-fixed-3collapse"
FIXED50_JSONL = ROOT / "audit-results" / "latent_mas-gsm8k-50-bf16-fixed-cap.jsonl"

TARGET_IDX = [396, 406, 952]


def detect_mode_collapse(raw: str) -> tuple[bool, str]:
    from collections import Counter
    n = len(raw)
    if n == 0:
        return True, "empty"
    ws = sum(1 for c in raw if c.isspace())
    if n > 100 and ws / n > 0.7:
        return True, f"ws_{int(ws*100/n)}%"
    words = raw.split()
    if len(words) >= 100:
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if ngrams:
            top = Counter(ngrams).most_common(1)[0]
            if top[1] > 30:
                return True, f"4gram_x{top[1]}"
    return False, ""


def per_sample_stats(npz_path: Path, role: str) -> tuple[float, float] | None:
    arr = np.load(npz_path)[role]
    if arr.shape[0] == 0:
        return None
    norms = np.linalg.norm(arr, axis=-1)
    return float(norms.mean()), float(norms.std())


def main():
    # Load fixed-50 baseline (skip any that happen to collapse — we want clean baseline)
    baseline_means = {"planner": [], "critic": [], "refiner": []}
    baseline_stds = {"planner": [], "critic": [], "refiner": []}
    n_loaded = 0
    n_collapse_in_baseline = 0
    if FIXED50_JSONL.exists():
        with open(FIXED50_JSONL) as f:
            for line in f:
                d = json.loads(line)
                if d.get("_meta"):
                    continue
                idx = d["index"]
                is_collapse, reason = detect_mode_collapse(d["raw_response"])
                if is_collapse and d["out_tokens"] < 2048:
                    n_collapse_in_baseline += 1
                    print(f"  [baseline collapse] idx={idx} reason={reason} tokens={d['out_tokens']}")
                    # still load — could be a finding
                npz = FIXED50_NPZ / f"sample_{idx:05d}.npz"
                if not npz.exists():
                    continue
                for role in ["planner", "critic", "refiner"]:
                    stats = per_sample_stats(npz, role)
                    if stats is None:
                        continue
                    if not is_collapse:
                        baseline_means[role].append(stats[0])
                        baseline_stds[role].append(stats[1])
                n_loaded += 1

    print(f"\nBaseline fixed-50: loaded={n_loaded}, spontaneous collapses={n_collapse_in_baseline}")

    # Load fixed-3 collapse stats
    fixed3 = {role: {} for role in ["planner", "critic", "refiner"]}
    for idx in TARGET_IDX:
        npz = FIXED3_NPZ / f"sample_{idx:05d}.npz"
        if not npz.exists():
            continue
        for role in ["planner", "critic", "refiner"]:
            stats = per_sample_stats(npz, role)
            if stats is not None:
                fixed3[role][idx] = stats

    print("\n" + "=" * 70)
    print("Per-role comparison: fixed-3 collapses vs fixed-50 baseline")
    print("=" * 70)

    for role in ["planner", "critic", "refiner"]:
        if not baseline_means[role]:
            print(f"\n{role}: no baseline data")
            continue
        bm = np.array(baseline_means[role])
        bs = np.array(baseline_stds[role])
        print(f"\n=== {role} ===")
        print(f"  Baseline (n={len(bm)}): mean ||h|| = {bm.mean():.2f} ± {bm.std():.2f}, "
              f"within-sample std = {bs.mean():.2f} ± {bs.std():.2f}")
        for idx in TARGET_IDX:
            if idx not in fixed3[role]:
                continue
            m, s = fixed3[role][idx]
            mean_z = (m - bm.mean()) / bm.std() if bm.std() > 0 else float('nan')
            std_z = (s - bs.mean()) / bs.std() if bs.std() > 0 else float('nan')
            print(f"  idx={idx}: mean={m:.2f} (z={mean_z:+.2f}), "
                  f"within-sample std={s:.2f} (z={std_z:+.2f})")

    print("\n" + "=" * 70)
    print("Verdict heuristic")
    print("=" * 70)
    print("Mean z-score:")
    print("  |z| < 2  →  fixed-3 magnitude IS within baseline → not a magnitude signature")
    print("  |z| ≥ 2  →  fixed-3 magnitude IS anomalous (supports H1a)")
    print("\nStd z-score:")
    print("  z ≥ 2    →  fixed-3 is MORE volatile (supports chaotic-attractor hypothesis)")
    print("  z < 2    →  fixed-3 volatility within normal range (supports generation-side)")


if __name__ == "__main__":
    main()
