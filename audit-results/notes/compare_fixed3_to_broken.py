"""Compare fixed-3 collapse trajectories to broken-code baseline (from MILESTONE-12).

Loads:
- 3 fixed-code .npz from audit-results/activations-fixed-3collapse/
- 100 broken-code .npz from audit-results/activations-broken/
- BROKEN-cap jsonl for correctness labels

Output: per-sample ||h|| trajectory comparison + verdict on
mechanism taxonomy (latent vs generation-side).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
BROKEN_NPZ = ROOT / "audit-results" / "activations-broken"
FIXED_NPZ = ROOT / "audit-results" / "activations-fixed-3collapse"
BROKEN_JSONL = ROOT / "audit-results" / "latent_mas-gsm8k-100-bf16-BROKEN-cap.jsonl"
FIXED_JSONL = ROOT / "audit-results" / "latent_mas-gsm8k-fixed-3collapse-cap.jsonl"

TARGET_IDX = [396, 406, 952]


def load_broken_norms(role: str) -> tuple[np.ndarray, list[bool]]:
    """Return [n_samples, n_steps] norms and correctness mask for broken-code."""
    norms_list = []
    corrects = []
    with open(BROKEN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("_meta"):
                continue
            idx = d["index"]
            npz_path = BROKEN_NPZ / f"sample_{idx:05d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)[role]
            if arr.shape[0] == 0:
                continue
            norms_list.append(np.linalg.norm(arr, axis=-1))
            corrects.append(d["correct"])
    return np.array(norms_list), corrects


def load_fixed_norms(role: str) -> dict[int, np.ndarray]:
    """Return {idx: [n_steps]} norm trajectories for fixed-3."""
    out = {}
    for idx in TARGET_IDX:
        npz_path = FIXED_NPZ / f"sample_{idx:05d}.npz"
        if not npz_path.exists():
            print(f"  [MISSING] {npz_path}")
            continue
        arr = np.load(npz_path)[role]
        if arr.shape[0] == 0:
            continue
        out[idx] = np.linalg.norm(arr, axis=-1)
    return out


def main():
    print("=" * 60)
    print("Fixed-3 vs Broken-100 ||h|| trajectory comparison")
    print("=" * 60)

    # Load fixed-3 response info
    fixed_info = {}
    if FIXED_JSONL.exists():
        with open(FIXED_JSONL) as f:
            for line in f:
                d = json.loads(line)
                fixed_info[d["index"]] = {
                    "tokens": d["out_tokens"],
                    "tail": d["raw_response"][-150:],
                }

    print(f"\nFixed-3 samples captured: {sorted(fixed_info.keys())}")
    for idx, info in sorted(fixed_info.items()):
        print(f"  idx={idx}: tokens={info['tokens']}")
        print(f"    tail: {info['tail']!r}")

    print("\n" + "-" * 60)
    print("Per-role trajectory stats")
    print("-" * 60)

    for role in ["planner", "critic", "refiner"]:
        print(f"\n=== {role} ===")
        broken_norms, broken_correct = load_broken_norms(role)
        fixed_norms = load_fixed_norms(role)

        if broken_norms.shape[0] == 0:
            print("  No broken data available!")
            continue

        bk_mean = broken_norms.mean(axis=0)
        bk_std = broken_norms.std(axis=0)
        print(f"  Broken (n={broken_norms.shape[0]}): "
              f"mean over steps = {broken_norms.mean():.2f} ± {broken_norms.std():.2f}, "
              f"range [{broken_norms.min():.1f}, {broken_norms.max():.1f}]")
        print(f"    per-step mean: step 0={bk_mean[0]:.1f}, "
              f"step {len(bk_mean)//2}={bk_mean[len(bk_mean)//2]:.1f}, "
              f"step -1={bk_mean[-1]:.1f}")

        for idx in TARGET_IDX:
            if idx not in fixed_norms:
                continue
            trace = fixed_norms[idx]
            within_band = ((trace >= bk_mean - 2 * bk_std) &
                           (trace <= bk_mean + 2 * bk_std)).mean()
            print(f"  Fixed idx={idx}: mean={trace.mean():.2f}, "
                  f"range [{trace.min():.1f}, {trace.max():.1f}], "
                  f"within ±2σ of broken: {100*within_band:.0f}%")
            print(f"    trace: {[f'{v:.0f}' for v in trace[::5].tolist()]}")

    # Verdict heuristic
    print("\n" + "=" * 60)
    print("Mechanism taxonomy verdict (per MILESTONE-12 H1a/H1b)")
    print("=" * 60)
    print("If fixed-3 trajectories are within ±2σ of broken-100 mean:")
    print("  → consistent with 'generation-side' (latent state normal)")
    print("If fixed-3 are anomalous (outside band, or different shape):")
    print("  → suggests prompt-specific latent pathology (different mechanism)")
    print("\nManual interpretation required based on numbers above.")


if __name__ == "__main__":
    main()
