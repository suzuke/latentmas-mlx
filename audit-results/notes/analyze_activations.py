"""Analyze captured activations for MILESTONE-12 (mode collapse precursor study).

Loads per-sample .npz files + corresponding JSONL row, computes ||h||
trajectories per agent per step, and classifies samples by collapse status.

Usage:
    python audit-results/notes/analyze_activations.py \\
        --jsonl audit-results/latent_mas-gsm8k-100-bf16-BROKEN-cap.jsonl \\
        --npz_dir audit-results/activations-broken \\
        --label BROKEN

Output:
    - summary table to stdout
    - audit-results/notes/activation-analysis-{label}.json with stats
    - audit-results/notes/h_norm-trajectories-{label}.png (if matplotlib)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def detect_mode_collapse(raw_response: str) -> tuple[bool, str]:
    """Heuristic detector matching the audit's compareN.py."""
    n_chars = len(raw_response)
    if n_chars == 0:
        return True, "empty"
    n_ws = sum(1 for c in raw_response if c.isspace())
    if n_chars > 100 and n_ws / n_chars > 0.7:
        return True, f"ws_{int(n_ws*100/n_chars)}%"
    words = raw_response.split()
    if len(words) >= 100:
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if ngrams:
            top = Counter(ngrams).most_common(1)[0]
            if top[1] > 30:
                return True, f"4gram_x{top[1]}"
    return False, ""


def load_run(jsonl_path: Path, npz_dir: Path) -> list[dict]:
    """Load all samples with both jsonl row and activation .npz."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("_meta"):
                continue
            idx = d["index"]
            npz_path = npz_dir / f"sample_{idx:05d}.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path, allow_pickle=False)
            samples.append({
                "index": idx,
                "correct": d["correct"],
                "raw_response": d["raw_response"],
                "out_tokens": d["out_tokens"],
                "planner": npz["planner"],   # shape [n_steps, hidden_dim]
                "critic": npz["critic"],
                "refiner": npz["refiner"],
            })
    return samples


def classify_samples(samples: list[dict]) -> tuple[list, list]:
    """Split into mode-collapse vs normal."""
    collapses, normals = [], []
    for s in samples:
        is_collapse, reason = detect_mode_collapse(s["raw_response"])
        # Also consider truncations as separate (not pure collapse, not normal)
        # For MILESTONE-12, focus on non-truncated collapses
        if is_collapse and s["out_tokens"] < 2048:
            s["collapse_reason"] = reason
            collapses.append(s)
        elif not is_collapse and s["correct"]:
            # only count truly-correct as normal (not just non-collapse-non-trunc)
            normals.append(s)
    return collapses, normals


def trajectory_stats(samples: list[dict], role: str) -> dict:
    """For a group, compute mean and std of ||h_t|| at each step for the given role."""
    if not samples:
        return {"n": 0}
    arrs = [s[role] for s in samples if s[role].shape[0] > 0]
    if not arrs:
        return {"n": 0}
    # Each arr shape: [n_steps, hidden_dim]
    norms = np.array([np.linalg.norm(a, axis=-1) for a in arrs])  # [n_samples, n_steps]
    return {
        "n": len(arrs),
        "n_steps": norms.shape[1],
        "mean": norms.mean(axis=0).tolist(),
        "std": norms.std(axis=0).tolist(),
        "max": norms.max(axis=0).tolist(),
        "min": norms.min(axis=0).tolist(),
        # also report mean over steps to compare averages
        "global_mean": float(norms.mean()),
        "global_max": float(norms.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--npz_dir", required=True)
    p.add_argument("--label", default="run")
    args = p.parse_args()

    samples = load_run(Path(args.jsonl), Path(args.npz_dir))
    print(f"Loaded {len(samples)} samples with activations from {args.label}")

    collapses, normals = classify_samples(samples)
    print(f"  Collapses (non-trunc): {len(collapses)}")
    print(f"  Normals (correct, non-collapse, non-trunc): {len(normals)}")

    summary = {"label": args.label, "n_total": len(samples),
               "n_collapses": len(collapses), "n_normals": len(normals)}

    for role in ["planner", "critic", "refiner"]:
        col_stats = trajectory_stats(collapses, role)
        nor_stats = trajectory_stats(normals, role)
        summary[role] = {"collapses": col_stats, "normals": nor_stats}
        print(f"\n=== {role} ===")
        if col_stats.get("n", 0) > 0 and nor_stats.get("n", 0) > 0:
            print(f"  Collapses (n={col_stats['n']}): mean ||h|| = {col_stats['global_mean']:.3f}, max = {col_stats['global_max']:.3f}")
            print(f"  Normals  (n={nor_stats['n']}): mean ||h|| = {nor_stats['global_mean']:.3f}, max = {nor_stats['global_max']:.3f}")
            # per-step comparison at a few key steps
            print(f"  ||h|| at steps 0, 10, 20, 30, last:")
            for s in [0, 10, 20, 30, col_stats["n_steps"]-1]:
                if 0 <= s < col_stats["n_steps"]:
                    print(f"    step {s}: collapse mean = {col_stats['mean'][s]:.3f}, normal mean = {nor_stats['mean'][s]:.3f}")
        else:
            print(f"  Insufficient data: collapses={col_stats.get('n',0)}, normals={nor_stats.get('n',0)}")

    # Save raw stats to JSON for further analysis / plotting
    out_path = Path(args.jsonl).parent / "notes" / f"activation-analysis-{args.label}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    # Print collapse details
    if collapses:
        print(f"\n=== Collapse samples ===")
        for s in collapses:
            print(f"  idx={s['index']} reason={s['collapse_reason']} tokens={s['out_tokens']}")


if __name__ == "__main__":
    main()
