"""Compare RecursiveMAS broken (no-fix) vs fixed (with norm) on MATH-500.

Usage:  uv run python audit-results/notes/compare_recursivemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(__file__).parent.parent

FILES = {
    "broken": AUDIT / "recursive_mas-math500-30-light-broken.jsonl",
    "fixed":  AUDIT / "recursive_mas-math500-30-light-fixed.jsonl",
}


def load(path):
    if not path.exists():
        return None, []
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


def main():
    rows = []
    for label, path in FILES.items():
        meta, samples = load(path)
        if meta is None:
            print(f"[MISSING] {label}: {path.name}")
            continue
        n = len(samples)
        correct = sum(1 for s in samples if s["correct"])
        avg_time = sum(s["elapsed_sec"] for s in samples) / n if n else 0
        rows.append({
            "label": label, "n": n, "correct": correct,
            "acc": correct / n if n else 0, "avg_time": avg_time,
        })

    if not rows:
        print("No data yet")
        return

    print(f"\n{'metric':<15} | " + " | ".join(f"{r['label']:>12}" for r in rows))
    print("-" * (18 + 15 * len(rows)))
    print(f"{'n':<15} | " + " | ".join(f"{r['n']:>12}" for r in rows))
    print(f"{'accuracy':<15} | " + " | ".join(f"{r['acc']:>11.1%}" for r in rows))
    print(f"{'correct':<15} | " + " | ".join(f"{r['correct']:>12}" for r in rows))
    print(f"{'avg_time_sec':<15} | " + " | ".join(f"{r['avg_time']:>12.1f}" for r in rows))

    # Head-to-head if both available
    if len(rows) == 2:
        broken, fixed = load(FILES["broken"])[1], load(FILES["fixed"])[1]
        bd = {s["index"]: s for s in broken}
        fd = {s["index"]: s for s in fixed}
        shared = sorted(set(bd) & set(fd))
        both_ok = sum(1 for i in shared if bd[i]["correct"] and fd[i]["correct"])
        only_broken = sum(1 for i in shared if bd[i]["correct"] and not fd[i]["correct"])
        only_fixed = sum(1 for i in shared if not bd[i]["correct"] and fd[i]["correct"])
        both_wrong = sum(1 for i in shared if not bd[i]["correct"] and not fd[i]["correct"])

        print()
        print(f"Head-to-head ({len(shared)} shared):")
        print(f"  Both correct:           {both_ok}")
        print(f"  Both wrong:             {both_wrong}")
        print(f"  Only BROKEN correct:    {only_broken}")
        print(f"  Only FIXED correct:     {only_fixed}")
        print()

        if only_fixed:
            print(f"=== Samples FIXED recovered (broken wrong → fixed correct) ===")
            for i in shared:
                if not bd[i]["correct"] and fd[i]["correct"]:
                    q = bd[i]["question"][:80].replace("\n", " ")
                    print(f"  idx={i} gold={bd[i]['gold']!r} broken={bd[i]['extracted_pred']!r} → fixed={fd[i]['extracted_pred']!r}")
                    print(f"     Q: {q}...")


if __name__ == "__main__":
    main()
