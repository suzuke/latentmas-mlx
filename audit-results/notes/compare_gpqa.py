"""Compare GPQA-Diamond baseline vs latent_mas (FIXED only) on Qwen3-4B 4-bit.

Tabulates accuracy, truncation, mode-collapse rates.

Note: We don't have a "broken" latent_mas GPQA run to compare directly;
RESULTS.md reports 60% latent vs 66% baseline on Gemma 26B (-6pp), and
17%/23%/37% (OBF/no-comp/baseline) on a different setup (Qwen3-4B likely).

The point of this experiment is: does the fix bring latent_mas to or
above baseline on a benchmark where it previously regressed?

Usage:  uv run python audit-results/notes/compare_gpqa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AUDIT = Path(__file__).parent.parent

FILES = {
    "base_4bit":         AUDIT / "baseline-gpqa-30-4bit.jsonl",
    "lat_4bit_FIXED":    AUDIT / "latent_mas-gpqa-30-4bit-fixed.jsonl",
}


def load(path: Path):
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


def detect_mode_collapse(s: dict) -> tuple[bool, str]:
    resp = s["raw_response"]
    n_chars = len(resp)
    if n_chars == 0:
        return True, "empty"
    n_ws = sum(1 for c in resp if c.isspace())
    if n_chars > 100 and n_ws / n_chars > 0.7:
        return True, f"whitespace_{int(n_ws*100/n_chars)}%"
    words = resp.split()
    if len(words) >= 100:
        from collections import Counter
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if ngrams:
            top = Counter(ngrams).most_common(1)[0]
            if top[1] > 30:
                return True, f"4gram_repeat_{top[1]}x"
        for i in range(0, len(words) - 99):
            chunk = words[i:i+100]
            if len(set(chunk)) <= 3:
                return True, f"token_loop"
                break
    return False, ""


def main() -> None:
    rows = []
    for label, path in FILES.items():
        meta, samples = load(path)
        if meta is None:
            print(f"  [MISSING] {label}: {path.name}")
            continue
        max_t = meta["max_tokens"]
        n = len(samples)
        correct = sum(1 for s in samples if s["correct"])
        truncated = sum(1 for s in samples if s["out_tokens"] >= max_t)
        fails = [s for s in samples if not s["correct"]]
        fail_trunc = sum(1 for s in fails if s["out_tokens"] >= max_t)
        fail_no_trunc = [s for s in fails if s["out_tokens"] < max_t]
        collapses = []
        for s in fail_no_trunc:
            col, reason = detect_mode_collapse(s)
            if col:
                collapses.append((s["index"], reason, s["out_tokens"]))
        rows.append({
            "label": label,
            "n": n,
            "correct": correct,
            "acc": correct / n,
            "fails": len(fails),
            "fail_trunc": fail_trunc,
            "mc": len(collapses),
            "collapse_detail": collapses,
            "avg_tokens": sum(s["out_tokens"] for s in samples) / n,
        })

    if not rows:
        print("No data yet")
        return

    print(f"\n{'metric':<25} | " + " | ".join(f"{r['label']:>16}" for r in rows))
    print("-" * (28 + 19 * len(rows)))
    print(f"{'n':<25} | " + " | ".join(f"{r['n']:>16}" for r in rows))
    print(f"{'accuracy':<25} | " + " | ".join(f"{r['acc']:>15.1%}" for r in rows))
    print(f"{'avg_tokens':<25} | " + " | ".join(f"{r['avg_tokens']:>16.1f}" for r in rows))
    print(f"{'failures':<25} | " + " | ".join(f"{r['fails']:>16}" for r in rows))
    print(f"{'  truncated':<25} | " + " | ".join(f"{r['fail_trunc']:>16}" for r in rows))
    print(f"{'  mode collapse':<25} | " + " | ".join(f"{r['mc']:>16}" for r in rows))

    print()
    print("=" * 60)
    print("Mode collapse detail")
    print("=" * 60)
    for r in rows:
        if r["mc"]:
            print(f"\n{r['label']}:")
            for idx, reason, tokens in r["collapse_detail"]:
                print(f"  idx={idx} tokens={tokens} reason={reason}")
        else:
            print(f"\n{r['label']}: (no mode collapse)")


if __name__ == "__main__":
    main()
