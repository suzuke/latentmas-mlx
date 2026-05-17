"""Three-way comparison: baseline vs latent_mas vs latent_mas (--no_compress).

Tabulates accuracy, truncation, mode-collapse rates across conditions.

Usage:  uv run python audit-results/notes/compare3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AUDIT = Path(__file__).parent.parent

FILES = {
    "base_bf16":         AUDIT / "baseline-gsm8k-50-bf16.jsonl",
    "lat_bf16_obf":      AUDIT / "latent_mas-gsm8k-50-bf16.jsonl",
    "lat_bf16_nocomp":   AUDIT / "latent_mas-gsm8k-50-bf16-nocompress.jsonl",
    "base_4bit":         AUDIT / "baseline-gsm8k-50-4bit.jsonl",
    "lat_4bit_obf":      AUDIT / "latent_mas-gsm8k-50-4bit.jsonl",
    "lat_bf16_FIXED":    AUDIT / "latent_mas-gsm8k-50-bf16-fixed.jsonl",
}


def load(path: Path) -> tuple[dict, list[dict]]:
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
    """Return (is_collapse, reason)."""
    resp = s["raw_response"]
    n_chars = len(resp)
    if n_chars == 0:
        return True, "empty"

    # Whitespace-dominant
    n_ws = sum(1 for c in resp if c.isspace())
    if n_ws / n_chars > 0.7:
        return True, f"whitespace_{int(n_ws*100/n_chars)}%"

    # Token repetition: same 4-gram appearing >50 times
    words = resp.split()
    if len(words) >= 100:
        from collections import Counter
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if ngrams:
            top = Counter(ngrams).most_common(1)[0]
            if top[1] > 30:
                return True, f"4gram_repeat_{top[1]}x"

    # Single-token repetition: same word > 100 times consecutively
    if len(words) >= 100:
        for i in range(len(words) - 99):
            chunk = words[i:i+100]
            if len(set(chunk)) <= 3:  # 100 words with <=3 unique
                return True, f"token_loop_at_word_{i}"

    return False, ""


def stats(label: str, meta: dict, samples: list[dict]):
    if not samples:
        return None
    max_t = meta["max_tokens"]
    n = len(samples)
    correct = sum(1 for s in samples if s["correct"])

    truncated = [s for s in samples if s["out_tokens"] >= max_t]
    fails = [s for s in samples if not s["correct"]]
    fail_trunc = [s for s in fails if s["out_tokens"] >= max_t]
    fail_no_trunc = [s for s in fails if s["out_tokens"] < max_t]

    # Mode collapse detection on FAILED non-truncated
    collapses = []
    for s in fail_no_trunc:
        is_col, reason = detect_mode_collapse(s)
        if is_col:
            collapses.append((s["index"], reason, s["out_tokens"]))

    return {
        "label": label,
        "n": n,
        "correct": correct,
        "acc": correct / n,
        "fails": len(fails),
        "fail_trunc": len(fail_trunc),
        "fail_no_trunc": len(fail_no_trunc),
        "mode_collapse": len(collapses),
        "collapse_detail": collapses,
        "avg_tokens": sum(s["out_tokens"] for s in samples) / n,
    }


def main() -> None:
    rows = []
    for label, path in FILES.items():
        meta, samples = load(path)
        if meta is None:
            print(f"  [MISSING] {label}: {path.name}")
            continue
        r = stats(label, meta, samples)
        rows.append(r)

    # Header table
    print()
    print(f"{'metric':<25} | " + " | ".join(f"{r['label']:>20}" for r in rows))
    print("-" * (28 + 23 * len(rows)))
    print(f"{'n':<25} | " + " | ".join(f"{r['n']:>20}" for r in rows))
    print(f"{'accuracy':<25} | " + " | ".join(f"{r['acc']:>19.1%}" for r in rows))
    print(f"{'avg_tokens':<25} | " + " | ".join(f"{r['avg_tokens']:>20.1f}" for r in rows))
    print(f"{'failures':<25} | " + " | ".join(f"{r['fails']:>20}" for r in rows))
    print(f"  of which truncated     | " + " | ".join(f"{r['fail_trunc']:>20}" for r in rows))
    print(f"  of which mode collapse | " + " | ".join(f"{r['mode_collapse']:>20}" for r in rows))
    print(f"  of which other(reasoning) | " + " | ".join(f"{r['fail_no_trunc']-r['mode_collapse']:>17}" for r in rows))

    # Detailed collapse cases per condition
    print()
    print("=" * 70)
    print("Mode collapse detail")
    print("=" * 70)
    for r in rows:
        if r["mode_collapse"]:
            print(f"\n{r['label']}:")
            for idx, reason, tokens in r["collapse_detail"]:
                print(f"  idx={idx} tokens={tokens} reason={reason}")
        else:
            print(f"\n{r['label']}: (no mode collapse)")


if __name__ == "__main__":
    main()
