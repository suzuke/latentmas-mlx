"""Capture latent activations on the 3 fixed-code collapse samples
(idx 396, 406, 952 from latent_mas-gsm8k-1319-bf16-fixed.jsonl).

Goal: validate the MILESTONE-12 hypothesis that fixed-code collapses are
generation-side (autoregressive token-attractor) rather than latent-side
pathology, by comparing their ||h|| trajectory to the broken-code baseline.

Note: original 1319 run was temp=0.6 stochastic. RNG state will differ
here so output text may not exactly reproduce the original collapse. The
trajectory itself for the same prompt is the data of interest — if the
prompt-induced latent state is anomalous regardless of sampling, we'd
see signal in the captured trajectory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

import mlx_lm

from latentmas.run import run_latent_mas, load_data

TARGET_IDX = [396, 406, 952]
MODEL = "mlx-community/Qwen3-4B-bf16"
TASK = "gsm8k"
MAX_TOKENS = 2048
N_LATENT = 40

OUT_DIR = ROOT / "audit-results" / "activations-fixed-3collapse"
OUT_JSONL = ROOT / "audit-results" / "latent_mas-gsm8k-fixed-3collapse-cap.jsonl"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[capture] model={MODEL}")
    model, tok = mlx_lm.load(MODEL)
    print(f"[capture] loading task={TASK} max_samples=1000")
    data = load_data(TASK, 1000)
    print(f"[capture] loaded {len(data)} GSM8K samples, target idx={TARGET_IDX}")

    with open(OUT_JSONL, "w") as f:
        for idx in TARGET_IDX:
            item = data[idx]
            print(f"\n[capture] === idx={idx} === q={item['question'][:80]!r}")
            buf: dict = {}
            try:
                resp, _ = run_latent_mas(
                    model, tok, item["question"], TASK, MAX_TOKENS, N_LATENT,
                    capture_buffer=buf,
                )
            except Exception as e:
                print(f"[capture] idx={idx} FAILED: {e!r}")
                continue
            out_tokens = len(tok.encode(resp))
            print(f"[capture] idx={idx} tokens={out_tokens} resp_tail={resp[-120:]!r}")

            npz_path = OUT_DIR / f"sample_{idx:05d}.npz"
            np.savez(
                npz_path,
                planner=np.array(buf.get("planner", [])),
                critic=np.array(buf.get("critic", [])),
                refiner=np.array(buf.get("refiner", [])),
                index=idx,
                out_tokens=out_tokens,
            )
            f.write(json.dumps({
                "index": idx,
                "question": item["question"],
                "gold": item["gold"],
                "raw_response": resp,
                "out_tokens": out_tokens,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[capture] idx={idx} saved to {npz_path}")


if __name__ == "__main__":
    main()
