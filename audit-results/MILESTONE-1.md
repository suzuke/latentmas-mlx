# Milestone Report 1 — GSM8K Audit Reveals Three Distinct LatentMAS Failure Modes

**Date**: 2026-05-17
**Branch**: `audit/gsm8k-extraction`
**Setup**: Qwen3-4B bf16, 50 GSM8K samples, max_tokens=2048, temp=0.6

---

## Bottom line

The "-1.8pp" gap reported in RESULTS.md for full GSM8K (94.0% baseline vs 92.2% LatentMAS)
**hides a more serious structural problem**. With apples-to-apples comparison
(same framework, same precision, same model) on a 50-sample audit:

| | Baseline | LatentMAS |
|---|---|---|
| Accuracy | **92.0%** | **84.0%** |
| Gap | — | **-8.0 pp** |
| Reasoning-only accuracy* | **100.0%** | 84.0% |
| Reasoning-only gap | — | **-16.0 pp** |

\* Excluding samples that hit max_tokens (those are token-budget artifacts, not reasoning failures).

The RESULTS.md -1.8pp number compared **PyTorch FP16 baseline** with **MLX bf16 LatentMAS**.
The apples-to-apples MLX-bf16-both comparison shows a much larger gap.

---

## Three distinct failure modes (the real finding)

The 8 LatentMAS failures decompose into:

### Mode A — Mode collapse (degenerate generation) — 3/8

LatentMAS produces sequences that baseline never produces:

| idx | tokens | whitespace_% | pattern |
|-----|--------|-------------|---------|
| 11 | 24 | **96.2%** | `<think>` + hundreds of `\n` + `\boxed{456}` (wrong) |
| 20 | 1798 | 39.6% | `... 3 3 3 3 3 3 3 3 3 ...` (single-token loop) |
| 40 | 1986 | 4.6% | `...DocumentDocumentDocument...` (single-token loop) |

These are **clear catastrophic generation failures**, not subtle reasoning errors.
The latent KV cache appears to be putting the model into a state where its
output distribution collapses to repetitive or near-empty sequences.

### Mode B — Reasoning errors with valid-looking output — 5/8

Indices 7, 12, 24, 37, 41 — model writes plausible reasoning but reaches
a wrong answer. These look like normal model errors.

### Mode C — None of LatentMAS failures are truncation

**0/8 LatentMAS failures hit max_tokens.** Compare baseline:
**4/4 baseline failures are exactly 2048 tokens (truncation).**

The two methods fail in **completely different ways**:
- Baseline: tokens run out before reasoning completes → truncation
- LatentMAS: tokens are spent on degenerate output OR concluded reasoning is wrong

---

## What the comparison table really shows

Head-to-head on the same 50 samples:

| Outcome | Count |
|---------|-------|
| Both correct | 41 |
| Both wrong | 3 |
| Only baseline correct | **5** |
| Only LatentMAS correct | **1** |

The "only LatentMAS correct" case is idx=45, where baseline truncated at 2048
tokens. LatentMAS finished in 461 tokens (faster reasoning, no truncation).

**LatentMAS wins 1 case via avoiding truncation, but loses 5 cases via worse reasoning.**
Net: -8pp.

---

## Why this matters

### 1. The LatentMAS paper's "lossless transfer" claim is contradicted

The paper (arxiv 2511.20639) abstract states LatentMAS achieves
"lossless information preservation". This audit shows:

- 37.5% of LatentMAS failures (3/8) involve **catastrophic generation collapse**
  that does not exist in baseline.
- This is qualitative evidence of **information distortion**, not preservation.

### 2. The "1.9× speedup" needs a qualifier

LatentMAS produces shorter outputs (avg 587 tokens vs baseline 989 tokens).
Some of this is legitimate efficiency. But some is degenerate generation
(idx=11 finished in 24 tokens — fastest "sample", with garbage output).

The speedup metric should be paired with **completion-of-reasoning rate**
or **mode-collapse rate**.

### 3. The -1.8pp number in RESULTS.md is itself misleading

Comparing PyTorch FP16 baseline (94%) with MLX bf16 LatentMAS (92.2%)
mixes two confounds:
- Different inference framework (PyTorch MPS vs MLX)
- Different numerical precision (FP16 vs bf16)

When controlled (both MLX bf16), gap widens to -8pp on this 50-sample subset.

---

## Implications for next research

### 4 candidate directions (in priority order)

1. **Replicate at scale**: Run 200-500 samples for tighter confidence interval
   on the -8pp gap. If the gap holds, this single finding suffices for a
   short paper.

2. **Characterize mode-collapse trigger**: For idx=11/20/40,
   what's different about the KV cache that pushes the model into degenerate
   modes? Is it the latent agent count? The compression level? The
   specific question structure?

3. **Mode-collapse rate across benchmarks**: Re-audit GPQA, ARC, math500.
   Is the 37.5%-of-failures collapse rate stable, or does it scale with
   task difficulty? (This explains the catastrophic -20pp on GPQA+OBF.)

4. **Cure attempt**: Can we detect mode-collapse in real-time (entropy
   threshold? repetition detection?) and fall back to baseline? This would
   give a **practical fix**, not just a critique.

### Recommended direction: 1 + 2

**1 establishes the finding firmly. 2 makes it explanatory rather than
just descriptive.** Together they're a 2-month publishable contribution.

---

## Caveats and limitations

- **n=50 only**: This is a pilot. The -8pp gap and 37.5% mode-collapse rate
  need replication at higher n.
- **One model**: Qwen3-4B only. Different models (Gemma, LLaMA) may have
  different mode-collapse propensity. The original RESULTS.md showed
  worse degradation on Gemma 4 — possibly the same phenomenon.
- **Temperature=0.6**: Sampling-induced. At temp=0.0 (greedy) the
  mode-collapse may be deterministic or vanish; not yet tested.
- **No probe analysis yet**: I haven't shown _why_ the latent KV
  triggers collapse — just that it does.

---

## Files in this milestone

```
audit-results/
├── baseline-gsm8k-50-bf16.jsonl   # 51 lines (1 meta + 50 samples)
├── latent_mas-gsm8k-50-bf16.jsonl # 51 lines
├── baseline.log                    # stdout
├── latent_mas.log
├── MILESTONE-1.md                  # this file
└── notes/
    ├── baseline-analysis.md
    ├── comm-activations-suspected-bug.md
    └── compare.py
```

## Reproducibility

```bash
cd ~/research/latentmas-mlx-audit
source .venv/bin/activate

# Baseline
python latentmas/run.py \
  --method baseline --model mlx-community/Qwen3-4B-bf16 \
  --task gsm8k --max_samples 50 --max_tokens 2048 \
  --save_outputs audit-results/baseline-gsm8k-50-bf16.jsonl

# LatentMAS
python latentmas/run.py \
  --method latent_mas --model mlx-community/Qwen3-4B-bf16 \
  --task gsm8k --max_samples 50 --max_tokens 2048 \
  --save_outputs audit-results/latent_mas-gsm8k-50-bf16.jsonl

# Compare
python audit-results/notes/compare.py
```

Both runs at git commit `270396c` on branch `audit/gsm8k-extraction`.
