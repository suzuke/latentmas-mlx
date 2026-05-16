# Baseline GSM8K 50-sample Audit — Preliminary Analysis

**Date**: 2026-05-16 / 17 (cross-midnight)
**Model**: Qwen3-4B-bf16
**Samples**: 50 GSM8K test
**max_tokens**: 2048

## Headline result

| Metric | Value |
|--------|-------|
| Accuracy | 46/50 = 92.0% |
| Wall time | 1316s total, 26.3s avg |
| Avg out_tokens | 989.8 |

## Failure analysis: **100% of failures are truncation, not reasoning**

All 4 wrong samples have **out_tokens = exactly 2048** (the cap).
All 4 lack `\boxed{}` in the response (truncated before answer).
All 4 end mid-sentence.

| index | gold | extracted (fallback) | last 50 chars |
|-------|------|---------------------|---------------|
| 7 | 160 | 20 | `+ 20 minutes for the restart +` |
| 12 | 13 | 7.50 | `the initial cost is separate` |
| 37 | 2 | 5 | `the total money from selling Legos is` |
| 45 | 104 | 4 | `Wednesday is twice that, so 4.` |

Max successful out_tokens was **1995** — no successful sample even
came close to the 2048 cap. So 2048 is exactly the threshold where
samples either fit or get cut off.

## Why this matters for the original -1.8pp claim

The reported -1.8pp gap (94.0% baseline → 92.2% LatentMAS on full 1319
samples) was computed with same max_tokens=2048 for both methods. But:

1. **Baseline reasoning fits in N tokens directly**
2. **LatentMAS judger** reasons AFTER receiving 3 latent agents' KV
   cache context. The judger prompt includes:
   > "The latent information might contain irrelevant contents.
   >  Ignore it if it is not helpful..."
   This *may* induce longer judger reasoning to handle the context.

**Hypothesis (testable when latent_mas run completes)**:
> A significant portion of the -1.8pp gap is **token-budget exhaustion
> in the LatentMAS judger**, not reasoning quality. If true, raising
> max_tokens to e.g. 4096 would close the gap or invert it.

## What this finding implies for the broader RESULTS.md

The same hypothesis likely applies to other benchmarks where LatentMAS
underperformed:

| Benchmark | Reported gap | Likely truncation effect? |
|-----------|-------------|--------------------------|
| GSM8K full (-1.8pp) | This audit |
| ARC-Challenge (-6pp) | Multiple-choice, shorter responses needed → maybe less |
| GPQA (-6pp) | Long context + complex reasoning → **likely high truncation** |
| OBF + GPQA (-20pp) | OBF compresses context → judger has less to work with → likely **even more truncation** |
| Gemma E4B (-14pp) | Weak math → many failures → **many truncations** |
| Activation Graft (-43pp) | Complete method failure (probably not truncation) |

If truncation explains most of the small-medium gaps, the **real**
failure modes worth studying narrow to:

- Activation Graft (-43pp): genuine method break
- OBF + GPQA: compression-induced reasoning degradation (vs. just truncation)

## What to do once latent_mas data arrives

1. Compute LatentMAS failure rate
2. Cross-tab failure × truncation:
   ```
   |              | Baseline | LatentMAS |
   | trunc fails  |    ?     |     ?     |
   | non-trunc fails |  ?     |     ?     |
   ```
3. If LatentMAS has MORE truncations → main finding confirmed
4. If LatentMAS has SAME truncations → -1.8pp is real reasoning gap
5. Plan re-run with max_tokens=4096 to settle
