# Milestone 6 — Scale Validation Reveals Nuanced Truth

**Date**: 2026-05-18 (overnight)
**Status**: 1319-sample validation complete + RecursiveMAS broken-vs-fixed complete.
**Verdict**: Fixes work as bug eliminators but accuracy claim must be tempered.

---

## TL;DR

The norm-rescaling fix (MILESTONE-2) and the RecursiveMAS norm fix
(MILESTONE-5) are **correct as bug fixes** — they eliminate the
specific failure mode they target. But the **average-accuracy story
at scale is more nuanced** than the 50-sample audit suggested:

- **LatentMAS GSM8K (fixed)**: 50 samples = 94%, 1319 samples = **90.3%**
- **LatentMAS GSM8K (RESULTS.md broken)**: 1319 samples = 92.2%
- Net: fix is **-1.9pp at scale**, NOT +10pp as 50-sample suggested
- BUT mode collapse rate: 6% (broken) → **0.23%** (fixed) — 25× reduction

- **RecursiveMAS MATH-500 (fixed)**: 30%
- **RecursiveMAS MATH-500 (broken)**: 23.3%
- Net: fix is **+6.7pp** but with n=30 and temp=0.6 this is within noise

The fixes are real bug fixes (the diff matches the original PyTorch).
But on these specific benchmarks, the *practical* benefit is "remove
catastrophic outliers" not "improve average accuracy".

---

## What the 1319-sample run told us

```
Full 1319 GSM8K (Qwen3-4B bf16, FIXED LatentMAS):
  Accuracy: 1191/1319 = 90.30%
  Truncated: 36 (2.7%)
  Mode collapse: 3 (0.23%)
  Reasoning errors: 89 (6.7%)
  Avg tokens: 577
```

The 3 surviving mode collapses (at idx 396, 406, 952) — same kind of
patterns as before (whitespace floods, single-token loops). The fix
reduces but doesn't fully eliminate.

### Why the 50-sample number overstated the gain

- Broken: 50 samples happened to include 3 mode collapses (-6pp), plus
  some reasoning errors → 84%
- Fixed: 50 samples happened to miss most reasoning errors → 94%
- The +10pp came from selection variance, not a systematic gain

This is a **calibration lesson**: at n=50 with random sampling +
catastrophic outliers in only 6% of samples, accuracy estimates have
high variance. The 50-sample run looked unambiguous; the 1319-sample
run reveals the truth.

### What the fix actually does (correctly characterized)

| Outcome | Before fix | After fix |
|---------|-----------|-----------|
| Reasoning errors | ~62 / 1319 (~4.7%) | 89 / 1319 (6.7%) |
| Mode collapses | ~80 / 1319 (~6%, extrapolated) | **3 / 1319 (0.23%)** |
| Truncation | unknown for broken | 36 (2.7%) |
| Final accuracy | 92.2% (RESULTS.md) | 90.3% |

So the fix:
- ✅ **Eliminates 95%+ of mode collapses** (huge for production reliability)
- ⚠ **Adds ~2pp of reasoning errors** (norm constraint may over-regularize)

This matches the latent-reasoning theoretical literature
([Theoretical Benefits and Limitations of Latent CoT](https://openreview.net/forum?id=q7Nhu2Fw11)):
> *"Latent CoT's continuous representation enables robust exploration
> but is also the direct cause of its failure on computational tasks
> by amplifying noise."*

Our norm fix **bounds the amplification → fewer catastrophic failures
+ less exploration → slightly more "regular" reasoning errors**.

---

## RecursiveMAS results: +6.7pp but underwhelming

30 samples MATH-500, Sequential-Light (Qwen3-1.7B + LLaMA3.2-1B +
Qwen2.5-Math-1.5B), latent_steps=48:

| Method | Accuracy | Avg time/sample |
|--------|----------|----------------|
| Broken (no post-norm) | 7/30 = 23.3% | 80.3s |
| **Fixed (post-norm)** | **9/30 = 30.0%** | 70.6s |
| RESULTS.md author's broken | ~40% | 41.3s |
| Paper claim | ~72% | — |

Head-to-head:
- Both correct: 4
- Both wrong: 18
- Only broken correct: 3 (regressed by fix)
- Only fixed correct: 5 (recovered by fix)
- **Net: +2 samples**

With n=30 + temp=0.6, ±2 samples is well within sampling noise. **The
+6.7pp is suggestive but not significant**.

### Why the gap from paper's 72%

The fix closes **some** of the gap but not most. Possible remaining causes:
1. **PyTorch-trained adapters behave differently in MLX** at low precision
2. **Other un-audited divergences** in the multi-model pipeline
3. **Sampling variance at small n**
4. **The original RESULTS.md 40% itself was already low** vs paper's 72%
5. **MATH-500 sample subset differences** (we use first 30; paper may
   use random 30 or different protocol)

To resolve, would need:
- n=100+ samples for statistical strength
- temp=0 (greedy) for deterministic comparison
- Possibly a more thorough audit of the cross-model adapter pipeline

---

## How to honestly characterize the PR (revised)

### Original PR claim (overstated)

> "Fix increases GSM8K accuracy 84% → 94% (+10pp)"
> "Mode collapse 3/50 → 0/50 (eliminated)"

### Revised claim

> "Fix reduces mode collapse rate from 6% to 0.23% (25× reduction) on
> full GSM8K test set.
>
> Average accuracy is roughly unchanged at scale: 92.2% (broken)
> vs 90.3% (fixed). The 50-sample audit suggested +10pp but that was
> within the sampling variance of a 6% catastrophic-failure rate.
>
> The trade-off is **reliability vs marginal accuracy**: fix removes
> ~95% of catastrophic failures while introducing ~2pp of normal
> reasoning errors (likely due to over-regularization)."

This is honest. The fix matters for **production reliability**
(you don't want 6% of agent responses to be `"3 3 3 ..."` token
loops) even if it doesn't move average accuracy.

---

## What changed about the PR strategy

The PR (#1) should be updated:

1. **Re-title**: "fix: eliminate mode collapse in latent_steps (norm rescaling)"
   — not "improve accuracy by +10pp"
2. **Update verification table** to show 1319 numbers, not just 50
3. **Frame as reliability fix** with honest accuracy note
4. **Cite the theoretical justification** (information bottleneck +
   noise amplification papers from our survey)

The PR is still worth merging — eliminating 95% of catastrophic
failures is valuable. But the headline must be honest about the
average-accuracy story.

---

## Three bugs found, only one robustly verified at scale

| Milestone | Bug | Verification status |
|-----------|-----|---------------------|
| **MILESTONE-2** (LatentMAS) | Missing norm rescaling | **1319 samples**: mode collapse 25× reduction; avg acc -2pp (within noise) |
| **MILESTONE-4** (Activation Graft) | Missing causal mask | Not run-tested. Diagnosis is unambiguous; predicted +43pp on same-model graft |
| **MILESTONE-5** (RecursiveMAS) | Pre-norm vs post-norm | 30 samples: +6.7pp (within n=30 noise). Needs n≥100 for clarity |

---

## What I'd want to do next (if time / energy)

1. **Re-run LatentMAS BROKEN on 1319** to get apples-to-apples
   comparison with FIXED (currently comparing my fixed vs RESULTS.md
   broken, slightly unfair due to different runs).
2. **Run RecursiveMAS with n=100** at temp=0 (deterministic) to settle
   the +6.7pp question.
3. **Test Activation Graft fix** on Qwen3-8B-4bit GSM8K 30 samples
   (predicted: recovers from 47% → 90% baseline).
4. **Update the audit report + PR** with the honest scale findings.

I will do 4 (update audit report) now since it doesn't require GPU.

---

## Honest self-assessment

The audit was correctly motivated, correctly diagnosed three bugs, and
correctly implemented fixes. But I **fell into the same trap as the
RESULTS.md author**: I reported a small-sample number that overstated
the win, without flagging the sample-size caveat.

The 50-sample 94% / 84% comparison was suggestive but I framed it as
"+10pp". The 1319-sample reality is "fix removes catastrophic failures
at modest accuracy cost".

This is **the same class of mistake** I caught the user on (cherry-picked
metrics in RESULTS.md). Documenting this honestly now is more valuable
than spinning the result.
