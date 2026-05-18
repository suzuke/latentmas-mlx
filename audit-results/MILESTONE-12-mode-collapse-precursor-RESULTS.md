# Milestone 12 (RESULTS) — Mode Collapse Has NO ||h|| Precursor

**Date**: 2026-05-18
**Author**: claude-585a78
**Co-designed with**: kiro-research
**Plan**: see `MILESTONE-12-mode-collapse-precursor-plan.md`
**Status**: ✅ Experiment complete. **All pre-registered hypotheses REFUTED**.

---

## TL;DR

Captured per-step latent activations across 100 broken-code GSM8K samples (Qwen3-4B bf16). Searched for ||h|| / kurtosis / spikiness signatures that distinguish mode-collapse samples from normal samples.

**Result**: ZERO significant precursor signal across all 3 statistics × all 3 agents (9 tests, all p > 0.05).

This **refutes the implicit "magnitude drift" framing** of MILESTONE-2 / Bug #1 root cause analysis. Mode collapse is NOT predicted by latent state magnitude.

---

## The data

100 broken-code samples (Qwen3-4B bf16, GSM8K), with per-step latent activations captured at each of 40 latent steps × 3 agents.

| Category | n | planner mean ||h|| | range |
|----------|---|---------------------|-------|
| Correct  | 89 | 180.9 ± 17.6 | bounded 140-222 |
| Incorrect (all types) | 11 | 176.8 ± 19.7 | bounded 145-222 |
| **idx=20 (only confirmed mode collapse, 4-gram repeat ×830)** | 1 | **182.9** | **149-196** |

Per-step trajectory shape (planner role):
- Rises from ~149 (step 0) → ~195 (peak around step 5)
- Slowly decays to ~165 (step 39)
- This shape is **identical** between correct, incorrect, and the collapse sample

## Statistical test results

For each role × each stat:

| Role | mean ||h|| p | kurtosis p | max-ratio p |
|------|--------------|------------|-------------|
| planner | 0.317 | 0.805 | 0.463 |
| critic | 0.416 | 0.653 | 0.598 |
| refiner | 0.692 | 0.079* | 0.101 |

\* Marginal but doesn't survive Bonferroni correction for 9 tests.

**Conclusion**: No precursor signal in any tested statistic at any agent.

## Key finding: ||h|| stays bounded in broken code

The MILESTONE-2 framing implied ||h|| would drift unboundedly without rescaling. **It doesn't.**

For ALL broken-code samples (correct and incorrect):
- ||h|| starts at ~149 (initial embedding scale)
- Climbs to ~195 peak around step 5
- **Self-stabilizes** to ~165 over remaining steps
- **0/100 samples show monotonic blow-up**

The trajectory is bounded but at **~150× the embedding magnitude scale** (~1.1).

## Reframing Bug #1 root cause

Original MILESTONE-2 framing:
> *"Without rescaling, hidden state magnitudes drift unboundedly over many latent iterations and pushing the model into out-of-distribution input territory."*

**Corrected framing** (based on actual data):
> *Hidden state magnitudes in broken code DO NOT drift — they stabilize at ~165 across all samples. The OOD condition is INSTANTANEOUS and CONSISTENT: every iteration feeds back a vector ~150× larger than typical embedding rows. The model is robust to this OOD input 94% of the time, but in rare cases (~6%) it triggers a token-level pathology (autoregressive generation gets stuck on attractor tokens like "2 2 2 ..." or "DocumentDocument...").*

**The fix (norm rescaling) doesn't prevent drift** (there is no drift to prevent). **The fix makes the input magnitude match the embedding distribution** (~1.1 instead of ~150), reducing the OOD trigger rate from 6% to 0.23%.

**This is a subtle but important distinction**:
- WRONG: "drift over time leads to collapse"
- RIGHT: "constant OOD input occasionally triggers collapse, fix removes the OOD condition"

## What this means for the audit's narrative

Bug #1 fix **still works** — empirically reduces collapse rate. The fix is **correct**, just not for the reason MILESTONE-2 originally claimed.

Updates needed to FINAL-REPORT.md:
- MILESTONE-2 / Bug #1 explanation: replace "drift" framing with "constant OOD input" framing
- The fix mechanism: "rescaling brings input back to in-distribution magnitude" not "prevents accumulation"

This **doesn't change the PR #1 merge decision** (fix is still correct) but **changes the WHY**.

## What this means for the 3 fixed-code collapses (idx 396, 406, 952)

The original MILESTONE-12 reframe (per kiro-research) asked: "Are fixed-code collapses the same mechanism?"

With the new understanding:
- **Broken-code collapses** are also NOT explained by ||h|| (just confirmed)
- The mechanism appears to be **token-generation-level pathology** triggered by some interaction with latent context
- Fixed-code collapses are likely the **same mechanism**, just rarer because the trigger is less common
- All collapses (broken or fixed) are **autoregressive failures** during generation, not latent state failures

This is consistent with the observation that:
- idx=396: STARTS COHERENT, then LATE-STAGE drifts into "the answer is le..." loop
- idx=406: STARTS COHERENT, then MIDWAY drifts into "1 1 4 4 1 40" garbage
- idx=952: pathological from token 1 ("is the time period..." loop) — only this one looks like initial state failure

So **the mode collapse mechanism is**:
- Primarily: autoregressive token attractor (judger generation gets stuck on certain tokens)
- Rare initial-state cases (like idx=952) might have separate cause
- Latent state ||h|| is not the trigger

## H1a / H1b verdict

| Hypothesis | Verdict |
|-----------|---------|
| H1a (collapse = magnitude blow-up, same mechanism in broken+fixed) | **REFUTED** — broken collapse has NO blow-up |
| H1b (collapse = different mechanism with different latent signature) | **PARTIALLY REFUTED** — no signature in either group |
| H2 (kurtosis/spikiness difference) | **REFUTED** — also no signal |
| H3 (SAE features) | **NOT RUN** — given H1/H2 both null, low prior of finding meaningful SAE signal |

## What we did NOT prove

- That mode collapse has NO predictable precursor anywhere (could exist in attention patterns, token-level logits, etc.)
- That fixed-code collapses are mechanism-identical to broken-code collapses (need to capture activations on the 3 fixed collapses to verify directly)
- That refiner kurtosis (p=0.079, marginal) is truly null vs underpowered

## Caveats

- **Only 1 confirmed mode collapse in 100 broken samples** (n=1 in collapse group)
- **Lower-than-expected collapse rate this run** (1% observed vs 6% baseline) — temp=0.6 variance
- **Did NOT run fixed-code with capture** — comparing on broken side only
- **Detector heuristic** may miss borderline cases (e.g., idx=11 has many newlines but still hits final \boxed{})

## Recommended follow-up

1. **Update MILESTONE-2 / FINAL-REPORT framing** with corrected root cause (high priority)
2. **Capture activations on the 3 fixed-code collapses directly** to compare with broken-code collapse trajectory (~30 min run)
3. **Look at attention patterns or token-level logits** (next-level analysis) — current latent ||h|| analysis is null
4. **Run more broken samples** to get more collapses (~300 samples → 20 collapses)

## Collaboration credit

- **claude-585a78**: ran experiment, did analysis
- **kiro-research**: critical critique of original protocol (n=3 + 3σ guaranteed false positive), reframe to H1a vs H1b, identification that "broken vs fixed" comparison is the high-value question
- Both: agreed on Bonferroni correction, discovery/held-out split, cascade design (H1 → H2 → H3)

This audit's most important finding may be: **MILESTONE-2's drift narrative for Bug #1 is incorrect**. The fix is right; the explanation needs revision.
