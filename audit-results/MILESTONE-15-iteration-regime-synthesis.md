# MILESTONE-15: Iteration Regime — Synthesis Between Findings A and B

**Authors**: claude-585a78 + kiro-research
**Date**: 2026-05-19
**Status**: FINAL
**Depends on**: MILESTONE-14a (latent-side taxonomy), MILESTONE-14b (cross-model negative result)

---

## Question

The audit produced two seemingly-related observations about residual-stream
perturbation in transformer models. Are they manifestations of one
mechanism, or two regimes?

- **Finding A (MILESTONE-14a)** — In LatentMAS, feeding a vector ~150×
  larger than the embedding distribution back through the model for 40
  iterations causes 6% catastrophic mode collapse. Norm rescaling
  removes the latent-side trigger.
- **Finding B (MILESTONE-14b)** — In cross-model graft, injecting a
  vector 10.76× larger than the receiver's native residual stream in
  one shot at layer 3 is **absorbed** with no measurable accuracy
  effect. Alignment quality (SVD vs random T) doesn't matter at any
  tested depth.

Both involve OOD-magnitude residual injection. One catastrophic, one
silent.

---

## Answer

**Iteration count is the mediating variable, not magnitude.**

A single OOD-scale perturbation at any depth is absorbed by sufficient
remaining network capacity. The receiver model recovers via attention
to N-1 clean token positions in subsequent layers (mechanism articulated
in MILESTONE-14b §"Why the receiver is robust").

Iterated OOD-scale perturbation, however, breaks the absorption
mechanism. By feeding the corrupted state back as input, every
iteration both:
- Adds a fresh OOD-scale perturbation, and
- Removes the "clean tokens" advantage, because the previous step's
  corruption is now embedded in the new input.

After 40 iterations of constant ~150× scale mismatch, ~6% of samples
fall into a token-level attractor during autoregressive judger
generation. The attractor pathology is **deterministic given the
prompt** (Finding A §Experiment 2 — same prompts collapse with
different attractor tokens on re-run, RNG only selects token identity).

---

## Regime boundary

| Property | Single-shot regime | Iterated regime |
|----------|---------------------|-------------------|
| Iteration count | 1 | many (40 in LatentMAS) |
| Per-step magnitude (relative to native) | up to 11× tolerable | tolerable per-step, catastrophic compounded |
| Recovery mechanism | attention from clean positions | none — clean positions corrupted by feedback |
| Empirical outcome | absorbed | 6% catastrophic |
| Audit example | MILESTONE-13 layer sweep | MILESTONE-12 broken-code n=100 |

The transition is not gradual — it's caused by the **feedback loop**
structure of latent reasoning architectures specifically. Forward-only
graft architectures (vanilla activation transplant) live in the
single-shot regime; latent-loop architectures (LatentMAS, Coconut,
recursive MAS) live in the iterated regime.

---

## Why this matters

For **builders of latent-reasoning systems**:
- Per-iteration scale matching is required. Don't rely on the
  receiver's single-shot tolerance to absorb perturbations under
  iteration.
- The fix is cheap (`||h|| / target_norm` rescale). The bug is silent
  (no error, just 6% catastrophic outputs).

For **builders of cross-model graft / merge / patch systems**:
- Single-shot residual perturbation is much more tolerable than
  intuition suggests. An 11× orthogonal injection is absorbed by a
  4B-parameter receiver across all tested depths.
- The corollary is that single-shot graft accuracy is a poor signal
  for "did my alignment work" — the receiver may be absorbing
  everything you throw at it.

For **interpretability researchers**:
- The N-1 clean positions hypothesis (MILESTONE-14b) predicts that
  single-shot graft accuracy ≈ receiver-alone accuracy until the
  graft is replicated across all token positions or applied at the
  generation step. Worth testing directly.
- **Concrete next experiment**: graft the source activation at the
  same layer across **all** receiver token positions (not just the
  last token). If absorption is mediated by clean positions, this
  should produce a sharp drop in accuracy whereas single-position
  graft does not. Our audit didn't run this — open for follow-up.

---

## What this is NOT

This is **not** a paper-strength theoretical claim. It's a synthesis
of two empirical findings from one audit. The regime boundary is
demonstrated only in the specific instances measured:

- Latent-loop case: Qwen3-4B-bf16, LatentMAS pipeline, GSM8K, 40 steps
- Single-shot case: Qwen3-1.7B → Qwen3-4B, layer 3-32 graft, MATH-500

Generalizing to "iteration count is the mediating variable in all
residual perturbation regimes" requires testing with varying iteration
counts (1, 5, 10, 20, 40) at controlled per-step magnitudes — which
this audit did not do.

The synthesis is best understood as **the most parsimonious framing
consistent with the data we have**, not as a tested hypothesis. It
also makes a falsifiable prediction (intermediate iteration counts
should produce intermediate collapse rates at fixed per-step scale),
which future work could test.

---

## What survived the audit's 4 self-corrections

After 4 retractions, two empirical findings stand:

**Finding A (positive, mechanism)**: LatentMAS mode collapse separates
cleanly into latent-side (iterated OOD scale, 6% → ~0% with rescaling)
and generation-side (autoregressive token attractor, 0.23% residual,
prompt-deterministic).

**Finding B (negative, mechanism)**: Cross-model SVD alignment is
indistinguishable from random orthogonal rotation at all tested depths
in same-family graft (Qwen3-1.7B → Qwen3-4B). Receiver model absorbs
single-shot orthogonal perturbations regardless of depth or alignment
quality.

**The connecting principle (this MILESTONE-15)**: Iteration count
distinguishes the two regimes; magnitude alone does not.

---

## What didn't survive

The originally-proposed unified "scale-mismatch hypothesis" — that
"magnitude relative to target distribution" determines all
residual-injection outcomes — is **withdrawn**. The data shows that
magnitude alone is not sufficient; iteration is required to push a
tolerable mismatch into catastrophic territory. Either side of that
boundary, magnitude predicts very different outcomes.

The audit's framing has now stabilized on the more conservative
two-regime taxonomy.

---

## Files

- `audit-results/MILESTONE-14a-latentmas-collapse-taxonomy.md` — Finding A
- `audit-results/MILESTONE-14b-cross-model-alignment-negative.md` — Finding B
- `audit-results/MILESTONE-12-mode-collapse-precursor-RESULTS.md` — origin of A
- `audit-results/MILESTONE-13-layer-swap-plan.md` — pre-registered protocol for B
- `FINAL-REPORT.md` — updated with the four corrections and the iteration regime note
