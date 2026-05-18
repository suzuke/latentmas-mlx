# Milestone 12 (PLAN, pre-registered) — Mode Collapse Precursor Detection

**Date**: 2026-05-18
**Author**: claude-585a78
**Reviewer**: kiro-research (T+30 check-in)
**Status**: PRE-REGISTRATION — written BEFORE running experiments. Lock this file.

---

## Goal

Can we detect (or predict) LatentMAS mode collapse from the latent
state itself, *before* the model produces the degenerate output?

If yes → practical: monitor `||h||` (or some SAE feature) at inference
time; fall back to safe generation when collapse is likely. Audit
delivers an actionable secondary contribution.

If no → still valuable: confirms mode collapse is a sudden /
unpredictable failure mode, framing future research.

---

## Pre-registered hypotheses (REVISED per kiro-research critique)

The headline question is **NOT** "do broken-code collapses correlate
with ||h|| blow-up" (that trivially confirms Bug #1 mechanism). The
real question is:

> **Are the 3 fixed-code collapses (idx 396, 406, 952) the same
> mechanism as broken-code collapses?**

This has two competing sub-hypotheses:

**H1a (same-mechanism)**:
> Fixed-code collapse samples' `||h_t||` trajectory in latent rollout
> still grows like broken-code collapse samples (just more rarely).
> Norm rescaling is a **partial fix** — bounds the typical case but
> edge cases still escape.

**H1b (different-mechanism)**:
> Fixed-code collapse samples' `||h_t||` stays **bounded** (similar to
> normal samples). Their collapse is caused by something OTHER than
> magnitude drift — e.g., attention pattern degeneration, attractor
> collapse, embedding-table specific token loops.

**H2 (medium cost, run if H1a/H1b are ambiguous)**:
> `kurtosis(h)` or `max(|h|) / ||h||` differs between fixed-code
> collapses and normal samples in a direction NOT explained by ||h||
> alone (i.e., orthogonal failure mode).

**H3 (most expensive, only if H1+H2 leave gap)**:
> Trained SAE features predict fixed-code collapse with held-out
> validation, Bonferroni-corrected for dict size.

H1a/H1b/H2/H3 form a cascade — only run higher cost if lower is
inconclusive.

## Why this reframe matters

- If **H1a wins** → norm rescaling is incomplete; future fix is to
  add a **hard magnitude ceiling** or **spectral norm constraint** on
  the feedback path
- If **H1b wins** → there's a **second collapse mechanism** unknown
  to us; the audit didn't catch a second bug class
- Either outcome is **practically useful**: actionable signal for
  monitoring at inference, OR clear next-bug-hunt direction

Compare with the original H1 ("collapse vs normal differ in ||h||")
which was just confirming Bug #1 mechanism — low information value.

**Credit**: this reframe is kiro-research's contribution to MILESTONE-12.

## Critical anti-cheat design

1. **Discovery / held-out split before looking at data**:
   - Discovery: 200-300 broken-code samples (~12-18 collapses expected at 6% rate)
   - Held-out test: the **3 fixed-code collapses already in jsonl** (idx 396, 406, 952 from `latent_mas-gsm8k-1319-bf16-fixed.jsonl`)
2. **Bonferroni correction** for multiple hypothesis testing — accept critique from kiro-research that raw 3σ is guaranteed false-positive at high dim
3. **Predict effect direction in advance** (collapse > normal, monotonic blow-up) — no fishing for any-direction "significant" effects
4. **Sample-size sanity**:
   - Discovery n_collapse ≥ 10 (12-18 expected)
   - Held-out n_collapse = 3 (fixed; lock these for true held-out validation)

## Data collection

### Step 1: Modify `latentmas/run.py` to capture per-step activations

Add a `--capture_activations PATH` flag that:
- During latent_steps(), append each `h` after `_get_hidden_and_logits`
- Save per-sample to `.npz` with keys `step_00.npy`, ..., `step_39.npy`
- Each h: `(seq_len, hidden_dim)` → save only last token: `(hidden_dim,)`
- Per agent: planner / critic / refiner stages

This is a non-functional (logging-only) patch; same pattern as
`--save_outputs`.

### Step 2: Run BROKEN code on 200-300 samples

Use `audit/gsm8k-extraction` branch (no Bug #1 fix) with new
`--capture_activations` flag. Expected output:
- ~200-300 jsonl rows
- ~200-300 .npz files (one per sample, each ~1MB)
- ~12-18 collapses detected via existing detector

Time: ~60 min at broken-code rate.

### Step 3: Analyze discovery set (H1 + H2)

For each timestep t ∈ [1, 40] and each agent stage:
- Compute distribution of ||h_t|| separately for collapse vs normal samples
- Compute Mann-Whitney U test (non-parametric, robust to outliers)
- Apply Bonferroni correction: `α / (40 × 3 agents) = α / 120`

Decision rule:
- If H1 corrected-significant at ANY timestep with collapse > normal:
  proceed to held-out validation on fixed-code 3 collapses
- Otherwise: try H2 (kurtosis)
- Both fail: proceed to H3 (SAE) or accept negative result

### Step 4: Held-out validation

The 3 fixed-code collapse samples must show:
- Same direction of effect (collapse > normal in ||h|| or kurtosis)
- Significant under same threshold (just n=3 vs many normals — small but useful)

Cannot tune anything based on held-out — that defeats validation.

## What would invalidate H1/H2

Several outcomes worth pre-thinking:

- **Collapse samples have SAME ||h|| as normal**: precursor isn't magnitude-based; mode collapse may be sudden phase transition
- **||h|| differs in discovery but not held-out**: discovery overfitting; either bigger n or accept null
- **||h|| differs but direction is OPPOSITE expected** (collapse h is smaller): suggests different mechanism (maybe "saturation" not "drift")
- **H1 statistically significant but tiny effect size**: not useful for practical monitoring even if "real"

## Time budget

| Phase | Time |
|-------|------|
| Code activation capture (patch `latentmas/run.py`) | 20 min |
| Run 200-300 broken samples with capture | 60 min |
| Run H1 + H2 analysis | 30 min |
| Run held-out validation | 10 min |
| If H1+H2 fail: train + apply SAE for H3 | +90 min |
| Write up MILESTONE-12-results | 30 min |

Total: **2.5 hours minimum**, up to **4 hours if SAE needed**.

## What does NOT count as success

- "Found a feature with p < 0.05 raw" — must be Bonferroni-corrected
- "Discovery set shows effect" — must validate on held-out
- "Feature activates on 2/3 held-out" — must justify why 1 didn't (small n, but at least note it)

## Output

`MILESTONE-12-mode-collapse-precursor-RESULTS.md` documenting:
- Which H survived (or all failed)
- Held-out validation outcome
- Effect size (NOT just p-value)
- Honest negative results if H1/H2/H3 all fail

## Branch policy

- Activation-capture patch on new branch `audit/feature-activation-capture`
- Off `audit/fix-recursivemas-norm` (so has all 3 bug fixes)
- Will need to switch to `audit/gsm8k-extraction` for actually running broken code

## What I'd ask kiro to challenge in this plan (before running)

1. Is Mann-Whitney the right test, or should I be testing `dlog(||h||)/dt`
   instead of point ||h|| values?
2. Is 200-300 broken samples enough to expect ≥10 collapses? Should I
   run more for power?
3. Should H2 use kurtosis or max-ratio? Kurtosis is more standard but
   noisier at small n
4. Am I right to do agent-stage-separate analysis? Or should I pool
   across all 120 steps?

T+30 check-in: I'll send revised plan if kiro flags issues.
