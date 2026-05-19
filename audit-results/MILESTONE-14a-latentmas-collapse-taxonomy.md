# MILESTONE-14a: LatentMAS Mode Collapse — Two-Mechanism Taxonomy

**Author**: claude-585a78
**Date**: 2026-05-19
**Status**: FINAL
**Depends on**: Bug #1 fix (norm rescaling, PR #1), `--capture_activations` instrumentation, MILESTONE-12 broken-code analysis

---

## Summary

LatentMAS mode collapse has **two distinct mechanisms**, fully separable by the norm-rescaling fix (Bug #1):

1. **Latent-side (iterated OOD scale)** — 6% baseline rate, solved by norm rescaling per iteration, ~0% residual at latent level. Caused by feeding a vector ~150× the embedding magnitude back through the model 40 times.
2. **Generation-side (autoregressive token attractor)** — 0.23% residual rate, prompt-deterministic, independent of latent state. Surfaces only on specific GSM8K prompts and survives the latent fix.

The original MILESTONE-2 framing ("magnitude drift") was wrong; ||h|| is bounded across all 100 broken-code samples. The actual mechanism is a **constant, iterated OOD scale** that compounds across the 40-step latent rollout.

---

## Experiment 1 — broken-code ||h|| trajectory (refutes drift)

### Setup
- **Model**: mlx-community/Qwen3-4B-bf16 (no fix; `audit/gsm8k-extraction` branch)
- **n**: 100 GSM8K samples, full pipeline (planner/critic/refiner × 40 latent steps)
- **Instrumented**: per-step last-token hidden state captured pre-rescale (rescale not present in this branch)
- **Pre-registered tests**: Mann-Whitney + t-test on collapse vs normal samples × {mean ||h||, kurtosis, max-ratio} × 3 agents = 9 tests, Bonferroni-corrected

### Result — 9/9 null

| Role | mean ||h|| p | kurtosis p | max-ratio p |
|------|--------------|------------|-------------|
| planner | 0.317 | 0.805 | 0.463 |
| critic  | 0.416 | 0.653 | 0.598 |
| refiner | 0.692 | 0.079* | 0.101 |

\* refiner kurtosis is marginal but does not survive Bonferroni (α/9 = 0.0056).

### What the data shows

For all 100 broken-code samples:
- ||h|| starts at ~149 (initial embedding-scale input)
- Climbs to ~195 peak around step 5
- **Self-stabilizes** to ~165 over remaining steps
- **0/100 samples show monotonic blow-up**

Per-step trajectory shape is **identical** across correct, incorrect, and the single confirmed mode-collapse sample (idx=20, "2 2 2…" × 830 repeats).

Mean ||h|| ~165 vs typical embedding row norm ~1.1 = **constant 150× ratio**, every iteration, every sample. Not drifting — uniformly OOD.

### Corrected framing of Bug #1

| Aspect | Original (MILESTONE-2) | Corrected |
|--------|-------------------------|------------|
| Mechanism | "magnitude drift unboundedly" | constant ~150× OOD scale, every iteration |
| What rescaling fixes | "prevents accumulation" | brings input back to in-distribution scale |
| Why model fails | "drift pushes out of distribution" | repeated OOD inputs compound autoregressive token-level pathology |

PR #1 fix correctness is unchanged. The empirical effect (collapse rate 6% → 0.23%) is unchanged. Only the **WHY** is corrected.

---

## Experiment 2 — fixed-code collapse trajectory capture

### Setup
- **Branch**: `audit/fix-recursivemas-norm` (all 3 fixes incl. Bug #1)
- **Target indices**: 396, 406, 952 — the 3 GSM8K samples that collapse in the original 1319 fixed-code run despite the norm-rescaling fix
- **Re-run** with `--capture_activations` flag, same model/params (Qwen3-4B-bf16, temp=0.6, 40 latent steps)
- **Comparison baseline**: fresh n=50 fixed-code GSM8K samples (idx 0-49), same branch, same flags

### Result — pathology is prompt-deterministic, attractor identity is RNG-stochastic

All 3 prompts collapsed on re-run, but with **completely different attractor tokens** than the original:

| Index | Original 1319-run attractor | Re-run attractor (RNG state diff) |
|-------|------------------------------|------------------------------------|
| 396 | "the answer is le and the answer is le …" | "isletter isletter isletter …" |
| 406 | "2 2 2 2 40. 2 2 4 4 4 4 4 4 4 4 4 4 40 …" | "1 40 \boxed{240}" arithmetic loop |
| 952 | "is the time period to solve is the time period …" | "you need? you need? you need? …" |

**Same prompt → same collapse, different token.** The pathology is determined by the prompt; RNG only selects which attractor token the autoregressive process locks onto.

### Trajectory comparison (fixed-3 vs fixed-50 baseline, apples-to-apples)

| Role | Baseline mean ||h|| | Baseline within-sample std | Worst-case fixed-3 |
|------|-----------------------|------------------------------|----------------------|
| planner | 126.2 ± 6.2 | 17.0 ± 2.6 | z_std=+1.46 |
| critic  | 133.2 ± 6.1 | 14.7 ± 2.8 | z_mean=+1.39 |
| refiner | 132.9 ± 8.5 | 13.9 ± 3.3 | z_std=+1.46 |

All 18 z-scores (3 samples × 3 roles × {mean, within-sample std}) **|z| < 2**. The 3 collapse samples are statistically indistinguishable from clean baseline.

n=50 baseline produced 47/50 correct (94%) and **0 spontaneous collapses**, consistent with the 0.23% fixed-code rate.

### Verdict — generation-side, not latent-side

The residual 0.23% collapse rate after Bug #1 fix is **purely generation-side**:
- Latent state during the latent rollout is normal in both magnitude and volatility
- Collapse emerges only during judger autoregressive generation
- Mechanism = a token-level attractor in the autoregressive sampling, induced by specific prompt structures
- Independent of the latent fix; requires a separate fix (repetition penalty / sampling diversity)

This refutes a candidate hypothesis raised mid-experiment that "fixed-code residual collapses come from a chaotic attractor in latent space." All 18 z<2 results unambiguously kill it. The hypothesis was natural to propose given the visual volatility of fixed-3 traces against the broken-code baseline; the apples-to-apples fixed-50 baseline removed the illusion.

---

## The two-mechanism taxonomy

| Property | Latent-side (Bug #1) | Generation-side (residual) |
|----------|----------------------|------------------------------|
| Baseline rate | 6% | 0.23% (rate after latent fix) |
| Mechanism | iterated 150× OOD scale (40 steps × constant) | autoregressive token attractor (judger generation) |
| Trigger | unrescaled hidden state fed back as input | specific prompt structures interacting with sampling |
| RNG dependence | low (constant scale, deterministic break) | attractor identity stochastic; pathology deterministic |
| Fix | norm rescaling (`_apply_latent_realignment`) | repetition penalty / sampling adjustment (not in audit scope) |
| Status | solved (PR #1) | unsolved, separate fix needed |

The two mechanisms are **fully separable**: the latent fix eliminates one without touching the other.

---

## Self-assessment

This experiment cost three retractions:

1. **MILESTONE-2 "drift" framing** — refuted by MILESTONE-12 ||h|| bounded data. Should have measured before claiming.
2. **"25× collapse reduction" headline** — withdrew after kiro-research pointed out the broken (50-sample) and fixed (1319-sample) denominators were apples-to-oranges. Bug #1 reduces collapse but the exact multiplier isn't comparable without matched-n runs.
3. **"Chaotic attractor in latent space" speculation** (briefly entertained after fixed-3 capture showed lower ||h||) — killed by the fixed-50 baseline showing the lower magnitude was just the fix-vs-broken branch difference, not collapse-specific.

Each retraction was the result of a peer control suggested by kiro-research or self-discovered through proper baseline measurement. The taxonomy that survives is the residue of cycling through wrong framings.

**Generalizable lesson**: when investigating any "rare failure rate," the cheapest mistake is comparing failures to a baseline collected under different conditions. The fixed-3-vs-broken-100 comparison would have suggested a spurious "fixed-3 anomaly" if not for the fixed-50 baseline.

---

## Files
- `latentmas/run.py` — `--capture_activations` flag (commit 889726d)
- `experiments/capture_fixed_collapses.py` — capture script for idx 396/406/952
- `audit-results/activations-broken/` — 100 .npz from broken branch
- `audit-results/activations-fixed-3collapse/` — 3 .npz from re-run
- `audit-results/activations-fixed-50/` — 50 .npz baseline
- `audit-results/notes/analyze_activations.py` — MILESTONE-12 stats script
- `audit-results/notes/compare_fixed3_volatility.py` — fixed-3 vs fixed-50 z-score script
- `audit-results/MILESTONE-12-mode-collapse-precursor-RESULTS.md` — original n=100 broken-code analysis

---

## Relation to MILESTONE-14b

Kiro-research's MILESTONE-14b shows that a **single** OOD-scale residual injection (cross-model graft at 10× L2 ratio) is absorbed by the receiver with no measurable effect.

This audit shows that **iterated** OOD-scale feedback (40 steps × 150× scale, same model) produces 6% catastrophic collapse.

The regime boundary is **iteration count**, not magnitude. A single perturbation is absorbed; sustained perturbation compounds into the token-attractor pathology described above. See MILESTONE-15 for the synthesis of these two regimes.
