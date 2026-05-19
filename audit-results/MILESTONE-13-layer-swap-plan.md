# MILESTONE-13: Layer Swap Sensitivity Experiment

**Author**: kiro-research
**Date**: 2026-05-19
**Status**: PROTOCOL (pre-registered, not yet run)
**Depends on**: Bug #2 fix (causal mask), cross-model graft framework from `audit/feature-cross-model-graft`

---

## Question

Does graft layer depth determine whether alignment (SVD vs random T) matters?
MILESTONE-11 showed SVD = random at late-layer graft. Is this because:
- (A) Late-layer injection is irrelevant (signal drowned by residual) → early-layer should show SVD > random
- (B) Receiver absorption is universal (any layer) → SVD = random everywhere
- (C) SVD alignment is genuinely useless (no information preserved) → SVD = random everywhere, but early-layer random T also catastrophically fails

## Hypotheses (pre-registered)

- **H1**: At early graft layers (≤25% depth), random T accuracy drops significantly below SVD-aligned T (>10pp gap)
- **H2**: cos_sim(residual_pre_graft, residual_post_graft) decreases monotonically as graft_layer_b moves earlier
- **H3**: There exists a "sensitivity threshold" layer below which graft alignment quality matters and above which receiver recovers regardless

**If H1 confirmed**: SVD alignment does useful work; MILESTONE-11's null result was a sensitivity problem (late-layer setup)
**If H1 rejected (SVD = random at all layers)**: Alignment is genuinely broken; zero-shot cross-model transfer needs trained translators
**If early-layer graft universally catastrophic (both SVD and random)**: The graft mechanism itself may be fundamentally incompatible with early injection

## Design

### Fixed parameters
- **Source model**: Qwen3-1.7B (sender)
- **Receiver model**: Qwen3-4B
- **Source graft layer**: 50% depth of Qwen3-1.7B (layer 14/28)
- **Benchmark**: MATH-500 (harder than GSM8K for 4B; expected baseline ~50-60%)
- **Temperature**: 0.0 (deterministic, eliminates sampling variance)
- **n per condition**: 50

### Swept parameter
- **graft_layer_b** (receiver injection point): 10%, 25%, 50%, 75%, 90% of Qwen3-4B depth
  - Qwen3-4B has 36 layers → layers 4, 9, 18, 27, 32

### Conditions per layer (3 × 5 = 15 cells)
1. **no-graft baseline** (receiver only, no injection)
2. **SVD-aligned T** (same as MILESTONE-10 setup)
3. **random orthogonal T** (same as MILESTONE-11 control)

### Additional measurements
- **cos_sim(h_pre, h_post)**: cosine similarity of residual stream immediately before vs after graft injection, averaged over all 50 samples
- **L2_ratio**: ||injected_activation|| / ||residual_at_graft_layer||, to quantify signal magnitude

### Controls
- Baseline (no-graft) should be constant across all "layers" (same model, same input) — serves as sanity check
- Random T at late layer should reproduce MILESTONE-11's 90% finding
- If baseline itself varies >5pp across runs, flag as instability

### Success criteria
- Clean monotonic or threshold pattern in accuracy vs layer
- cos_sim measurement confirms whether injection is perturbative or negligible
- Unambiguous answer to "does layer depth mediate alignment sensitivity"

## Implementation plan

1. Checkout `audit/feature-cross-model-graft` branch
2. Modify `comm_activations.py` to accept `--graft_layer_b` as absolute layer index
3. Add cos_sim + L2_ratio logging (print per-sample, aggregate at end)
4. Run 15 conditions × 50 samples = 750 inference calls
5. Tabulate results, plot accuracy vs layer for each condition

## Estimated runtime
- ~50 samples × 15 conditions = 750 forward passes
- Qwen3-4B at 4-bit: ~3-5 sec/sample on M-series Mac → ~40-60 min total
- Will run in batches of 5 layers × 3 conditions if memory allows

## Risks
- Qwen3-4B may be too strong even on MATH-500 (baseline >80%) → switch to MATH-500 level 4-5 only
- Early-layer graft may produce garbage (not informative failure, just OOD crash) → still informative if it's systematic

---

## Amendment 1 (T+30 review, 2026-05-19)

**Triggered by**: claude-585a78 challenge — n=50 underpowered for 10pp detection at 85% baseline.

**Power analysis**: For two proportions p1=0.85 (SVD) vs p2=0.75 (random), α=0.05 two-sided:
- n=50/group: power ≈ 0.42 (coin flip — unacceptable)
- n=100/group: power ≈ 0.73 (marginal)
- n=150/group: power ≈ 0.87 (acceptable)

**Changes**:
1. **n=100 per condition** (up from 50). Total = 100 × 15 = 1500 calls. Runtime ~2h on M-series.
2. **MATH-500 level 3-5 subset only** (filter out level 1-2 which 4B model trivially solves). Target baseline 60-75%.
3. **Pre-registered decision rule**: Accept H1 if `SVD_acc - random_acc > 5pp` at any layer with p<0.05 (Fisher exact, Bonferroni-corrected for 5 layers).
4. **Fallback**: If baseline on MATH-500 level 3-5 is still >80%, switch to GPQA-diamond (baseline ~40-50% for 4B).

**Unchanged**: cos_sim + L2_ratio measurements, layer sweep points, temp=0.

*Amendment locked.*
