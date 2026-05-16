# Communicating Activations — Suspected Implementation Issue

**File**: `experiments/comm_activations.py`
**Observation**: same-model graft drops 90% → 47% (-43pp per RESULTS.md)

## Why this looks suspicious

For **same-model** activation grafting (A = B = same model, same input),
the captured activation at layer k SHOULD equal the recomputed activation
at layer k. Therefore "replacing" should be a no-op.

A -43pp catastrophic drop on a no-op suggests either:

1. **Real subtle difference** — MLX layer behavior differs between
   `cache=None` (step 1) and `cache=KVCache` (step 2). Position
   encoding, attention mask handling, or memory layout differ.

2. **Concat-vs-overwrite mismatch** — Step 2 line 76:
   ```python
   h = mx.concatenate([h[:, :-1, :], grafted.reshape(1, 1, -1)], axis=1)
   ```
   This reconstructs `h`, but the KV cache for layer k has already been
   populated with the ORIGINAL last-token's K/V from earlier in the layer
   call. So the cache is one step "ahead" of the modified hidden state.

3. **Docstring mismatch** — Docstring describes "Model A generates a
   completion at temp=0.7", but code only captures one activation from
   a single forward pass on the prompt. The intended algorithm and
   implementation don't match.

## What to test (later, after main audit)

- [ ] Verify: with `graft_layer = -1` (no graft), does `generate_with_grafted_activation`
      produce baseline-equivalent output? If not, the framework itself
      is broken.
- [ ] Verify: capture activation at step 1, compute activation
      at the same point in step 2 BEFORE replacement, compute `|delta|`.
      If `|delta| > epsilon`, that's the source.
- [ ] Verify: does the KV cache for layers `[0, graft_layer]` store
      values derived from h BEFORE or AFTER the replacement? If before,
      then subsequent layers see inconsistent state (cache says X,
      input is X').

## Why this matters

If the -43pp is a real implementation bug, the RESULTS.md's "Same-model
activation grafting not effective; method designed for cross-model
communication" claim is **scientifically incorrect** — the method
might work fine cross-model in their paper but their MLX port has a bug.

This affects the broader narrative of the repo's results.
