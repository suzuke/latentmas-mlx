# Milestone 4 — Activation Graft -43pp Root Cause Identified

**Date**: 2026-05-17
**Status**: Bug identified via code-diff. Fix sketched. Not yet tested.

---

## Bottom line

The Activation Graft same-model -43pp regression reported in
RESULTS.md (90% → 47%) is caused by a **missing causal mask in
`generate_with_grafted_activation`'s manual layer-by-layer prefill**.

Without a causal mask, every prompt token attends to every other
token (including "future" tokens). This is severe OOD for the model
and produces near-random output, which matches the 47% accuracy we
see (slightly above chance for math).

This is independent of the LatentMAS norm-rescaling bug — completely
different code path, different problem.

---

## How I found it

### Step 1: Read the suspect code

`experiments/comm_activations.py:71-78` (Step 2 — generation with graft):

```python
h = inner.embed_tokens(input_ids[None])
kv_cache = mlx_cache.make_prompt_cache(model)

for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
    h = layer(h, None, c)              # <-- mask=None for prefill
    if i == graft_layer:
        h = mx.concatenate([h[:, :-1, :], grafted.reshape(1, 1, -1)], axis=1)
```

Same pattern in Step 1 (`cache_a[i]` = None, mask = None).

### Step 2: Compare with standard mlx_lm prefill

`mlx_lm.models.qwen3.Qwen3Model.__call__`:

```python
def __call__(self, inputs, cache=None, input_embeddings=None):
    h = ...
    if cache is None:
        cache = [None] * len(self.layers)
    mask = create_attention_mask(h, cache[0])    # <-- ALWAYS creates mask
    for layer, c in zip(self.layers, cache):
        h = layer(h, mask, c)                    # <-- mask propagated
    return self.norm(h)
```

### Step 3: Verify `create_attention_mask` actually does what I think

`mlx_lm.models.base.create_attention_mask`:

```python
def create_attention_mask(h, cache=None, ...):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, ...)
    if N == 1:
        return None                      # decode: no mask needed
    if return_array or ...:
        return create_causal_mask(N, ...)
    return "causal"                       # prefill: causal mask
```

For prompt prefill (N = prompt length, many tokens), the mask is "causal".
For decode (N = 1), mask is None.

### Conclusion

The manual forward in `generate_with_grafted_activation` does **prefill
(many tokens) with mask=None**, which the model interprets as "no
masking" — full bidirectional attention. The model is trained for causal
attention; it has never seen non-causal at inference. The output is
severely corrupted.

This perfectly explains:
- **Why same-model graft fails by 43pp**: the graft itself should be a
  no-op (replacing same activation with same), but the BROKEN PREFILL
  before the graft corrupts everything.
- **Why timing is slower (30.8s vs 23.6s)**: corrupted hidden state
  leads to less efficient generation (more rambling, more retries).
- **The accuracy (47%) is just above chance for short-numeric GSM8K**
  — consistent with the model producing random-ish answers.

---

## The fix

Two-line change in `experiments/comm_activations.py`.

Step 1 (capture A's activation):

```python
+from mlx_lm.models.base import create_attention_mask

 h = inner.embed_tokens(input_ids[None])
 cache_a = [None] * n_layers
+mask = create_attention_mask(h, None)  # None cache for non-cached forward
 activations = {}
 for i, layer in enumerate(inner.layers):
-    h = layer(h, None, cache_a[i])
+    h = layer(h, mask, cache_a[i])
     if i == graft_layer:
         activations[i] = h[0, -1]
```

Step 2 (re-forward with graft):

```python
 h = inner.embed_tokens(input_ids[None])
 kv_cache = mlx_cache.make_prompt_cache(model)
+mask = create_attention_mask(h, kv_cache[0])
 for i, (layer, c) in enumerate(zip(inner.layers, kv_cache)):
-    h = layer(h, None, c)
+    h = layer(h, mask, c)
     if i == graft_layer:
         h = mx.concatenate([h[:, :-1, :], grafted.reshape(1, 1, -1)], axis=1)
```

Expected result with fix: same-model graft accuracy ≈ baseline (~90%),
because the graft is mathematically a no-op when A=B.

---

## Why this wasn't caught earlier

The author's note in RESULTS.md:
> "Same-model activation grafting not effective; method designed for
>  cross-model communication."

This rationalizes the failure as method-design-intent rather than as a
bug. The intuition is wrong — for same-model graft, the entire pipeline
should reproduce baseline (graft is a no-op). The 47% is not a feature
of "method not designed for same-model"; it's a symptom of the prefill
corruption.

Cross-model graft would also be affected by the same bug, but the
underlying noise of cross-model transfer makes it harder to spot.

---

## Recommended actions

1. **Apply the two-edit fix** to `experiments/comm_activations.py`.
2. **Verify on Qwen3-8B-4bit GSM8K 30 samples** — expected
   same-model accuracy ≈ baseline 90% (was 47%).
3. **Then test cross-model graft** with the fix — get the first
   meaningful number for the method's actual purpose.

---

## Confidence level

**High** for diagnosis (code-diff is unambiguous).
**Medium-high** for impact prediction. The fix should mostly restore
same-model accuracy. Cross-model accuracy is unpredictable since the
method's underlying viability was never properly tested due to the bug.

---

## Comparison to MILESTONE-2 (norm-rescaling bug)

| Aspect | Norm rescaling bug | Activation Graft bug |
|--------|-------------------|---------------------|
| Found via | code diff vs original repo | code diff vs standard mlx_lm |
| Location | `latentmas/run.py:latent_steps` | `experiments/comm_activations.py:generate_with_grafted_activation` |
| Severity | -8 to -10pp on GSM8K, mode collapse | -43pp on Qwen3-8B same-model |
| Fix size | ~25 lines | ~3 lines |
| Confidence pre-test | High | High |
| Confidence post-test | Confirmed (this audit) | Pending |

Both bugs share a pattern: **the MLX port re-implements existing model
machinery instead of calling the standard mlx_lm/transformers API**.
Standard machinery handles things like norm rescaling and causal mask
creation automatically; manual reimplementation skips them.
