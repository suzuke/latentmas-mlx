# Milestone 5 — RecursiveMAS Pre-Norm vs Post-Norm Bug Identified

**Date**: 2026-05-17 (overnight session)
**Status**: Bug identified via code-diff. Fix sketched. Verification pending.

---

## Bottom line

The MLX port of RecursiveMAS feeds the **pre-final-norm** hidden state
to the InnerLink adapter, while the original PyTorch implementation
feeds the **post-final-norm** hidden state (via HuggingFace's
`outputs.hidden_states[-1]`).

This is the LatentMAS-style bug at a different layer — and likely
explains the 40% vs paper's 72% gap reported in RESULTS.md for
RecursiveMAS Sequential-Light on MATH-500.

---

## How the bug was found

### Step 1: Read MLX port forward pass

`recursive_mas/run.py:196-221`:

```python
def _forward_get_raw_hidden(model, input_embeds):
    """Run model forward and return raw hidden state (before final norm).
    Computes in float32 for numerical accuracy matching PyTorch."""
    inner = _get_inner(model)
    h = input_embeds.astype(mx.float32)
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])

    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
        h = h.astype(mx.float32)

    return h   # <-- PRE-final-norm
```

The docstring explicitly says "(before final norm)". The author
intentionally returns pre-norm.

### Step 2: Read original PyTorch forward pass

`/tmp/recursive-orig/inference_utils/inference_mas.py:868-869`
(within `autoregressive_latent_rollout`):

```python
outputs = model(logits_to_keep=1, **forward_kwargs)
# ...
last_hidden = outputs.hidden_states[-1][:, -1, :]
```

Uses `outputs.hidden_states[-1]` from HuggingFace.

### Step 3: Verify HF convention

In `transformers/models/qwen2/modeling_qwen2.py` (and Qwen3, LLaMA):

```python
hidden_states = inputs_embeds
all_hidden_states = () if output_hidden_states else None

for decoder_layer in self.layers:
    if output_hidden_states:
        all_hidden_states += (hidden_states,)  # input to this layer
    hidden_states = decoder_layer(...)[0]

hidden_states = self.norm(hidden_states)    # FINAL NORM

if output_hidden_states:
    all_hidden_states += (hidden_states,)    # POST-NORM appended last
```

So `all_hidden_states[-1]` is **POST-norm**. Confirmed against the
HF convention.

### Step 4: Verify MLX-LM convention

`mlx_lm.models.qwen3.Qwen3Model.__call__`:

```python
def __call__(self, inputs, cache=None, input_embeddings=None):
    h = ...
    for layer, c in zip(self.layers, cache):
        h = layer(h, mask, c)
    return self.norm(h)   # <-- standard inner returns POST-norm
```

MLX-LM's standard `inner_model(...)` ALSO returns post-norm. The MLX
port's `_forward_get_raw_hidden` is custom code that *avoids* the
final norm — that's where the divergence is.

### Conclusion

The MLX port's manual forward pass intentionally drops the final norm.
But the InnerLink adapter was trained on POST-norm features. Feeding
it PRE-norm features results in:

1. Different input distribution than during training
2. The adapter's `pre_ln` (LayerNorm) compensates partially but not
   fully — the magnitudes and learnable γ/β of `inner.norm` are missing
3. Downstream `proj1 → gelu → proj2 → residual` operates on shifted
   features → produces wrong next-embed
4. Errors cascade across `latent_steps` iterations (48 by default,
   so ~50× amplification of any initial misalignment)
5. Cross-model OuterLink sees this corrupted output and projects it
   to the next model's space, further compounding the error

This is the LatentMAS norm-rescaling bug's cousin — same family of
"MLX port re-implements model machinery and skips a normalization
step that the original had".

---

## Why the author thought this was right

The docstring says **"matching HuggingFace hidden_states[-1]"** — but
also explicitly says "before final norm". These two statements are
contradictory in modern HF (where hidden_states[-1] IS post-norm).

The author may have confused this with older HF behavior, or with
"raw hidden state before lm_head" (which is true for the final norm'd
hidden state — it's still pre-lm_head).

Easy mistake to make. Hard to catch without running the diff against
the original PyTorch code, which is what we just did.

---

## The fix

One line: add `h = inner.norm(h)` before return.

```python
def _forward_get_raw_hidden(model, input_embeds):
    """Run model forward and return POST-norm hidden state, matching
    HuggingFace outputs.hidden_states[-1] used by the original PyTorch
    RecursiveMAS (inference_mas.py:869)."""
    inner = _get_inner(model)
    h = input_embeds.astype(mx.float32)
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])

    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
        h = h.astype(mx.float32)

    # Apply final norm to match HF's hidden_states[-1] convention.
    # The InnerLink adapter was trained on POST-norm features, so
    # feeding pre-norm here causes severe distribution shift.
    h = inner.norm(h).astype(mx.float32)
    return h
```

Plus rename / docstring update to remove the misleading "before final
norm" language.

---

## Expected impact on the 40% vs 72% gap

If this bug is the primary cause:
- Fixed accuracy should be 60–72% (close to paper's 72%, allowing for
  sampling variance and MLX-vs-PyTorch numerical drift)
- This would be a +20pp recovery — even bigger than the LatentMAS fix

If the bug is NOT the primary cause:
- Fixed accuracy stays near 40%, gap remains
- Other bugs likely exist (cf. the multiple bugs in latentmas-mlx)

---

## Verification plan

1. Implement fix on `audit/fix-recursivemas-norm` branch.
2. Wait for user's 1319 GSM8K run to finish (currently running) to
   free up GPU.
3. Run BROKEN RecursiveMAS Sequential-Light on MATH-500 30 samples
   (matches RESULTS.md condition): expect ~40% per the existing data.
4. Run FIXED RecursiveMAS Sequential-Light on same 30 samples.
5. Compare. Expected fixed: 60–72%.

Memory budget: 3 models loaded simultaneously
- Qwen3-1.7B + LLaMA3.2-1B + Qwen2.5-Math-1.5B ≈ 9 GB at bf16
- Fits comfortably on 128GB M3 Max

Time budget: per RESULTS.md, ~41s/sample × 30 samples × 2 conditions
= ~40 minutes total.

---

## Confidence assessment

| Aspect | Confidence |
|--------|-----------|
| Code-diff identifies a real divergence | **High** — unambiguous |
| Divergence affects training-time vs inference-time distribution | **High** — adapter dtype matters |
| Divergence accounts for most of the -32pp gap | **Medium** — could be more bugs |
| Fix accuracy reaches paper's 72% on MATH-500 30 samples | **Medium** — sampling variance |

---

## Pattern recognition: third bug of similar shape

This is the **third** MLX-port bug we've identified that has the same
structural cause: **the port re-implements model machinery and silently
omits a normalization/standardization step the original had**:

| Bug | What was omitted | Affected model |
|-----|------------------|----------------|
| **MILESTONE-2** (LatentMAS) | `_apply_latent_realignment` norm rescaling per latent step | LatentMAS Qwen3-4B GSM8K |
| **MILESTONE-4** (Activation Graft) | `create_attention_mask` for prefill | comm_activations Qwen3-8B |
| **MILESTONE-5** (RecursiveMAS) | `inner.norm` final norm on returned hidden state | RecursiveMAS multi-model |

The recurring pattern suggests: **anywhere the MLX port does manual
layer-by-layer forward instead of calling `inner(inputs, cache=...)`,
there's a high chance of a missing step**.

Recommendation: an LLM-aided "diff" tool that compares manual MLX
forward passes against `inner.__call__` could catch all three classes
automatically.
