# Milestone Report 2 — Bug Identified: Missing Norm Rescaling

**Date**: 2026-05-17
**Branch**: `audit/gsm8k-extraction`
**Status**: HIGH CONFIDENCE root cause of mode collapse identified.

---

## Bottom line

**The MLX port is missing the norm rescaling step that the original
PyTorch LatentMAS implementation applies at every latent step.**

This omission likely causes the mode collapse observed in 3/50 (= 6%)
of LatentMAS GSM8K samples — consistent across bf16 / 4-bit precision
and with/without OBF compression.

---

## Evidence trail

### Hypothesis search

We tested whether mode collapse is caused by:
- ❌ bf16 precision — also occurs at 4-bit
- ❌ OBF compression — also occurs with `--no_compress`
- ❌ Token budget interaction — collapses ≠ truncations

This left "structural to the LatentMAS pipeline" — but was it the
**algorithm** itself, or the **MLX port's implementation**?

### The discriminating test: read the source

Cloned the original LatentMAS at <https://github.com/Gen-Verse/LatentMAS>
and diff'd against the MLX port. Critical algorithmic divergences:

#### 1. **Missing norm rescaling at every latent step (THE BUG)**

Original (`models.py:204-211`):
```python
def _apply_latent_realignment(self, hidden, model):
    matrix, target_norm = self._ensure_latent_realign_matrix(model, ...)
    hidden_fp32 = hidden.to(torch.float32)
    aligned = torch.matmul(hidden_fp32, matrix)  # identity when realign off
    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / aligned_norm)  # <-- RESCALING
    return aligned.to(hidden.dtype)
```

Where `target_norm = input_embeds.weight.norm(dim=1).mean()` — the
**typical norm of an entry in the embedding matrix**.

This is applied **unconditionally** every latent step, even when
`--latent_space_realign` is off (matrix is then identity, but
norm rescaling still happens).

The MLX port (`latentmas/run.py:197-212`):
```python
def latent_steps(model, prompt_ids, kv_cache, n_steps=20):
    _get_hidden_and_logits(model, prompt_ids[None], kv_cache)
    for _ in range(n_steps):
        if _ == 0:
            h = _get_hidden_and_logits(model, mx.array([[0]]), kv_cache)
        else:
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
    return kv_cache
```

**No rescaling.** Hidden state magnitudes drift uncorrected for 40 (×3
agents = 120 total) iterations.

#### 2. Other hyperparameter divergences

| Parameter | Original | MLX port | Ratio |
|-----------|----------|----------|-------|
| `latent_steps` default | **10** | **40** | 4× |
| judger `max_new_tokens` | **256** | **2048** (via CLI) | 8× |
| temperature | 0.7 | 0.6 | — |
| top_p | 0.95 | none | — |
| `<think>` prompt wrapping | yes | no | absent |
| latent_only KV truncation mode | yes | no | absent |
| OBF compression | no | yes | added |

The 4× latent_steps amplifies the drift; the 8× judger budget gives
the model room to degenerate visibly when it does collapse.

---

## Why this explains every observation

| Observation | Norm-drift hypothesis |
|-------------|----------------------|
| Mode collapse appears in all LatentMAS conditions (3/50) | ✅ Drift happens regardless of OBF |
| Same rate across bf16 and 4-bit | ✅ Drift is magnitude, not precision |
| Different *samples* collapse in different runs | ✅ Different initial hidden norms cross OOD at different iteration counts |
| Baseline never shows collapse | ✅ Baseline has no latent feedback loop, no drift |
| Specific patterns: whitespace, "3 3 3...", "DocumentDocument..." | ✅ Classic OOD-input failure modes — model defaults to attractors |
| Collapse appears late in generation (idx=20 at token 1798) | ✅ KV state pathological → generation gradually breaks down |
| `<think>` token appears in collapse outputs | ✅ Model still enters reasoning mode, but reasoning is corrupted |

No competing hypothesis explains all of these.

---

## The fix (sketch — not yet implemented)

In MLX:

```python
# Once, during initialization:
embed_weight = model.model.embed_tokens.weight  # [vocab, d_model]
target_norm = mx.mean(mx.linalg.norm(embed_weight, axis=1))

# Every latent step, before feeding h back:
h_norm = mx.linalg.norm(h, axis=-1, keepdims=True).maximum(1e-6)
h_rescaled = h * (target_norm / h_norm)
```

Estimated additional compute per step: ~one matrix norm + scalar
divide. Negligible vs. a full forward pass.

---

## Recommended next actions

### Option A (high value, ~30 min): Implement fix + verify
1. Add `apply_latent_realignment_mlx` function to MLX port (norm rescale).
2. Re-run 50-sample GSM8K bf16+OBF (the worst-affected condition).
3. If mode collapse drops from 3/50 to 0/50 → **bug confirmed and fixed**.
4. If mode collapse persists → other algorithmic divergences still in play.

### Option B (longer, ~2 hr): Align all hyperparameters
- Change `latent_steps` default 40 → 10
- Change judger max_new_tokens 2048 → 256
- Add `<think>` wrapping
- Add temp=0.7, top_p=0.95
- Re-run audit
- Compare against fresh "all-aligned" numbers

### Option C: Stop here, hand back to advisor
The bug is identified with evidence; advisor decides whether to fix
or formally write up the finding.

---

## What we can already say robustly

Even without implementing the fix:

1. **MLX port `latent_steps` diverges from the original algorithm.**
   The omitted norm rescaling is a real, documentable code-level
   divergence, not a matter of opinion.

2. **The observed mode collapse is consistent with what omitted norm
   rescaling would predict** — magnitude drift, OOD inputs, attractor
   tokens.

3. **The "92.2% on full GSM8K" reported in RESULTS.md was achieved despite
   this bug**, suggesting either: (a) the bug only manifests on
   ~6% of samples, (b) different sampling masks the collapses
   numerically, or (c) the bug interacts with the original ~94%
   number in ways that average out.

4. **RESULTS.md's "94% latent on 4-bit 50 samples" does not replicate.**
   Re-run on same model gives 86%. This is independent of the norm
   bug — could be sample subset, seed, or `<think>` mode interaction.

---

## Files added this milestone

```
audit-results/
├── baseline-gsm8k-50-4bit.jsonl
├── latent_mas-gsm8k-50-4bit.jsonl
├── latent_mas-gsm8k-50-bf16-nocompress.jsonl
└── MILESTONE-2.md            # this file
```

External reference checked:
- `/tmp/latentmas-orig/` (clone of Gen-Verse/LatentMAS, depth=1)

---

## Caveats

- 50-sample audit. Statistical power for 6% rate is limited; CI on the
  3/50 estimate is wide.
- Haven't yet verified the fix experimentally; the connection between
  "no rescaling" and "mode collapse" is mechanistic reasoning, strong
  but not yet experimental.
- Other algorithmic divergences (think wrapping, judger budget) also
  may contribute; the norm fix may not be sufficient alone.
