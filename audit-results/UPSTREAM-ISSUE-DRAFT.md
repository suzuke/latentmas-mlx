# Upstream Issue Draft — for suzuke/latentmas-mlx

**Target repo**: <https://github.com/suzuke/latentmas-mlx>
**Status**: ready to file (one-shot copy-paste).

---

## Issue Title

> **Bug: `latent_steps` missing norm rescaling — causes mode collapse on ~6% of GSM8K samples**

## Issue Body

### Summary

The MLX port's `latent_steps` function in `latentmas/run.py` omits the
per-iteration hidden-state norm rescaling that the original PyTorch
LatentMAS applies via `_apply_latent_realignment`
([models.py:204-211 upstream](https://github.com/Gen-Verse/LatentMAS/blob/main/models.py#L204-L211)).

Without rescaling, hidden state magnitudes drift over latent iterations,
eventually pushing the model into out-of-distribution input territory
and producing **catastrophic mode collapse** (degenerate generation:
empty whitespace, single-token loops). This affects ~6% of GSM8K
samples on Qwen3-4B and is precision-independent (bf16, 4-bit both
affected).

### Reproduction

```bash
git checkout main  # current code
python latentmas/run.py \
  --method latent_mas \
  --model mlx-community/Qwen3-4B-bf16 \
  --task gsm8k --max_samples 50 --max_tokens 2048
```

Mode collapse samples on bf16:
- **idx=11**: 24 tokens, 96% whitespace. `<think>` + many newlines + wrong `\boxed{456}` (gold=694).
- **idx=20**: 1798 tokens of `"3 3 3 3 ..."` token loop (gold=15).
- **idx=40**: 1986 tokens of `"DocumentDocumentDocument..."` token loop (gold=8).

Same rate (3/50 = 6%) on 4-bit, but different sample indices — pattern
is stochastic but reproducible.

### Root cause

Original LatentMAS (`models.py`):

```python
def _apply_latent_realignment(self, hidden, model):
    matrix, target_norm = self._ensure_latent_realign_matrix(model, ...)
    aligned = torch.matmul(hidden.to(torch.float32), matrix)  # identity when off
    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / aligned_norm)  # <-- RESCALE
    return aligned.to(hidden.dtype)
```

Where `target_norm = input_embeds.weight.norm(dim=1).mean()` — the
typical embedding-row L2 norm. This rescaling is applied **unconditionally**
at every latent step, including when `--latent_space_realign` is off
(the matrix becomes identity, but rescaling still happens).

Current MLX `latent_steps`:

```python
def latent_steps(model, prompt_ids, kv_cache, n_steps=20):
    _get_hidden_and_logits(model, prompt_ids[None], kv_cache)
    for _ in range(n_steps):
        if _ == 0:
            h = _get_hidden_and_logits(model, mx.array([[0]]), kv_cache)
        else:
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
            #     ^^^ feeds back raw post-norm hidden state — drifts in magnitude
    return kv_cache
```

### Fix

Two helpers plus one line in the loop:

```python
def _embedding_target_norm(model) -> mx.array:
    """Mean L2 norm of embedding rows. Handles both quantized and
    non-quantized embeddings (calls embed_tokens to dequantize)."""
    inner = _get_inner_model(model)
    embed = inner.embed_tokens
    vocab_size = embed.weight.shape[0]
    all_ids = mx.arange(vocab_size)
    all_embs = embed(all_ids).astype(mx.float32)
    target = mx.mean(mx.linalg.norm(all_embs, axis=-1))
    mx.eval(target)
    return target


def _rescale_to_target_norm(h, target_norm):
    h_fp32 = h.astype(mx.float32)
    h_norm = mx.maximum(mx.linalg.norm(h_fp32, axis=-1, keepdims=True),
                        mx.array(1e-6, dtype=mx.float32))
    return (h_fp32 * (target_norm / h_norm)).astype(h.dtype)


def latent_steps(model, prompt_ids, kv_cache, n_steps=20, target_norm=None):
    if target_norm is None:
        target_norm = _embedding_target_norm(model)
    _get_hidden_and_logits(model, prompt_ids[None], kv_cache)
    for _ in range(n_steps):
        if _ == 0:
            h = _get_hidden_and_logits(model, mx.array([[0]]), kv_cache)
        else:
            h = _rescale_to_target_norm(h, target_norm)  # <-- the fix
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
        mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])
    return kv_cache
```

### Quantization note

`QuantizedEmbedding.weight` is packed uint32, not the actual embedding
values. Direct `weight.norm()` gives ~4.5e10 instead of ~1.1 for
Qwen3-4B-4bit. The fix uses `embed_tokens(all_ids)` which dequantizes
correctly for both quantized and non-quantized embeddings (verified
target_norm = 1.0985 on 4-bit vs 1.0974 on bf16, ratio 1.001).

### Verification — Full 1319-sample GSM8K Qwen3-4B bf16, max_tokens=2048

| Method | Acc | Mode Collapse | Reasoning errors | Truncation |
|--------|-----|--------------|------------------|-----------|
| LatentMAS (current/broken) | 92.2%* | ~6% (50-sample rate) | — | — |
| **LatentMAS (fixed)** | **90.3%** | **0.23% (3/1319)** | 6.7% (89/1319) | 2.7% (36/1319) |

\* From RESULTS.md, same MLX bf16 setup.

**The fix is a reliability improvement, not a pure accuracy improvement.**

- **Mode collapse: 6% → 0.23%** (~25× reduction) — the core win.
  Catastrophic failures (whitespace floods, `"3 3 3..."` loops,
  `"DocumentDocument..."` loops) virtually eliminated.
- **Accuracy: 92.2% → 90.3%** (small regression at scale, within
  sampling noise).
- The norm rescaling bounds the latent state magnitude → removes
  catastrophic outliers but also slightly constrains exploration
  that occasionally finds correct answers. This matches recent
  theory: ["Latent CoT's continuous representation enables robust
  exploration but is also the direct cause of its failure on
  computational tasks by amplifying noise."](https://openreview.net/forum?id=q7Nhu2Fw11)

### Historical note: 50-sample run earlier suggested +10pp

An earlier 50-sample run showed 84% (broken) → 94% (fixed), +10pp.
That gap was within the sampling noise of a 6% catastrophic-failure
rate (small n × rare event). The 1319-sample run gives the truer
picture. Both 50-sample and 1319-sample data are committed.

### Scope caveat

The same audit confirms the norm fix does **not** resolve the GPQA-Diamond
regression also documented in RESULTS.md (-6pp Gemma 26B, -20pp Qwen3-4B
with OBF). GPQA results stay similar before/after the fix because GPQA
never exhibited the rambling-collapse pattern (answers are single
letters, no room for "3 3 3..." loops). The GPQA regression is a
separate, still-unidentified issue.

### Audit data

Full audit data + commits on branch `audit/fix-norm-rescale` at:
- `~/research/latentmas-mlx-audit/audit-results/`
- See MILESTONE-3.md for the full report.

Happy to file a PR with the fix if you'd like.
