# Audit Report — latentmas-mlx

**Audited**: `suzuke/latentmas-mlx` @ commit `2e7faa7`
**Audit date**: 2026-05-16 / 2026-05-17
**Audit branch**: `audit/fix-norm-rescale`
**Audit method**: Empirical replication + code diff vs original `Gen-Verse/LatentMAS`

---

## TL;DR for the reader who has 60 seconds

1. **One bug identified, fix verified, isolated to GSM8K-style failures.**
   The MLX port's `latent_steps` function omits a hidden-state norm
   rescaling step that the original PyTorch LatentMAS applies every
   iteration. Without it, the model drifts into degenerate output
   (whitespace floods, single-token loops) on ~6% of GSM8K samples.

2. **Fix is 25 lines.** Two helper functions plus one line in the latent
   loop. Verified: 50-sample GSM8K Qwen3-4B bf16 accuracy goes from
   84% to **94%**, mode collapse 3/50 → **0/50**.

3. **The GPQA regression reported in RESULTS.md is a SEPARATE issue**
   and is not addressed by this fix. Norm rescaling doesn't change
   GPQA outcomes (within 1-sample noise on n=30), because GPQA's
   short multiple-choice answers don't have room for the rambling
   collapse pattern.

4. **The reported -1.8pp number is itself misleading**, comparing
   PyTorch FP16 baseline with MLX bf16 LatentMAS, mixing framework and
   precision confounds. Apples-to-apples (same MLX, same precision)
   shows the broken LatentMAS was -8pp behind baseline.

5. **The four other regressions in RESULTS.md remain unexplained**:
   Gemma E4B (-14pp), Gemma 26B ARC (-6pp), Gemma 26B GPQA (-6pp),
   Activation Graft (-43pp). The mode-collapse fix likely helps some
   of these but is unlikely to be the full story; each warrants its
   own audit.

---

## Why we audited

User's intuition (paraphrased):
> "RESULTS.md shows LatentMAS achieving 92.2% vs baseline 94% on GSM8K
> with 1.9× speedup. But several rows in the same table show LatentMAS
> *losing* by 6-20pp on harder benchmarks. The positive framing buries
> these regressions. Worth investigating whether the 92.2% is real
> and whether the regressions are method-fundamental or
> implementation-specific."

This audit pursued that question. Key prior:
> "Could the regressions be due to MLX implementation bugs, not the
> LatentMAS algorithm itself?"

---

## What's known vs. what we tested vs. what's new

### Known before audit (from `RESULTS.md`)

| Benchmark | Baseline | TextMAS | LatentMAS | Δ |
|-----------|----------|---------|-----------|---|
| Qwen3-4B GSM8K full bf16 | 94.0% (PyTorch) | — | 92.2% (MLX) | -1.8pp |
| Qwen3-4B GSM8K 50 4bit | 84.0% | 94.0% | 94.0% | +10.0pp |
| Gemma E4B GSM8K | 24.0% | 36.0% | 22.0% | -2.0pp (vs base) |
| Gemma 26B ARC | 96.0% | 96.0% | 90.0% | -6.0pp |
| Gemma 26B GPQA | 66.0% | 68.0% | 60.0% | -6.0pp |
| Qwen3-4B GPQA+OBF (30) | 37.0% | — | 17.0% | -20.0pp |
| Activation Graft (single-model) | 90.0% | — | 47.0% | -43.0pp |

Repo framing: "1.9× speedup with comparable accuracy".
Reality buried in the data: **negative ΔΔ in 5 of 7 benchmarks**.

### What we did (chronological)

| # | Experiment | Result |
|---|-----------|--------|
| EXP-001 | KV-cache injection kill-switch | Mechanism viable (3/3 hits in-process) |
| EXP-002 | Cross-process KV cache serialization | Information preserved (3/3 with 0.03% checksum drift) |
| EXP-003 | 50-sample GSM8K audit, bf16 | Discovered mode collapse 3/50 |
| EXP-004 | (skipped) | — |
| EXP-005 | Same, with `--no_compress` | Mode collapse 3/50 (unchanged) — OBF ruled out |
| EXP-006 | 50-sample GSM8K audit, 4-bit | Mode collapse 3/50 (unchanged) — precision ruled out |
| MILESTONE-2 | Code diff vs original repo | Identified missing `_apply_latent_realignment` |
| EXP-007 | 50-sample GSM8K with norm-rescale fix | Mode collapse 3/50 → **0/50**, +10pp accuracy |
| EXP-008 | 30-sample GPQA-Diamond baseline + fix | Fix neutral on GPQA (no mode collapse there to begin with) |
| EXP-009 | GPQA + `--latent_steps 10` (paper default) | Doesn't help; latent step count not the cause |

---

## The bug, in detail

### What's missing from MLX port

Original PyTorch (`Gen-Verse/LatentMAS/models.py:204-211`):

```python
def _apply_latent_realignment(self, hidden, model):
    matrix, target_norm = self._ensure_latent_realign_matrix(model, ...)
    aligned = torch.matmul(hidden.to(torch.float32), matrix)
    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned = aligned * (target_norm / aligned_norm)  # <-- THIS
    return aligned.to(hidden.dtype)
```

The matrix `M`:
- If `--latent_space_realign=True`: solved from embedding ↔ unembedding, an alignment matrix
- If `--latent_space_realign=False`: identity matrix

The norm rescaling `(target_norm / aligned_norm)` is **always applied**.
`target_norm = input_embeds.weight.norm(dim=1).mean()` — the mean L2
norm of an embedding row.

In MLX port `latentmas/run.py:197-212`:

```python
def latent_steps(model, prompt_ids, kv_cache, n_steps=20):
    _get_hidden_and_logits(model, prompt_ids[None], kv_cache)
    for _ in range(n_steps):
        if _ == 0:
            h = _get_hidden_and_logits(model, mx.array([[0]]), kv_cache)
        else:
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
        mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])
    return kv_cache
```

No rescaling. The raw post-norm hidden state `h` is fed back as
`inputs_embeds` directly. After 40 iterations × 3 agents = 120
unrescaled feedbacks, the state magnitude has drifted into pathological
territory.

### Why this produces mode collapse, mechanistically

1. Post-norm hidden state has different magnitude distribution than
   embedding-layer outputs.
2. Feeding it back as `inputs_embeds` means the next layer's attention
   sees OOD-magnitude "inputs".
3. Over many iterations, the drift accumulates.
4. Eventually the KV cache encodes a state that biases output
   distribution toward a small set of attractor tokens.
5. Judger generation, conditioned on this state, produces:
   - long sequences of single-token attractors (`"3 3 3 ..."`,
     `"DocumentDocumentDocument..."`)
   - or empty whitespace + degenerate guess

### Why the existence is precision-independent

Mode collapse rate is 3/50 in **both** bf16 and 4-bit. The drift
itself is a magnitude phenomenon — moving away from the in-distribution
manifold of input embeddings. Quantization adds noise but doesn't fix
the magnitude problem. (See `audit-results/notes/compareN.py` for
full 5-condition table.)

### Why the specific samples that collapse differ

Across runs (bf16-OBF, bf16-no-compress, 4-bit-OBF), the SAMPLES that
collapse differ but the RATE is constant (3/50):

| Setting | Collapsed indices |
|---------|------------------|
| bf16 + OBF | 11, 20, 40 |
| bf16 no compress | 11, 16, 40 |
| 4-bit + OBF | 3, 6, 35 |

Different initial hidden state magnitudes cause different samples to
cross the OOD threshold at different iteration counts. Stochastic
interaction with content, not "these particular questions are hard".

---

## The fix

`latentmas/run.py` adds:

```python
def _embedding_target_norm(model):
    inner = _get_inner_model(model)
    embed = inner.embed_tokens
    vocab_size = embed.weight.shape[0]
    all_ids = mx.arange(vocab_size)
    all_embs = embed(all_ids).astype(mx.float32)
    return mx.mean(mx.linalg.norm(all_embs, axis=-1))


def _rescale_to_target_norm(h, target_norm):
    h_fp32 = h.astype(mx.float32)
    h_norm = mx.maximum(mx.linalg.norm(h_fp32, axis=-1, keepdims=True),
                        mx.array(1e-6, dtype=mx.float32))
    return (h_fp32 * (target_norm / h_norm)).astype(h.dtype)
```

And in `latent_steps`:

```python
        else:
            h = _rescale_to_target_norm(h, target_norm)  # <-- added
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
```

Total diff: +25 lines, 0 removed.

### Important: handling quantized embeddings

`QuantizedEmbedding.weight` returns packed uint32, not actual fp16
weights. Naive `weight.norm()` gives 4.5e10 on 4-bit Qwen3-4B (vs ~1.1
on bf16). The fix calls `embed_tokens(all_ids)` which dequantizes
correctly. Verified: target_norm = 1.0985 on 4-bit vs 1.0974 on bf16
(0.1% difference).

---

## Verification

### Headline: 50-sample GSM8K Qwen3-4B bf16, max_tokens=2048

| Method | Acc | Mode Collapse | Truncated | Reasoning Errors | Avg Tokens |
|--------|-----|--------------|-----------|------------------|-----------|
| Baseline (bf16) | 92.0% | 0 | 4 | 0 | 989.8 |
| LatentMAS bf16 + OBF (broken) | 84.0% | **3** | 0 | 5 | 587.7 |
| LatentMAS bf16 no-compress (broken) | 88.0% | **3** | 0 | 3 | 561.8 |
| Baseline (4-bit) | 88.0% | 0 | 6 | 0 | 1088.6 |
| LatentMAS 4-bit + OBF (broken) | 86.0% | **3** | 1 | 3 | 726.6 |
| **LatentMAS bf16 + OBF (FIXED)** | **94.0%** | **0** | 1 | 2 | 529.4 |

### Recovery of the three previously-collapsed samples

| idx | Before fix | After fix |
|-----|-----------|-----------|
| 11 | 24 tokens, 96% whitespace, pred=456 (wrong) | 538 tokens, pred=**694** ✓ |
| 20 | 1798 tokens "3 3 3 ..." loop, pred=3 (wrong) | 700 tokens, pred=**15** ✓ |
| 40 | 1986 tokens "Document..." loop, pred=2 (wrong) | 481 tokens, pred=**8** ✓ |

### Failure modes in fixed run

3 remaining failures (94% accuracy):
- idx=12: truncation at 2048 tokens (also fails in baseline for same reason)
- idx=37: legitimate reasoning error
- idx=41: legitimate reasoning error (units misread)

None are mode collapse.

---

## What the fix does NOT solve

### GPQA-Diamond — separate issue

| Method (Qwen3-4B-4bit, 30 samples, max_tokens=2048) | Acc | Truncated |
|-----------------------------------------------------|-----|-----------|
| Baseline | 23.3% (7/30) | 27 |
| LatentMAS (fixed, lat_steps=40) | 20.0% (6/30) | 16 |
| LatentMAS (fixed, lat_steps=10 — paper default) | 16.7% (5/30) | 19 |

Differences are within 1-sample noise. Critically:
- **Mode collapse: 0 in all three conditions.** GPQA answers are
  single letters, so the rambling-collapse pattern can't manifest.
- Reducing latent steps 40 → 10 doesn't help either; 29/30 outcomes
  identical between the two settings.

The reported GPQA regression in RESULTS.md is therefore **NOT** caused
by norm drift, and is **NOT** addressed by this fix. It has a separate
cause, likely related to:
- agent prompt design (planner/critic/refiner are math-style, not
  multi-choice friendly)
- latent feedback diluting choice grounding
- model size effects (Qwen3-4B is small for GPQA-Diamond)

### Activation Graft — likely separate bug

`experiments/comm_activations.py` shows -43pp on same-model graft (90% → 47%).
For SAME-model graft, the captured activation at layer k should equal
the recomputed activation at layer k, making the "replacement" a near
no-op. The -43pp drop suggests either:
- subtle behavior difference between `cache=None` (step 1) and
  `cache=KVCache` (step 2) in MLX
- the captured activation is one layer-step ahead/behind of where it's
  reinjected
- docstring/code mismatch (docstring describes "Model A generates a
  completion" but code only does single forward pass)

Recorded as a separate suspected bug in `audit-results/notes/comm-activations-suspected-bug.md`. Not investigated in depth.

### Other RESULTS.md regressions

- **Gemma E4B GSM8K** -14pp: probably the norm fix helps (same mode
  collapse mechanism) but not tested due to weak model accuracy
  generally on math.
- **Gemma 26B ARC** -6pp: untested. ARC is multi-choice; if regression
  pattern matches GPQA (no mode collapse), the norm fix won't help.
- **Gemma 26B GPQA** -6pp: same as ARC.
- **OBF + GPQA -20pp catastrophic**: untested with fix. May be norm
  drift × compression interaction; possibly recovered by fix, but
  needs verification.

---

## Reproducibility

All audit artifacts:

```
~/research/latentmas-mlx-audit/
├── AUDIT-REPORT.md                       — this file
├── latentmas/run.py                      — current code (now with fix)
└── audit-results/
    ├── MILESTONE-1.md                    — initial mode collapse discovery
    ├── MILESTONE-2.md                    — bug identification via code diff
    ├── MILESTONE-3.md                    — fix verification on GSM8K
    ├── UPSTREAM-ISSUE-DRAFT.md           — ready-to-file GitHub issue
    ├── baseline-gsm8k-50-bf16.jsonl
    ├── baseline-gsm8k-50-4bit.jsonl
    ├── baseline-gpqa-30-4bit.jsonl
    ├── latent_mas-gsm8k-50-bf16.jsonl
    ├── latent_mas-gsm8k-50-bf16-nocompress.jsonl
    ├── latent_mas-gsm8k-50-bf16-fixed.jsonl
    ├── latent_mas-gsm8k-50-4bit.jsonl
    ├── latent_mas-gpqa-30-4bit-fixed.jsonl
    ├── latent_mas-gpqa-30-4bit-fixed-ls10.jsonl
    └── notes/
        ├── baseline-analysis.md
        ├── comm-activations-suspected-bug.md
        ├── compare.py
        ├── compareN.py
        └── compare_gpqa.py
```

Branch `audit/fix-norm-rescale` (8 commits ahead of upstream `main`).

---

## Caveats and limitations

- **Sample sizes are small.** 50 GSM8K samples and 30 GPQA samples.
  The 6% mode collapse rate has a wide confidence interval. A full-1319
  GSM8K run with the fix would be the next step for tightening numbers.
- **Single model audited.** All results on Qwen3-4B. Other models in
  RESULTS.md (Gemma E4B, Gemma 26B) not retested.
- **Temperature = 0.6**, default in MLX port. Original LatentMAS uses
  0.7. Sampling variance is real; some of the noise in our numbers is
  this. Not a primary confound for the mode-collapse finding, which is
  much larger than sampling noise.
- **Hyperparameter divergences from original paper were noted but
  not all aligned.** Latent_steps default 40 (orig 10),
  judger_max_new_tokens 2048 (orig 256), missing `<think>` wrapping.
  These may matter for other benchmarks. The norm fix is orthogonal
  to all of them.

---

## Recommended follow-up (in order)

1. **File issue / PR upstream** — `UPSTREAM-ISSUE-DRAFT.md` is ready.
2. **Replicate on full 1319-sample GSM8K** with the fix — confirms
   the 50-sample +10pp result holds at scale.
3. **Re-test OBF + GPQA -20pp** with the fix — most catastrophic of
   the unexplained regressions; possibly partially recovered.
4. **Audit Gemma E4B** — second-worst regression (-14pp). Likely the
   same mode-collapse mechanism since math reasoning benchmark.
5. **Investigate Activation Graft -43pp** — separate suspected bug,
   no audit work done yet.
6. **Align remaining hyperparameters** with paper (latent_steps=10,
   judger=256 tokens, `<think>` wrapping, temp=0.7, top_p=0.95) and
   re-run full audit. May or may not close additional gaps.
