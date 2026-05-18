# Milestone 9 — Naive SVD Alignment Doesn't Save Cross-Model Graft

**Date**: 2026-05-18
**Status**: Negative result. Cross-LoRA-style SVD subspace alignment didn't help cross-model latent graft.

---

## TL;DR

Tried Cross-LoRA-style training-free SVD alignment as a translator between Qwen3-1.7B-4bit and LLaMA3.2-1B-4bit hidden states. **Result: no improvement over no-alignment baseline. Both stay at 10% cross-model graft accuracy.**

This suggests the naive assumption "top-k PCs of two LLMs' embedding spaces encode similar concepts in rank order" is wrong for cross-arch model pairs.

---

## Setup

```bash
python experiments/comm_activations.py \
  --model_a mlx-community/Qwen3-1.7B-4bit \
  --model_b mlx-community/Llama-3.2-1B-Instruct-4bit \
  --task gsm8k --max_samples 30 \
  --graft_layer_a 23 --graft_layer_b 13 \
  --align svd       # NEW vs MILESTONE-8
```

Same as MILESTONE-8 but with `--align svd` added. SVD computed once at startup on first 8192 vocab rows of each model.

## Alignment formula

```
T = V_a^T @ V_b
where E_a = U_a S_a V_a^T,  E_b = U_b S_b V_b^T  (right singular vectors)
Apply to row-vector h_a:  h_aligned = h_a @ T
```

T shape: (2048, 2048) since both models have hidden=2048. Full-rank, no truncation.

The intent: rotate `h_a` from Qwen3's natural basis into LLaMA's natural basis, assuming the **i-th PC of each model encodes the same concept** (by rank).

## Results

| Method | MILESTONE-8 (no align) | MILESTONE-9 (SVD align) |
|--------|------------------------|------------------------|
| Qwen3-1.7B alone | 80.0% | 73.3% (sampling variance) |
| LLaMA3.2-1B alone | 20.0% | 16.7% |
| **cross_model_graft** | **10.0%** | **10.0%** |

The cross_model_graft column is unchanged. Within-method variance (Qwen3 80% → 73%, LLaMA 20% → 17%) reflects temp=0.6 sampling noise across runs.

## Why SVD alignment didn't help

The PC-by-index assumption fails for cross-arch pairs:

1. **Independent training trajectories**: Qwen3 and LLaMA were trained from different initializations on overlapping but distinct corpora with different optimization choices. Their "natural" axes in hidden space are rotations of each other only by coincidence.
2. **Variance ranking ≠ concept ranking**: Even when two models encode the same set of concepts in their hidden space, PC ordering depends on which concepts have most token-distribution variance — likely similar but not identical between models.
3. **Cross-LoRA's actual mechanism is more sophisticated**: their paper uses "Frobenius-optimal linear transformation" between rank-truncated subspaces, plus they apply this to LoRA weight transfer (which has different structure than hidden state translation).

## What this rules out

| Approach | Verdict |
|----------|---------|
| **No alignment** (MILESTONE-8) | 10% — well below baseline, expected |
| **Naive SVD PC alignment** (this milestone) | 10% — also no help |
| Procrustes with shared tokens | not tested, may work |
| Trained linear translator | not tested, expected to work (per Direct Semantic Communication, cosine 0.538) |
| Same-family cross-size graft (e.g., Qwen3-1.7B → Qwen3-4B) | not tested, more likely to work |

## What this confirms

Cross-arch latent space alignment is **harder than just SVD rotation**. The Direct Semantic Communication paper's cosine 0.538 (with their trained translator) is the realistic upper bound for current techniques. Anything better likely requires careful training of an alignment module.

## Suggested next steps

### Path A: Same-family cross-size (lowest effort, highest payoff probability)

Qwen3-1.7B-4bit → Qwen3-4B-4bit (both Qwen family, similar pretraining). Same alignment infrastructure, ~15 min run. If even same-family cross-size doesn't work, the method has deeper issues. If it works, cross-family is the harder challenge.

### Path B: Procrustes with anchor tokens

For each pair of vocabs, find common substrings (e.g., "Hello", "world", common English words). Capture both models' embeddings for these. Solve for orthogonal rotation R minimizing ||E_a R - E_b||_F over anchor rows. Apply to all hidden states.

### Path C: Direct Semantic Communication approach

Capture (h_a, h_b) pairs for the SAME prompt at corresponding layers. Train a small linear or MLP head to map h_a → h_b. ~1 hour of training, ~30 min of run.

### Path D: Honest stopping point

This audit's main goal (verify the 3 bugs + fixes) is complete. Cross-model alignment was a stretch goal. Document the negative result and stop here unless there's specific motivation to push further.

---

## Comparison with the field

Recent work I'm aware of on cross-model latent transfer:

- **Direct Semantic Communication** (arxiv 2511.03945, 2025): cosine 0.538 with trained translator on LLaMA-2-7B ↔ Mistral-7B
- **Cross-LoRA** (arxiv 2508.05232, 2025-08): training-free LoRA weight transfer (not hidden state) between heterogeneous LLMs
- **RecursiveMAS** (the audit's secondary target): trained InnerLink + OuterLink adapters; on RecursiveMAS-Light, fixed gets ~30% (paper claim 72%, gap unexplained)

Our 10% no-help-from-SVD result is consistent: zero-shot cross-model alignment without any training/data is essentially random. **Training is required**.

## Caveats

- **n=30 + temp=0.6**: result variance is large. The 10% ↔ 10% comparison is in the noise; could be 5-20% in either case. But the lack of any clear signal (e.g., we don't see SVD alignment reaching 15-25%) suggests no real effect.
- **Single architecture pair**: Qwen vs LLaMA may be unusually misaligned. Same-family pairs could behave differently.
- **Single graft layer pair**: 23/13 is 85% depth in both, but other layer choices (e.g., 70% depth, 50% depth) might give different results.
- **Greedy/temp=0 not tested**: noise-free comparison would be cleaner.

---

## Conclusion

Cross-LoRA-style SVD subspace alignment, applied as a hidden-state translator, **does not improve cross-model graft accuracy** in our zero-shot setup on a cross-architecture pair (Qwen3 → LLaMA). The naive PC-by-index assumption is too strong for arbitrary model pairs.

Higher-promise next steps: same-family pairs, Procrustes with anchor tokens, or trained linear translator. All are out of scope for this audit but documented for future work.
