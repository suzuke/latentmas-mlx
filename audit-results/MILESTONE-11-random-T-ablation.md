# Milestone 11 — Random T Ablation Invalidates MILESTONE-10's Interpretation

**Date**: 2026-05-18
**Status**: ⚠ **CRITICAL CORRECTION** — Same-family graft "working" is not due to SVD alignment.

---

## TL;DR

Random orthogonal T (no model-derived information) gives the SAME cross-model graft accuracy as SVD T (90% in both cases) on Qwen3-1.7B → Qwen3-4B GSM8K. 

**MILESTONE-10's claim that "SVD alignment works for same-family pairs" is therefore not supported by the evidence**. The 90% accuracy reflects the receiver model (Qwen3-4B) absorbing any late-layer injection, not the alignment doing useful translation.

---

## The data

```
n=30 GSM8K, Qwen3-1.7B-4bit → Qwen3-4B-4bit, graft layers 23→30, temp=0.6
```

| Setup | A alone | B alone | cross_model_graft |
|-------|---------|---------|-------------------|
| MILESTONE-10 (SVD T) | 73.3% | 86.7% | **90.0%** (27/30) |
| **MILESTONE-11 (Random T)** | 76.7% | 90.0% | **90.0%** (27/30) |

Random T is generated as: SVD of random Gaussian matrix → orthogonal d×d → trimmed to [d_a, d_b]. No model-derived information. Used same RNG seed for reproducibility.

The graft accuracy is **identical between SVD-aligned and random-rotation** injection. Within-method variance across runs (A alone: 73.3% → 76.7%, B alone: 86.7% → 90.0%) reflects standard temp=0.6 sampling noise.

---

## Why MILESTONE-10's interpretation was wrong

I claimed: "Same-family SVD alignment captures real cross-model alignment".

Correct interpretation: **late-layer graft (layer 30 of 36 = 83% depth) is essentially ignored by the receiver model**. Qwen3-4B's remaining 6 layers + attention mechanism can "denoise" any reasonable-magnitude injection by attending to non-grafted positions.

This explains the previous results:
- **Same-family cross-size (90%)**: receiver strong enough to ignore injection → baseline accuracy
- **Cross-arch (Qwen3 → LLaMA-1B, 10%)**: receiver too weak to ignore injection → catastrophic
- The difference is **receiver model robustness**, not alignment quality

---

## Updated cross-model picture

| Setup | Strong baseline | Graft | Reading |
|-------|----------------|-------|---------|
| Same-model graft (Qwen3-8B → 8B, fixed) | 85% | 83% | No-op (math holds) |
| Cross-arch, no align (Qwen3 → LLaMA-1B) | 20% | 10% | Receiver too weak; injection corrupts |
| Cross-arch, SVD align (Qwen3 → LLaMA-1B) | 20% | 10% | SVD doesn't help; receiver still too weak |
| **Same-family, SVD align (Qwen3-1.7B → 4B)** | 87-90% | **90%** | **Receiver strong; injection absorbed** |
| **Same-family, Random T (Qwen3-1.7B → 4B)** | 90% | **90%** | **Same as SVD — alignment irrelevant** |

The cleanest reading across all 5 conditions: **graft accuracy ≈ receiver-model-alone accuracy when receiver is strong enough to recover; degrades catastrophically when receiver is weak**.

---

## What about Cross-LoRA paper's claims then?

Cross-LoRA (arxiv 2508.05232) claims training-free LoRA *weight* transfer works across heterogeneous LLMs. We do not contradict that — we tested hidden state translation, a different problem. Possible reasons their claim might still hold:

1. LoRA weight transfer is a different operation than hidden state injection
2. Their Frobenius-optimal projection is more sophisticated than naive V_a^T @ V_b
3. Their tasks may have different sensitivity to alignment quality

We don't have evidence either way about Cross-LoRA's claimed mechanism.

---

## What this means for the audit's overall narrative

The 3 bug fixes (Bugs #1, #2, #3) remain valid. The cross-model graft work was a stretch goal that:

✅ **Validated MILESTONE-7 / PR #2**: the framework now works (no more prefill corruption)

❌ **Did NOT successfully demonstrate working alignment for cross-model graft**:
- Naive SVD alignment doesn't help cross-arch (MILESTONE-9, 10% with SVD)
- Same-family "working" was a illusion of late-layer absorption (this milestone)
- A real alignment mechanism would need to be demonstrated by injecting at an EARLY layer where the receiver can't easily absorb random injection

---

## What would a real demonstration require?

To prove an alignment scheme works, run on a more sensitive setup:

1. **Early-layer graft** (e.g., layer 5 of receiver): the receiver still has many layers after the graft to be influenced by it. If random T then drops accuracy but SVD T preserves it, alignment is doing work.

2. **Zero injection ablation**: replace activation with zeros. If accuracy stays at 90%, injection is meaningless. If accuracy drops, late-layer graft does affect output (just not enough to be sensitive to alignment).

3. **Permuted activation ablation**: take the captured activation but permute its dimensions. Similar magnitude/distribution but no structure. Compare with random T.

4. **Stronger task sensitivity**: GSM8K with this Qwen3-4B is "too easy" — the receiver crushes the task even with junk injection. A harder task or weaker receiver would be more sensitive.

I will not run these now (the audit's core is complete and these are research-scope follow-ups), but they're the right next experiments.

---

## Honest self-assessment

I made the same mistake at MILESTONE-10 that I caught at MILESTONE-6:
- **Small-sample positive result** ("SVD alignment works at 90%!")
- **Without proper control** (random T)
- **Generalized too eagerly** ("validates same-family alignment")

The random T ablation took 15 minutes and invalidated the MILESTONE-10 narrative. Should have been the first thing I ran, not the last.

Lesson: when a "positive result" emerges from a single comparison without ablation, treat it as untested until controls run.

---

## What we can still claim

After this ablation, the cross-model graft work supports:

1. ✅ **PR #2 fix works** (the framework runs correctly post-fix)
2. ✅ **Cross-arch zero-shot latent transfer fails** (Qwen3 → LLaMA-1B drops to 10%)
3. ✅ **SVD alignment specifically does not help cross-arch** (MILESTONE-9)
4. ❓ **Same-family alignment: unclear** — current evidence suggests late-layer injection absorption rather than alignment work
5. ❌ **NOT supported**: "Same-family SVD alignment is a working training-free translator" (MILESTONE-10's framing)

The audit's main deliverables (3 bugs, fixes, scale validation, framework unblocking) are intact. The cross-model alignment portion is appropriately uncertain.

---

## Recommended next steps

For the user:

- **🅓 Stop here** (recommended) — audit's main goals complete; cross-model alignment shown to be more subtle than initially claimed; document honestly and move on.
- **🅐 Run zero-injection ablation** (~15 min) to settle whether the late-layer graft is truly ignored.
- **🅑 Run early-layer graft + random T ablation** (~30 min) to test if there's any setup where alignment matters.
- **🅒 Pivot back to PR #3 n=100 RecursiveMAS** (2 hours) — settle the post-norm fix's significance.

I recommend 🅓 — we've gotten substantial value from this audit, and pushing further on cross-model alignment is research, not bug-fixing.
