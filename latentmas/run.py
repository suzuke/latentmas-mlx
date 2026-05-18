"""
MLX LatentMAS — LatentMAS rewritten for Apple Silicon using MLX.

Usage:
  python mlx_latent_mas.py --method baseline --model mlx-community/Qwen3-4B-4bit --task gsm8k --max_samples 50
  python mlx_latent_mas.py --method text_mas --model mlx-community/Qwen3-4B-4bit --task gsm8k --max_samples 50
  python mlx_latent_mas.py --method latent_mas --model mlx-community/Qwen3-4B-4bit --task gsm8k --max_samples 50
"""
import argparse, datetime, json, re, subprocess, time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.generate import generate_step, cache as mlx_cache
from datasets import load_dataset


# ── Prompt builders ──────────────────────────────────────

SYSTEM = "You are a helpful assistant."

def _chat(tokenizer, messages, add_gen=True):
    """Apply chat template and return token ids."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_gen)
    if "gemma" in getattr(tokenizer, 'name_or_path', '').lower():
        kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **kwargs)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, mx.array(ids)

def build_baseline_prompt(q, task):
    suffix = "\nYour final answer must be selected from A,B,C,D. For example \\boxed{A}. Do not add any other contents inside the box." if task in ("arc_challenge", "gpqa") else ""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Target Question: {q}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.{suffix}

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."""},
    ]

AGENT_PROMPTS = {
    "planner": """You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {q}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:""",

    "critic": """
Question: {q}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:""",

    "refiner": """
Question: {q}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:""",
}

def build_agent_prompt(role, q):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": AGENT_PROMPTS[role].format(q=q)},
    ]

def build_judger_prompt(q, task):
    if task in ("arc_challenge", "gpqa"):
        suffix = "\nYour final answer must be selected from A,B,C,D. For example \\boxed{A}. Do not add any other contents inside the box."
    else:
        suffix = ""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Target Question: {q}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.{suffix}

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."""},
    ]

def build_text_agent_prompt(role, q, context=""):
    prompts = {
        "planner": f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{q}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:""",

        "critic": f"""You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{q}

## Plan from Planner Agent:
{context}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:""",

        "refiner": f"""You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{q}

## Original Plan and Critic Feedback:
{context}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:""",

        "judger": f"""Target Question: {q}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{context}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.""",
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompts[role]},
    ]


# ── Core: latent step generation ─────────────────────────

def _get_inner_model(model):
    """Get the inner model that has embed_tokens and layers."""
    if hasattr(model, 'language_model'):
        return model.language_model.model  # Gemma 4
    return model.model  # Qwen, LLaMA


def _get_hidden_and_logits(model, tokens_or_embeds, kv_cache, is_embed=False):
    """Run forward, return (hidden_state, logits). hidden_state is pre-lm_head."""
    inner = _get_inner_model(model)
    if is_embed:
        h = inner(None, cache=kv_cache, input_embeddings=tokens_or_embeds)
    else:
        h = inner(tokens_or_embeds, cache=kv_cache)
    # h is norm(last_hidden) — this is what we want for latent steps
    return h


def _embedding_target_norm(model) -> mx.array:
    """Compute the mean L2 norm of the embedding rows.

    The original LatentMAS rescales every fed-back hidden state to this
    target via `_apply_latent_realignment` (models.py:204-211 in the
    Gen-Verse/LatentMAS repo). The MLX port previously omitted this step,
    causing hidden state magnitude to drift over many latent iterations
    and pushing the model into out-of-distribution input territory.

    Note: for quantized models (e.g. 4-bit), `embed_tokens.weight` is the
    packed uint32 representation, not the actual fp16 weight. We call
    `embed_tokens(...)` on all vocab IDs to dequantize before computing
    norms. This works correctly for both quantized and non-quantized
    embeddings.
    """
    inner = _get_inner_model(model)
    embed = inner.embed_tokens
    vocab_size = embed.weight.shape[0]
    all_ids = mx.arange(vocab_size)
    all_embs = embed(all_ids).astype(mx.float32)
    # row-wise L2 norm then mean
    target = mx.mean(mx.linalg.norm(all_embs, axis=-1))
    mx.eval(target)
    return target


def _rescale_to_target_norm(h: mx.array, target_norm: mx.array) -> mx.array:
    """Rescale last-axis vectors of h to have norm == target_norm."""
    h_fp32 = h.astype(mx.float32)
    h_norm = mx.linalg.norm(h_fp32, axis=-1, keepdims=True)
    h_norm = mx.maximum(h_norm, mx.array(1e-6, dtype=mx.float32))
    rescaled = h_fp32 * (target_norm / h_norm)
    return rescaled.astype(h.dtype)


def latent_steps(model, prompt_ids, kv_cache, n_steps=20, target_norm=None):
    """Run n latent steps: feed hidden states back as input embeddings, accumulating KV cache.

    Now applies norm rescaling each iteration to match the original
    LatentMAS algorithm (see _embedding_target_norm docstring).
    Pass `target_norm` precomputed to avoid recomputing per agent.
    """
    if target_norm is None:
        target_norm = _embedding_target_norm(model)

    # Prefill prompt into cache
    _get_hidden_and_logits(model, prompt_ids[None], kv_cache)
    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])

    for _ in range(n_steps):
        if _ == 0:
            h = _get_hidden_and_logits(model, mx.array([[0]]), kv_cache)
        else:
            # Rescale h from previous iteration to target embedding norm
            # before feeding it back as inputs_embeds. This matches the
            # original PyTorch implementation's _apply_latent_realignment.
            h = _rescale_to_target_norm(h, target_norm)
            h = _get_hidden_and_logits(model, h, kv_cache, is_embed=True)
        mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])

    return kv_cache


def compute_attention_importance(kv_cache, prompt_start, prompt_len):
    """Compute importance of prompt tokens using attention from latent tokens.
    
    For each layer, compute dot product between the last few KV entries (latent steps)
    and the prompt keys, then aggregate across layers and heads.
    Returns importance scores [prompt_len].
    """
    all_scores = []
    for lc in kv_cache:
        if not hasattr(lc, 'state') or lc.offset == 0:
            continue
        keys, _ = lc.state  # [1, H, seq_len, D]
        seq_len = lc.offset
        prompt_end = prompt_start + prompt_len
        if prompt_end > seq_len:
            continue
        
        # Prompt keys: [H, prompt_len, D]
        k_prompt = keys[0, :, prompt_start:prompt_end, :]
        
        # Use last 5 tokens as "query" (these are latent step tokens)
        n_query = min(5, seq_len - prompt_end)
        if n_query <= 0:
            # No latent tokens yet, fall back to key norms
            scores = mx.sum(k_prompt * k_prompt, axis=(0, 2))  # [prompt_len]
        else:
            k_query = keys[0, :, seq_len - n_query:seq_len, :]  # [H, n_query, D]
            # Attention scores: dot product between query and prompt keys
            # [H, n_query, D] @ [H, D, prompt_len] -> [H, n_query, prompt_len]
            attn = mx.matmul(k_query, mx.transpose(k_prompt, (0, 2, 1)))
            # Sum across query positions and heads
            scores = mx.sum(attn, axis=(0, 1))  # [prompt_len]
        
        all_scores.append(scores)
    
    if not all_scores:
        return mx.ones(prompt_len)
    
    # Average across layers
    total = all_scores[0]
    for s in all_scores[1:]:
        total = total + s
    return total


def compress_kv_obf(kv_cache, prompt_start, prompt_len, keep_k=32, pca_rank=8, sink_size=4):
    """Compress CURRENT AGENT's prompt KV states using OBF.
    
    Key fixes vs naive implementation:
    1. Always preserve first sink_size tokens of the prompt (attention sink)
    2. Use attention-based importance from latent steps for remaining tokens
    3. Apply OBF residual injection on values only
    """
    eps = 1e-12
    prompt_end = prompt_start + prompt_len
    actual_keep = min(keep_k, prompt_len)
    if prompt_len <= actual_keep:
        return kv_cache

    # Compute importance using attention from latent steps
    importance = compute_attention_importance(kv_cache, prompt_start, prompt_len)
    mx.eval(importance)

    # Protect sink tokens (first sink_size tokens of this prompt)
    sink = min(sink_size, prompt_len)
    # Set sink tokens to very high importance so they're always kept
    sink_boost = mx.concatenate([
        mx.full((sink,), 1e10),
        mx.zeros((prompt_len - sink,))
    ])
    importance = importance + sink_boost
    mx.eval(importance)

    top_indices = mx.sort(mx.argpartition(importance, kth=prompt_len - actual_keep)[-actual_keep:])
    mx.eval(top_indices)
    top_set = set(top_indices.tolist())
    del_list = [i for i in range(prompt_len) if i not in top_set]

    if len(del_list) == 0:
        return kv_cache
    del_indices = mx.array(del_list)

    for layer_cache in kv_cache:
        if not hasattr(layer_cache, 'state') or layer_cache.offset == 0:
            continue
        keys, values = layer_cache.state
        seq_len = layer_cache.offset
        B, H, S, D = keys.shape

        # Per-head OBF
        new_v_heads = []
        for h in range(H):
            v_prompt = values[0, h, prompt_start:prompt_end, :]
            v_keep = v_prompt[top_indices].astype(mx.float32)
            v_disc = v_prompt[del_indices].astype(mx.float32)

            # SVD for orthonormal basis of retained span
            U_k, S_k, Vh_k = mx.linalg.svd(v_keep, stream=mx.cpu)
            mx.eval(Vh_k)
            r = min(actual_keep, D, Vh_k.shape[0])
            Q = Vh_k[:r, :].T

            proj = v_disc @ Q
            residual = v_disc - proj @ Q.T
            resid_norm = mx.sqrt(mx.sum(residual * residual))
            mx.eval(resid_norm)

            if float(resid_norm) < 1e-10:
                new_v_heads.append(v_keep.astype(values.dtype))
                continue

            U_r, S_r, Vh_r = mx.linalg.svd(residual, stream=mx.cpu)
            mx.eval(Vh_r)
            p = min(pca_rank, Vh_r.shape[0])
            C = Vh_r[:p, :]
            r_mean = mx.mean(residual, axis=0)
            delta = (r_mean @ C.T) @ C

            imp_keep = float(mx.sum(importance[top_indices]))
            imp_del = float(mx.sum(importance[del_indices]))
            scale = imp_del / (imp_keep + eps)
            delta = delta * scale

            v_keep_new = v_keep + delta[None, :]
            new_v_heads.append(v_keep_new.astype(values.dtype))

        # Reconstruct
        k_prompt_kept = keys[0, :, prompt_start:prompt_end, :][:, top_indices, :]
        v_prompt_kept = mx.stack(new_v_heads, axis=0)

        parts_k, parts_v = [], []
        if prompt_start > 0:
            parts_k.append(keys[0, :, :prompt_start, :])
            parts_v.append(values[0, :, :prompt_start, :])
        parts_k.append(k_prompt_kept)
        parts_v.append(v_prompt_kept)
        if seq_len > prompt_end:
            parts_k.append(keys[0, :, prompt_end:seq_len, :])
            parts_v.append(values[0, :, prompt_end:seq_len, :])

        new_k = mx.concatenate(parts_k, axis=1)
        new_v = mx.concatenate(parts_v, axis=1)
        layer_cache.state = (new_k[None], new_v[None])
        layer_cache.offset = new_k.shape[1]

    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])
    return kv_cache
    prompt_end = prompt_start + prompt_len
    actual_keep = min(keep_k, prompt_len)
    if prompt_len <= actual_keep:
        return kv_cache

    for layer_cache in kv_cache:
        if not hasattr(layer_cache, 'state') or layer_cache.offset == 0:
            continue
        keys, values = layer_cache.state
        seq_len = layer_cache.offset
        B, H, S, D = keys.shape

        # Importance of current agent's prompt tokens — use attention from latent steps
        importance = compute_attention_importance(kv_cache, prompt_start, prompt_len)
        mx.eval(importance)

        top_indices = mx.sort(mx.argpartition(importance, kth=prompt_len - actual_keep)[-actual_keep:])
        mx.eval(top_indices)
        top_set = set(top_indices.tolist())
        del_list = [i for i in range(prompt_len) if i not in top_set]

        if len(del_list) == 0:
            continue
        del_indices = mx.array(del_list)

        # Per-head OBF
        new_v_heads = []
        for h in range(H):
            v_prompt = values[0, h, prompt_start:prompt_end, :]
            v_keep = v_prompt[top_indices].astype(mx.float32)
            v_disc = v_prompt[del_indices].astype(mx.float32)

            U_k, S_k, Vh_k = mx.linalg.svd(v_keep, stream=mx.cpu)
            mx.eval(Vh_k)
            r = min(actual_keep, D, Vh_k.shape[0])
            Q = Vh_k[:r, :].T

            proj = v_disc @ Q
            residual = v_disc - proj @ Q.T
            resid_norm = mx.sqrt(mx.sum(residual * residual))
            mx.eval(resid_norm)

            if float(resid_norm) < 1e-10:
                new_v_heads.append(v_keep.astype(values.dtype))
                continue

            U_r, S_r, Vh_r = mx.linalg.svd(residual, stream=mx.cpu)
            mx.eval(Vh_r)
            p = min(pca_rank, Vh_r.shape[0])
            C = Vh_r[:p, :]
            r_mean = mx.mean(residual, axis=0)
            delta = (r_mean @ C.T) @ C

            imp_keep = float(mx.sum(importance[top_indices]))
            imp_del = float(mx.sum(importance[del_indices]))
            scale = imp_del / (imp_keep + eps)
            delta = delta * scale

            v_keep_new = v_keep + delta[None, :]
            new_v_heads.append(v_keep_new.astype(values.dtype))

        # Reconstruct: [before_prompt] + [kept_prompt] + [after_prompt]
        k_prompt_kept = keys[0, :, prompt_start:prompt_end, :][:, top_indices, :]
        v_prompt_kept = mx.stack(new_v_heads, axis=0)

        parts_k = []
        parts_v = []
        if prompt_start > 0:
            parts_k.append(keys[0, :, :prompt_start, :])
            parts_v.append(values[0, :, :prompt_start, :])
        parts_k.append(k_prompt_kept)
        parts_v.append(v_prompt_kept)
        if seq_len > prompt_end:
            parts_k.append(keys[0, :, prompt_end:seq_len, :])
            parts_v.append(values[0, :, prompt_end:seq_len, :])

        new_k = mx.concatenate(parts_k, axis=1)
        new_v = mx.concatenate(parts_v, axis=1)
        layer_cache.state = (new_k[None], new_v[None])
        layer_cache.offset = new_k.shape[1]

    mx.eval([c.state for c in kv_cache if hasattr(c, 'state')])
    return kv_cache


def generate_text(model, tokenizer, prompt_ids, kv_cache=None, max_tokens=2048, temp=0.6):
    """Generate text, optionally with a pre-filled KV cache."""
    if kv_cache is None:
        kv_cache = mlx_cache.make_prompt_cache(model)

    tokens = []
    for token, _ in generate_step(
        prompt_ids, model,
        max_tokens=max_tokens,
        prompt_cache=kv_cache,
        sampler=lambda x: mx.random.categorical(x * (1.0 / temp)),
    ):
        t = token.item() if hasattr(token, 'item') else int(token)
        eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]
        if t in eos_ids:
            break
        tokens.append(t)

    return tokenizer.decode(tokens)


# ── Methods ──────────────────────────────────────────────

def run_baseline(model, tokenizer, question, task, max_tokens):
    msgs = build_baseline_prompt(question, task)
    _, ids = _chat(tokenizer, msgs)
    text = generate_text(model, tokenizer, ids, max_tokens=max_tokens)
    return text, 0  # 0 latent tokens

def run_text_mas(model, tokenizer, question, task, max_tokens):
    total_tokens = 0
    context = ""
    for role in ["planner", "critic", "refiner"]:
        msgs = build_text_agent_prompt(role, question, context)
        _, ids = _chat(tokenizer, msgs)
        resp = generate_text(model, tokenizer, ids, max_tokens=max_tokens)
        total_tokens += len(tokenizer.encode(resp))
        context = resp

    # Judger
    msgs = build_text_agent_prompt("judger", question, context)
    _, ids = _chat(tokenizer, msgs)
    resp = generate_text(model, tokenizer, ids, max_tokens=max_tokens)
    total_tokens += len(tokenizer.encode(resp))
    return resp, total_tokens

def run_latent_mas(model, tokenizer, question, task, max_tokens, n_latent=20, adaptive_compress=True):
    kv = mlx_cache.make_prompt_cache(model)

    for role in ["planner", "critic", "refiner"]:
        msgs = build_agent_prompt(role, question)
        _, ids = _chat(tokenizer, msgs)
        prompt_len = len(ids)
        prompt_start = kv[0].offset
        kv = latent_steps(model, ids, kv, n_steps=n_latent)

        # Adaptive compression: only compress if prompt is long enough
        if adaptive_compress and prompt_len > 200:
            keep_k = max(32, int(prompt_len * 0.3))
            kv = compress_kv_obf(kv, prompt_start, prompt_len, keep_k=keep_k)

    msgs = build_judger_prompt(question, task)
    _, ids = _chat(tokenizer, msgs)
    text = generate_text(model, tokenizer, ids, kv_cache=kv, max_tokens=max_tokens)
    return text, 0


def run_latent_mas_obf(model, tokenizer, question, task, max_tokens, n_latent=20, keep_k=32):
    kv = mlx_cache.make_prompt_cache(model)

    for role in ["planner", "critic", "refiner"]:
        msgs = build_agent_prompt(role, question)
        _, ids = _chat(tokenizer, msgs)
        prompt_len = len(ids)
        prompt_start = kv[0].offset  # where this agent's prompt starts in the cache
        kv = latent_steps(model, ids, kv, n_steps=n_latent)
        # Compress only THIS agent's prompt tokens
        kv = compress_kv_obf(kv, prompt_start, prompt_len, keep_k=keep_k)

    msgs = build_judger_prompt(question, task)
    _, ids = _chat(tokenizer, msgs)
    text = generate_text(model, tokenizer, ids, kv_cache=kv, max_tokens=max_tokens)
    return text, 0


# ── Answer extraction ────────────────────────────────────

def extract_answer(text, task):
    if task == "arc_challenge" or task == "gpqa":
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxes:
            return boxes[-1].strip().upper()
        for letter in ["D", "C", "B", "A"]:
            if letter in text.upper():
                return letter
        return ""
    # GSM8K / numeric
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", boxes[-1])
        return nums[0] if nums else boxes[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""

def extract_gold(answer_text):
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    return m.group(1) if m else ""


# ── Data loading ─────────────────────────────────────────

def _numeric_equal(pred, gold):
    """Compare answers, handling numeric equivalence (e.g., 29.00 == 29)."""
    if not pred or not gold:
        return False
    if pred.strip().lower() == gold.strip().lower():
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return False


def load_data(task, max_samples):
    if task == "gsm8k":
        ds = list(load_dataset("gsm8k", "main", split="test"))[:max_samples]
        return [{"question": d["question"], "gold": extract_gold(d["answer"])} for d in ds]
    elif task == "arc_challenge":
        ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))[:max_samples]
        items = []
        for d in ds:
            labels = d["choices"]["label"]
            texts = d["choices"]["text"]
            choices = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
            items.append({"question": f"{d['question']}\n{choices}", "gold": d["answerKey"].upper()})
        return items
    elif task == "gpqa":
        ds = list(load_dataset("fingertap/GPQA-Diamond", split="test"))[:max_samples]
        return [{"question": d["question"], "gold": d["answer"].strip().upper()} for d in ds]
    raise ValueError(f"Unknown task: {task}")


# ── Main ─────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "latent_mas_obf"], required=True)
    p.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--task", choices=["gsm8k", "arc_challenge", "gpqa"], default="gsm8k")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--latent_steps", type=int, default=40)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--no_compress", action="store_true", help="Disable adaptive KV compression in latent_mas")
    p.add_argument("--save_outputs", type=str, default=None, help="If set, write per-sample (question, gold, raw_response, pred, ok) as JSONL to this path")
    p.add_argument("--resume", action="store_true", help="If --save_outputs exists, skip already-processed indices and append new results")
    args = p.parse_args()

    # Resume support: if file exists and --resume, read existing indices + stats.
    done_indices: set[int] = set()
    resume_correct = 0
    resume_time = 0.0
    resume_tokens = 0
    if args.save_outputs and args.resume:
        import os
        if os.path.exists(args.save_outputs):
            with open(args.save_outputs) as f:
                for line in f:
                    d = json.loads(line)
                    if "_meta" in d:
                        continue
                    if "index" in d:
                        done_indices.add(d["index"])
                        if d.get("correct"):
                            resume_correct += 1
                        resume_time += d.get("elapsed_sec", 0)
                        resume_tokens += d.get("out_tokens", 0)
            print(f"[resume] Found {len(done_indices)} completed samples ({resume_correct} correct) in {args.save_outputs}")

    # Open file: append if resuming with existing data, otherwise write fresh
    if args.save_outputs:
        mode = "a" if done_indices else "w"
        out_file = open(args.save_outputs, mode)
        if mode == "w":
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
                ).strip()
            except Exception:
                git_commit = "unknown"
            out_file.write(json.dumps({
                "_meta": True,
                "method": args.method,
                "model": args.model,
                "task": args.task,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "latent_steps": args.latent_steps,
                "temp": args.temp,
                "no_compress": args.no_compress,
                "git_commit": git_commit,
                "timestamp": datetime.datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")
            out_file.flush()
    else:
        out_file = None

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)

    data = load_data(args.task, args.max_samples)
    print(f"Running {args.method} on {args.task} ({len(data)} samples)...")

    correct = resume_correct
    total_time = resume_time
    total_out_tokens = resume_tokens

    try:
        for i, item in enumerate(data):
            if i in done_indices:
                continue
            t0 = time.time()

            if args.method == "baseline":
                resp, _ = run_baseline(model, tokenizer, item["question"], args.task, args.max_tokens)
            elif args.method == "text_mas":
                resp, _ = run_text_mas(model, tokenizer, item["question"], args.task, args.max_tokens)
            elif args.method == "latent_mas":
                resp, _ = run_latent_mas(model, tokenizer, item["question"], args.task, args.max_tokens, args.latent_steps, adaptive_compress=not args.no_compress)
            elif args.method == "latent_mas_obf":
                resp, _ = run_latent_mas_obf(model, tokenizer, item["question"], args.task, args.max_tokens, args.latent_steps, keep_k=32)

            elapsed = time.time() - t0
            total_time += elapsed

            out_tokens = len(tokenizer.encode(resp))
            total_out_tokens += out_tokens

            pred = extract_answer(resp, args.task)
            gold = item["gold"]
            ok = _numeric_equal(pred, gold)
            correct += ok

            print(f"  [{i+1}/{len(data)}] {'✓' if ok else '✗'} pred={pred} gold={gold} time={elapsed:.1f}s tokens={out_tokens}")

            if out_file is not None:
                out_file.write(json.dumps({
                    "index": i,
                    "question": item["question"],
                    "gold": gold,
                    "raw_response": resp,
                    "extracted_pred": pred,
                    "correct": bool(ok),
                    "elapsed_sec": round(elapsed, 2),
                    "out_tokens": out_tokens,
                }, ensure_ascii=False) + "\n")
                out_file.flush()
    finally:
        if out_file is not None:
            out_file.close()

    acc = correct / len(data)
    avg_time = total_time / len(data)
    avg_tokens = total_out_tokens / len(data)

    result = {
        "method": args.method,
        "model": args.model,
        "task": args.task,
        "samples": len(data),
        "accuracy": round(acc, 4),
        "correct": correct,
        "avg_time_sec": round(avg_time, 2),
        "avg_tokens": round(avg_tokens, 1),
        "total_time_sec": round(total_time, 2),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
