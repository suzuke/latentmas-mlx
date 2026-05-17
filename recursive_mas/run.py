"""
MLX RecursiveMAS — Port of RecursiveMAS Sequential-Light for Apple Silicon.

Heterogeneous multi-agent latent collaboration:
  Planner (Qwen3-1.7B) → Critic (LLaMA3.2-1B) → Solver (Qwen2.5-Math-1.5B)

Each agent has a trained InnerLink adapter; cross-model OuterLink adapters
bridge different hidden spaces.

Usage:
  python mlx_recursive_mas.py --task gsm8k --max_samples 30
"""
import argparse, json, os, re, time
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.generate import generate_step, cache as mlx_cache
from huggingface_hub import snapshot_download
from datasets import load_dataset


# ── MLX Adapter modules ──────────────────────────────────

class Adapter(nn.Module):
    """InnerLink: LayerNorm → Linear → GELU → Linear → residual → LayerNorm"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.pre_ln(x)
        out = self.proj2(nn.gelu(self.proj1(h)))
        return self.post_ln(x + out)


class CrossModelAdapter(nn.Module):
    """OuterLink: LayerNorm(in) → MLP(in→2*out→out) + residual_proj(in→out) → LayerNorm(out)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = out_dim * 2
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.ln_source = nn.LayerNorm(in_dim)
        self.ln_target = nn.LayerNorm(out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln_source(x)
        out = self.proj2(nn.gelu(self.proj1(h)))
        return self.ln_target(out + self.residual_proj(x))


# ── Weight loading ───────────────────────────────────────

def _pt_to_mlx(pt_path: str) -> Dict[str, mx.array]:
    """Load PyTorch .pt file and convert to MLX arrays."""
    import torch
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    return {k: mx.array(v.float().numpy()) for k, v in state.items()}


def load_inner_adapter(repo_path: Path, hidden_size: int, task: str = "math") -> Adapter:
    """Load InnerLink adapter from a RecursiveMAS agent repo."""
    # Try task-specific adapter first, then generic
    pt_path = repo_path / f"adapter({task}).pt"
    if not pt_path.exists():
        pt_path = repo_path / "adapter.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No adapter.pt found in {repo_path}")

    adapter = Adapter(hidden_size)
    weights = _pt_to_mlx(str(pt_path))
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    return adapter


def load_outer_adapters(repo_path: Path, agent_dims: Dict[str, int], task: str = "math") -> Dict[str, CrossModelAdapter]:
    """Load OuterLink adapters from the Outerlinks repo."""
    config_path = repo_path / "outerlink_config.json"
    with open(config_path) as f:
        config = json.load(f)

    if "tasks" in config:
        adapters_list = config["tasks"][task]["adapters"]
    else:
        adapters_list = config["adapters"]

    # Map legacy_key to (src_role, dst_role)
    key_to_roles = {
        "outer_12": ("planner", "critic"),
        "outer_23": ("critic", "solver"),
        "outer_31": ("solver", "planner"),
    }

    adapters = {}
    for item in adapters_list:
        key = item["legacy_key"]
        pt_path = repo_path / item["filename"]
        if key not in key_to_roles:
            continue
        src_role, dst_role = key_to_roles[key]
        in_dim = agent_dims[src_role]
        out_dim = agent_dims[dst_role]

        adapter = CrossModelAdapter(in_dim, out_dim)
        weights = _pt_to_mlx(str(pt_path))
        adapter.load_weights(list(weights.items()))
        mx.eval(adapter.parameters())
        adapters[key] = adapter

    return adapters


# ── System loading ───────────────────────────────────────

REPOS_LIGHT = {
    "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
    "critic": "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
    "solver": "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
    "outer": "RecursiveMAS/Sequential-Light-Outerlinks",
}

REPOS_SCALED = {
    "planner": "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
    "critic": "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
    "solver": "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
    "outer": "RecursiveMAS/Sequential-Scaled-Outerlinks",
}

REPOS = REPOS_LIGHT  # default, overridden by --style


def load_system(task: str = "math", style: str = "light"):
    """Load all models and adapters."""
    repos = REPOS_SCALED if style == "scaled" else REPOS_LIGHT
    print(f"Loading RecursiveMAS Sequential-{style.capitalize()} system...")

    # Download and load models
    agents = {}
    for role, repo_id in repos.items():
        if role == "outer":
            continue
        print(f"  Loading {role}: {repo_id}")
        # Some RecursiveMAS repos have incomplete configs; provide defaults
        model_config = {}
        if "Qwen3" in repo_id and "Qwen3.5" not in repo_id:
            model_config["rope_theta"] = 1000000.0
        if "Qwen3.5" in repo_id:
            model_config["model_type"] = "qwen3_5"
        model, tok = mlx_lm.load(repo_id, model_config=model_config)
        # Convert to float32 for numerical accuracy (adapters trained in float32)
        model = model.astype(mx.float32) if hasattr(model, 'astype') else model
        inner_model = model.model if hasattr(model, 'model') else model
        if hasattr(inner_model, 'embed_tokens'):
            hidden_size = inner_model.embed_tokens.weight.shape[1]
        elif hasattr(model, 'language_model'):
            hidden_size = model.language_model.model.embed_tokens.weight.shape[1]
        else:
            hidden_size = model.args.hidden_size
        agents[role] = {"model": model, "tok": tok, "hidden_size": hidden_size}

    # Load inner adapters
    for role in ["planner", "critic", "solver"]:
        repo_path = Path(snapshot_download(repos[role]))
        adapter = load_inner_adapter(repo_path, agents[role]["hidden_size"], task)
        agents[role]["inner"] = adapter
        print(f"  Loaded {role} InnerLink (hidden={agents[role]['hidden_size']})")

    # Load outer adapters
    outer_path = Path(snapshot_download(repos["outer"]))
    agent_dims = {role: agents[role]["hidden_size"] for role in agents}
    outers = load_outer_adapters(outer_path, agent_dims, task)
    print(f"  Loaded {len(outers)} OuterLinks")

    return agents, outers


# ── Latent rollout ───────────────────────────────────────

def _get_inner(model):
    """Get the inner model that has embed_tokens and layers."""
    if hasattr(model, 'language_model'):
        return model.language_model.model  # Gemma
    if hasattr(model, 'model'):
        return model.model  # Qwen, LLaMA
    return model


def _forward_get_raw_hidden(model, input_embeds):
    """Run model forward and return POST-norm hidden state.

    Matches the original PyTorch RecursiveMAS (inference_mas.py:868-869):
        outputs = model(..., output_hidden_states=True, ...)
        last_hidden = outputs.hidden_states[-1][:, -1, :]

    In modern HuggingFace transformers (Qwen2/Qwen3/LLaMA), the LAST
    entry of `outputs.hidden_states` is appended AFTER `self.norm` —
    i.e. it is POST-final-norm. The InnerLink adapter is trained on
    these POST-norm features.

    Previous version of this function omitted `inner.norm` and returned
    PRE-norm, causing a distribution shift at the adapter's input that
    compounds across the ~40 latent_steps iterations.
    See audit-results/MILESTONE-5-recursivemas-norm.md.

    Computes in float32 for numerical accuracy.
    """
    inner = _get_inner(model)
    h = input_embeds.astype(mx.float32)
    cache = [None] * len(inner.layers)
    try:
        from mlx_lm.models.qwen2 import create_attention_mask
    except ImportError:
        try:
            from mlx_lm.models.llama import create_attention_mask
        except ImportError:
            create_attention_mask = None

    if create_attention_mask is not None:
        mask = create_attention_mask(h, cache[0])
    else:
        T = h.shape[1]
        mask = mx.full((T, T), -1e9)
        mask = mx.triu(mask, k=1)[None, None, :, :]

    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
        h = h.astype(mx.float32)  # keep float32 between layers

    # Apply final norm to match HF's hidden_states[-1] convention.
    h = inner.norm(h).astype(mx.float32)
    return h


def latent_rollout(model, inner_adapter, input_embeds, n_steps):
    """Autoregressive latent rollout with InnerLink.
    Collects POST-norm hidden states, matching HuggingFace
    `outputs.hidden_states[-1]` (post-final-norm).
    """
    hidden_states = []

    for _ in range(n_steps):
        h = _forward_get_raw_hidden(model, input_embeds)
        last_h = h[:, -1:, :]
        hidden_states.append(last_h)

        next_embed = inner_adapter(last_h)
        input_embeds = mx.concatenate([input_embeds, next_embed], axis=1)

    return mx.concatenate(hidden_states, axis=1)


def tokenize_prompt(tok, role_prompt, question):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": role_prompt.format(q=question)},
    ]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return mx.array(tok.encode(text, add_special_tokens=False))


# ── Pipeline ─────────────────────────────────────────────

PLANNER_SLOT = "<<LATENT_PLANNER_SLOT>>"
REFINED_SLOT = "<<LATENT_REFINED_SLOT>>"

PLANNER_PROMPT = """You are a planner agent in a multi-agent system.
Give a plan for the question below.
Question:
{q}
Your response should be in the format of:
Step 1: ...
...
Step n: ..."""

# Critic/Refiner prompt WITH slot — latent goes where PLANNER_SLOT is
CRITIC_PROMPT = """You are a refiner agent in a multi-agent system.
The question is:
Question:
{q}
The initial plan from the planner:
Initial Plan:
""" + PLANNER_SLOT + """
You should refine the initial plan and respond with pure plan only in the format of:
Step 1: ...
...
Step n: ..."""

# Solver prompt — plan BEFORE question (matches reference)
SOLVER_PROMPT = """You are a solver agent in a multi-agent system.
Here is the refined plan:
Refined Plan:
""" + REFINED_SLOT + """
The question is:
Question:
{q}

Solve the question given information and put the final answer inside \\boxed{{}}, for example \\boxed{{1}}."""


def split_prompt_at_slot(tok, prompt_text: str, slot: str):
    """Split a prompt into (prefix_ids, suffix_ids) around a slot placeholder."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}]
    try:
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except TypeError:
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    pos = full_text.find(slot)
    if pos < 0:
        raise ValueError(f"Slot {slot!r} not found in rendered prompt")

    prefix_text = full_text[:pos]
    suffix_text = full_text[pos + len(slot):]

    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    return mx.array(prefix_ids), mx.array(suffix_ids)


FEEDBACK_SLOT = "<<LATENT_FEEDBACK_SLOT>>"

PLANNER_PROMPT_R2 = """You are a planner agent in a recursive multi-agent system.
This is round 2.
Question:
{q}
Feedback signal from the previous solver round:
""" + FEEDBACK_SLOT + """
Use the feedback as a soft correction signal to improve the plan.
If there is any conflict, prioritize the question constraints.
Output only a concise plan in the format:
Step 1: ...
...
Step n: ..."""


def run_pipeline(agents, outers, question, task="gsm8k", latent_steps=48, max_tokens=2048, temp=0.6, num_rounds=1):
    """Run Sequential pipeline with recursive rounds.
    
    Rounds 1..num_rounds-1: Planner→Critic→Solver (all latent, no text)
    Final round: Planner→Critic→Solver (solver generates text)
    """
    p_model, p_tok, p_inner = agents["planner"]["model"], agents["planner"]["tok"], agents["planner"]["inner"]
    c_model, c_tok, c_inner = agents["critic"]["model"], agents["critic"]["tok"], agents["critic"]["inner"]
    s_model, s_tok = agents["solver"]["model"], agents["solver"]["tok"]
    s_inner = agents["solver"].get("inner")  # solver may not have inner adapter for final text gen
    p_inner_model = _get_inner(p_model)
    c_inner_model = _get_inner(c_model)
    s_inner_model = _get_inner(s_model)

    feedback_to_planner = None  # latent feedback from solver (for rounds > 1)

    for round_idx in range(num_rounds):
        is_final = (round_idx == num_rounds - 1)

        # 1. Planner
        if round_idx == 0:
            p_ids = tokenize_prompt(p_tok, PLANNER_PROMPT, question)
            p_embeds = p_inner_model.embed_tokens(p_ids[None])
        else:
            # Round 2+: inject feedback at FEEDBACK_SLOT
            p_prefix_ids, p_suffix_ids = split_prompt_at_slot(
                p_tok, PLANNER_PROMPT_R2.format(q=question), FEEDBACK_SLOT)
            p_prefix_embeds = p_inner_model.embed_tokens(p_prefix_ids[None])
            p_suffix_embeds = p_inner_model.embed_tokens(p_suffix_ids[None])
            p_embeds = mx.concatenate([p_prefix_embeds, feedback_to_planner, p_suffix_embeds], axis=1)

        p_hidden = latent_rollout(p_model, p_inner, p_embeds, latent_steps)
        p_self = p_inner(p_hidden)
        p_to_c = outers["outer_12"](p_self)
        mx.eval(p_to_c)

        # 2. Critic
        c_prefix_ids, c_suffix_ids = split_prompt_at_slot(
            c_tok, CRITIC_PROMPT.format(q=question), PLANNER_SLOT)
        c_prefix_embeds = c_inner_model.embed_tokens(c_prefix_ids[None])
        c_suffix_embeds = c_inner_model.embed_tokens(c_suffix_ids[None])
        c_combined = mx.concatenate([c_prefix_embeds, p_to_c, c_suffix_embeds], axis=1)
        c_hidden = latent_rollout(c_model, c_inner, c_combined, latent_steps)
        c_self = c_inner(c_hidden)
        c_to_s = outers["outer_23"](c_self)
        mx.eval(c_to_s)

        if not is_final:
            # Non-final round: solver does latent rollout, feeds back to planner
            s_prefix_ids, s_suffix_ids = split_prompt_at_slot(
                s_tok, SOLVER_PROMPT.format(q=question), REFINED_SLOT)
            s_prefix_embeds = s_inner_model.embed_tokens(s_prefix_ids[None])
            s_suffix_embeds = s_inner_model.embed_tokens(s_suffix_ids[None])
            s_combined = mx.concatenate([s_prefix_embeds, c_to_s, s_suffix_embeds], axis=1)

            if s_inner is not None:
                s_hidden = latent_rollout(s_model, s_inner, s_combined, latent_steps)
                s_self = s_inner(s_hidden)
            else:
                # If solver has no inner adapter, use raw hidden states
                kv_tmp = mlx_cache.make_prompt_cache(s_model)
                h = s_inner_model(None, cache=kv_tmp, input_embeddings=s_combined)
                mx.eval([c.state for c in kv_tmp if hasattr(c, 'state')])
                s_self = h[:, -latent_steps:, :] if h.shape[1] >= latent_steps else h

            feedback_to_planner = outers["outer_31"](s_self)
            mx.eval(feedback_to_planner)
        else:
            # Final round: solver generates text
            s_prefix_ids, s_suffix_ids = split_prompt_at_slot(
                s_tok, SOLVER_PROMPT.format(q=question), REFINED_SLOT)
            s_prefix_embeds = s_inner_model.embed_tokens(s_prefix_ids[None])
            s_suffix_embeds = s_inner_model.embed_tokens(s_suffix_ids[None])
            s_combined = mx.concatenate([s_prefix_embeds, c_to_s, s_suffix_embeds], axis=1)

            kv_solver = mlx_cache.make_prompt_cache(s_model)
            h = s_inner_model(None, cache=kv_solver, input_embeddings=s_combined)
            mx.eval([c.state for c in kv_solver if hasattr(c, 'state')])

            if hasattr(s_model, 'args') and getattr(s_model.args, 'tie_word_embeddings', False):
                logits = s_inner_model.embed_tokens.as_linear(h)
            else:
                logits = s_model.lm_head(h) if hasattr(s_model, 'lm_head') else s_inner_model.embed_tokens.as_linear(h)
            logits = logits[:, -1, :]
            first_token = mx.random.categorical(logits * (1.0 / temp)).item()

            tokens = [first_token]
            eos_ids = getattr(s_tok, 'eos_token_ids', None) or {s_tok.eos_token_id}
            if first_token not in eos_ids:
                for tok_val, _ in generate_step(
                    mx.array([first_token]), s_model, max_tokens=max_tokens - 1, prompt_cache=kv_solver,
                    sampler=lambda x: mx.random.categorical(x * (1.0 / temp)),
                ):
                    t = tok_val.item() if hasattr(tok_val, 'item') else int(tok_val)
                    if t in eos_ids:
                        break
                    tokens.append(t)

            return s_tok.decode(tokens)


# ── Eval helpers ─────────────────────────────────────────

def _normalize_latex(text):
    """Normalize LaTeX answer for comparison."""
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("$", "").replace("{", "").replace("}", "")
    t = t.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    t = re.sub(r"\s+", "", t)
    return t

def _normalize_int(text):
    """Extract first integer from text (handles fractions)."""
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    # Replace simple fractions
    def repl(m):
        n, d = int(m.group(1)), int(m.group(2))
        return str(n / d) if d != 0 else m.group(0)
    t = re.sub(r"\\frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}", repl, t)
    t = t.replace("\\left", "").replace("\\right", "").replace(",", "").replace("\\circ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return ""
    return m.group(0).split(".")[0]

def extract_answer(text, task):
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not boxes:
        boxes = re.findall(r"\\boxed\{(.*?)\}", text)
    pred = boxes[-1].strip() if boxes else ""
    if not pred:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        pred = nums[-1] if nums else ""
    return pred

def compare_answer(pred, gold, task):
    """Multi-strategy answer comparison for MATH-500."""
    if not pred or not gold:
        return False
    # Strategy 1: exact match
    if pred.strip() == gold.strip():
        return True
    # Strategy 2: normalized LaTeX
    if _normalize_latex(pred) == _normalize_latex(gold):
        return True
    # Strategy 3: integer extraction
    pi, gi = _normalize_int(pred), _normalize_int(gold)
    if pi and gi and pi == gi:
        return True
    # Strategy 4: no-space
    if re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", gold):
        return True
    return False

def load_data(task, n):
    if task == "gsm8k":
        ds = list(load_dataset("gsm8k", "main", split="test"))[:n]
        return [{"question": d["question"],
                 "gold": re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", d["answer"]).group(1)} for d in ds]
    if task == "math500":
        ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[:n]
        return [{"question": d["problem"], "gold": str(d["answer"]).strip()} for d in ds]
    raise ValueError(f"Unknown task: {task}")


# ── Main ─────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--style", default="light", choices=["light", "scaled"])
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--latent_steps", type=int, default=48)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--rounds", type=int, default=1, help="Number of recursive rounds")
    args = p.parse_args()

    agents, outers = load_system(task="math", style=args.style)
    data = load_data(args.task, args.max_samples)

    print(f"\nRunning RecursiveMAS Sequential-Light on {args.task} ({len(data)} samples)...")
    correct = 0
    total_time = 0

    for i, item in enumerate(data):
        t0 = time.time()
        resp = run_pipeline(agents, outers, item["question"], args.task, args.latent_steps, args.max_tokens, args.temp, num_rounds=args.rounds)
        elapsed = time.time() - t0
        total_time += elapsed

        pred = extract_answer(resp, args.task)
        gold = item["gold"]
        ok = compare_answer(pred, gold, args.task)
        correct += ok
        print(f"  [{i+1}/{len(data)}] {'✓' if ok else '✗'} pred={pred} gold={gold} time={elapsed:.1f}s")

    acc = correct / len(data)
    print(json.dumps({
        "method": "recursive_mas_light",
        "task": args.task,
        "samples": len(data),
        "accuracy": round(acc, 4),
        "correct": correct,
        "avg_time_sec": round(total_time / len(data), 2),
        "total_time_sec": round(total_time, 2),
    }))


if __name__ == "__main__":
    main()
