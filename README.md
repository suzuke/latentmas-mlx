# LatentMAS-MLX

Multi-agent latent-space collaboration for Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A port and extension of [LatentMAS](https://github.com/Gen-Verse/LatentMAS) optimized for M-series Macs.

## Features

- **LatentMAS**: Multi-agent latent collaboration via KV cache transfer (training-free)
- **Adaptive OBF Compression**: Orthogonal Backfill KV cache compression with automatic threshold
- **RecursiveMAS**: Heterogeneous multi-agent collaboration with trained adapters (port of [RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS))
- **Communicating Activations**: Activation grafting between models

## Results

### LatentMAS — Qwen3-4B bf16, GSM8K (1319 samples)

| Method | Accuracy | Time/sample | Tokens/sample |
|--------|----------|-------------|---------------|
| Baseline (single agent) | 94%* | 18.5s | 786 |
| TextMAS (4-agent text) | 94% | 35.4s | 613 |
| **LatentMAS (4-agent latent)** | **92.2%** | **15.5s** | **560** |

*Baseline from PyTorch MPS run; MLX LatentMAS achieves comparable accuracy at 1.9× speed.

### Speed Comparison: MLX vs PyTorch MPS

| Framework | Baseline | LatentMAS | Speedup |
|-----------|----------|-----------|---------|
| PyTorch MPS | 48.1s | 29.7s | — |
| MLX | 18.5s | 15.5s | **1.9×** |

## Quick Start

```bash
# Install dependencies
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install mlx mlx-lm datasets tqdm numpy

# Run LatentMAS on GSM8K (5 samples, quick test)
python latentmas/run.py --method latent_mas --model mlx-community/Qwen3-4B-bf16 \
  --task gsm8k --max_samples 5 --max_tokens 2048

# Compare with baseline
python latentmas/run.py --method baseline --model mlx-community/Qwen3-4B-bf16 \
  --task gsm8k --max_samples 5 --max_tokens 2048

# Run with adaptive OBF compression (auto-enabled for long prompts)
python latentmas/run.py --method latent_mas --model mlx-community/Qwen3-4B-bf16 \
  --task gpqa --max_samples 10 --max_tokens 8192
```

## Supported Models

Any model supported by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms):
- Qwen3 (4B, 8B, 14B)
- Gemma 4 (E4B, 26B-A4B, 31B)
- LLaMA 3.x
- And more

## Project Structure

```
latentmas-mlx/
├── latentmas/
│   ├── run.py              # Main entry point
│   ├── core.py             # Latent step generation & KV cache ops
│   ├── compression.py      # OBF KV cache compression
│   ├── prompts.py          # Agent prompt templates
│   └── utils.py            # Answer extraction & evaluation
├── recursive_mas/
│   ├── run.py              # RecursiveMAS entry point
│   ├── adapters.py         # InnerLink & OuterLink (MLX)
│   └── pipeline.py         # Sequential pipeline
├── experiments/
│   ├── comm_activations.py # Communicating Activations experiment
│   └── ple_analysis.py     # Gemma 4 PLE analysis
├── README.md
└── requirements.txt
```

## Methods

### LatentMAS (Training-Free)

Agents communicate by passing KV cache in latent space — no token generation for intermediate agents:

```
Planner (latent) → Critic (latent) → Refiner (latent) → Judger (text output)
```

### Adaptive OBF Compression

Automatically compresses KV cache for long prompts (>200 tokens) using Orthogonal Backfill:
- Preserves attention sink tokens
- Attention-based importance ranking
- Orthogonal residual injection into retained values

### RecursiveMAS (Trained Adapters)

Heterogeneous multi-agent collaboration with lightweight RecursiveLink modules:
- Supports different model families (Qwen + LLaMA + Gemma)
- InnerLink: per-agent latent step adapter
- OuterLink: cross-model latent space bridge

## Citation

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and others},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## License

MIT
