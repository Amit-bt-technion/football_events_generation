# Architecture Overview

This document provides a technical overview of the Diffusion Transformer (DiT) architecture used in this project.

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Overview                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Loading → Training → Evaluation → Visualization            │
│       ↓             ↓           ↓              ↓                 │
│  Embeddings    Checkpoints  Metrics        Plots/GIFs            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Model Architecture

### DiT (Diffusion Transformer)

**Input:** (batch, seq_len=50, dim=32)

```
┌─────────────────────────┐
│  Input Projection       │  32 → model_dim (e.g., 512)
│  + Position Embedding   │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Timestep Embedding     │  Sinusoidal → MLP
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  DiT Block × N          │  N = num_layers (e.g., 8)
│  ├─ AdaLN (t-cond)     │  - Adaptive LayerNorm
│  ├─ Self-Attention     │  - Multi-head attention
│  └─ MLP                │  - Feed-forward network
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Output Projection      │  model_dim → 32
└─────────────────────────┘
```

**Key Features:**
- **Self-attention**: Captures dependencies across all sequence positions simultaneously.
- **Adaptive LayerNorm (AdaLN)**: Efficiently conditions the model on diffusion timesteps.
- **Positional embeddings**: Informs the model about the temporal order of events.
- **Scalability**: Easily adjustable complexity via `model_dim`, `num_layers`, and `num_heads`.

**Best For:** Capturing long-range dependencies and complex patterns in event sequences.

## Diffusion Process

### Forward Process (Noise Addition)

```
x₀ (clean) → x₁ → x₂ → ... → x_T (noise)

q(xₜ | x₀) = 𝒩(xₜ; √ᾱₜ x₀, (1 - ᾱₜ)I)
```

Where:
- `αₜ = 1 - βₜ` (beta schedule)
- `ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ` (cumulative product)

### Reverse Process (Denoising)

```
x_T (noise) → x_{T-1} → ... → x₁ → x₀ (clean)

pθ(x_{t-1} | xₜ) = 𝒩(x_{t-1}; μθ(xₜ, t), Σθ(xₜ, t))
```

**DDPM Sampling:** Full T-step reverse process for maximum quality.
**DDIM Sampling:** Deterministic, accelerated sampling (e.g., 50 steps) for fast inference.

### Noise Schedules

1. **Linear:** Standard interpolation between beta values.
2. **Cosine:** Optimized schedule that preserves more information at low noise levels (recommended).
3. **Quadratic:** Squared linear interpolation for specific noise profiles.

## Training Objective

```
L = 𝔼ₜ,x₀,ε [‖ε - εθ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t)‖²]
```

Where:
- `ε ~ 𝒩(0, I)` is random noise.
- `εθ` is the denoising DiT model.
- `t ~ Uniform(1, T)` is a random timestep.

## Code Organization

```
diffusion_transformer/
├── data/
│   ├── dataset.py           # PyTorch Dataset and DataLoader logic
│   ├── preprocessing.py     # Event embedding and caching utilities
│   └── event_autoencoder_model.py  # Latent space encoding
├── models/
│   ├── diffusion.py         # DiffusionProcess (noise schedules, sampling)
│   └── dit.py              # DiT architecture implementation
├── training/
│   └── trainer.py          # Training loop, optimization, and checkpointing
├── evaluation/
│   ├── evaluator.py        # Realism and diversity metrics
│   └── generator.py        # Latent sample generation
├── visualization/
│   └── visualizer.py       # Pitch plots, heatmaps, and animations
└── utils/
    └── logger.py           # Centralized logging configuration
```

## Evaluation Metrics

### Statistical Metrics
- **Frechet Distance:** Measures the distance between generated and real distributions.
- **Mean/Std Difference:** Checks if basic statistical moments are preserved.

### Diversity Metrics
- **Average Pairwise Distance:** Quantifies the variety in generated sequences.
- **Unique Ratio:** Detects mode collapse by checking for identical samples.

### Coverage
- **Coverage Score:** Estimates how well the generated distribution covers the real event space.

## Hyperparameter Guidelines

| Parameter | Small | Medium | Large |
|-----------|-------|--------|-------|
| model_dim | 256 | 512 | 768 |
| num_layers | 4 | 8 | 12 |
| num_heads | 4 | 8 | 12 |
| Parameters | ~4M | ~18M | ~45M |

## References

1. **DiT:** Peebles & Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
2. **DDPM:** Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
3. **DDIM:** Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.
