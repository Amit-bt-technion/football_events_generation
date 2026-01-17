# Architecture Overview

This document provides a technical overview of the diffusion model architectures.

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Overview                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Loading → Training → Evaluation → Visualization       │
│       ↓             ↓           ↓              ↓            │
│  Embeddings    Checkpoints  Metrics        Plots/GIFs       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Architectures

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
- Self-attention across all sequence positions
- Timestep-adaptive normalization (AdaLN)
- Positional embeddings for sequence order
- ~18M parameters (default: 512 dim, 8 layers)

**Best For:** Capturing long-range dependencies and complex patterns

### U-Net (DDPM-style)

**Input:** (batch, seq_len=50, dim=32) → transpose to (batch, 32, 50)

```
┌─────────────────────────┐
│  Input Projection       │  32 → base_channels
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Timestep Embedding     │  Sinusoidal → MLP
└─────────────────────────┘
           ↓
╔═════════════════════════╗
║      ENCODER            ║
╠═════════════════════════╣
║  Level 1 (ch × 1)      ║──┐ Skip connection
║  ├─ ResBlock           ║  │
║  ├─ Attention?         ║  │
║  └─ Downsample         ║  │
╠═════════════════════════╣  │
║  Level 2 (ch × 2)      ║──┼─┐
║  ├─ ResBlock           ║  │ │
║  ├─ Attention?         ║  │ │
║  └─ Downsample         ║  │ │
╠═════════════════════════╣  │ │
║  Level 3 (ch × 4)      ║──┼─┼─┐
║  └─ ...                ║  │ │ │
╚═════════════════════════╝  │ │ │
           ↓                 │ │ │
┌─────────────────────────┐  │ │ │
│     BOTTLENECK          │  │ │ │
│  ├─ ResBlock           │  │ │ │
│  ├─ Attention          │  │ │ │
│  └─ ResBlock           │  │ │ │
└─────────────────────────┘  │ │ │
           ↓                 │ │ │
╔═════════════════════════╗  │ │ │
║      DECODER            ║  │ │ │
╠═════════════════════════╣  │ │ │
║  Level 3 (ch × 4)      ║←─┘ │ │
║  ├─ Upsample           ║    │ │
║  ├─ Concat skip        ║    │ │
║  └─ ResBlock           ║    │ │
╠═════════════════════════╣    │ │
║  Level 2 (ch × 2)      ║←───┘ │
║  └─ ...                ║      │
╠═════════════════════════╣      │
║  Level 1 (ch × 1)      ║←─────┘
║  └─ ...                ║
╚═════════════════════════╝
           ↓
┌─────────────────────────┐
│  Output Projection      │  base_channels → 32
└─────────────────────────┘
```

**Key Features:**
- Multi-scale processing with downsampling/upsampling
- Skip connections preserve information
- Efficient 1D convolutions
- ~8M parameters (default: 128 base channels)

**Best For:** Efficient training and local pattern modeling

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

**DDPM Sampling:** Full T-step reverse process
**DDIM Sampling:** Deterministic, faster (e.g., 50 steps)

### Noise Schedules

1. **Linear:** `β_t = β_start + (β_end - β_start) × t/T`
2. **Cosine:** `ᾱₜ = cos²((t/T + s)/(1 + s) × π/2)` (recommended)
3. **Quadratic:** Squared linear interpolation

## Training Objective

```
L = 𝔼ₜ,x₀,ε [‖ε - εθ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t)‖²]
```

Where:
- `ε ~ 𝒩(0, I)` is random noise
- `εθ` is the denoising model (DiT or U-Net)
- `t ~ Uniform(1, T)` is random timestep

## Code Organization

```
diffusion_transformer/
├── data/
│   ├── dataset.py           # PyTorch Dataset and DataLoader
│   ├── preprocessing.py     # Event embedding and caching
│   └── event_autoencoder_model.py  # Autoencoder (if used)
├── models/
│   ├── diffusion.py         # DiffusionProcess (noise schedules, sampling)
│   ├── dit.py              # DiT architecture
│   └── unet.py             # U-Net architecture
├── training/
│   └── trainer.py          # Training loop, optimization, checkpointing
├── evaluation/
│   ├── evaluator.py        # Metrics computation
│   └── generator.py        # Sample generation
├── visualization/
│   └── visualizer.py       # Plotting and animations
└── utils/
    └── logger.py           # Centralized logging
```

## Evaluation Metrics

### Statistical Metrics
- **Frechet Distance:** Distribution similarity between generated and real samples
- **Mean/Std Difference:** First and second moment matching

### Diversity Metrics
- **Average Pairwise Distance:** Measures sample diversity
- **Unique Ratio:** Fraction of unique samples

### Coverage
- **Coverage Score:** Percentage of real samples with close generated counterparts

### Combined Score
```
Score = (1 - w) × Realism + w × Diversity
```
where `w = diversity_weight`

## Hyperparameter Guidelines

### Training

| Hyperparameter | Recommended | Notes |
|----------------|-------------|-------|
| Batch Size | 128 | Larger = more stable |
| Learning Rate | 1e-4 | With warmup |
| Warmup Epochs | 5 | Stabilizes training |
| Grad Clip | 1.0 | Prevents explosions |
| Num Timesteps | 1000 | More = better quality |

### DiT

| Parameter | Small | Medium | Large |
|-----------|-------|--------|-------|
| model_dim | 256 | 512 | 768 |
| num_layers | 4 | 8 | 12 |
| num_heads | 4 | 8 | 12 |
| Parameters | ~4M | ~18M | ~45M |

### U-Net

| Parameter | Small | Medium | Large |
|-----------|-------|--------|-------|
| model_dim | 64 | 128 | 192 |
| channel_mult | 1,2,4 | 1,2,4,8 | 1,2,4,8 |
| num_res_blocks | 1 | 2 | 3 |
| Parameters | ~2M | ~8M | ~15M |

## Performance Characteristics

### Speed (per epoch on GPU)

| Model | Small | Medium | Large |
|-------|-------|--------|-------|
| DiT | ~25s | ~45s | ~90s |
| U-Net | ~15s | ~35s | ~50s |

### Memory Usage

| Model | Small | Medium | Large |
|-------|-------|--------|-------|
| DiT | ~2GB | ~4GB | ~8GB |
| U-Net | ~1.5GB | ~3GB | ~6GB |

## References

1. **DiT:** Peebles & Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
2. **U-Net:** Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
3. **DDPM:** Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
4. **DDIM:** Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.

