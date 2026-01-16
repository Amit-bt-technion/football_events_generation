# Architecture Overview

Visual guide to the diffusion model architectures.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Diffusion Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  Data    │──▶│ Training │──▶│Generation│──▶│  Eval   │ │
│  │ Loading  │   │  Loop    │   │ Sampling │   │ Metrics │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Comparison

### DiT (Diffusion Transformer)

```
Input: (batch, 50, 32)
    ↓
┌─────────────────────────┐
│   Input Projection      │  Linear: 32 → 512
│   + Position Embedding  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Timestep Embedding    │  Sinusoidal → MLP
│   (batch, 512)          │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  DiT Block 1            │ ╮
│  ├─ AdaLN (t)          │ │
│  ├─ Self-Attention     │ │  Repeated
│  └─ MLP                │ │  8 times
├─────────────────────────┤ │
│  DiT Block 2            │ │
│  ...                    │ ╯
├─────────────────────────┤
│  DiT Block 8            │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Layer Norm            │
│   + Output Projection   │  512 → 32
└─────────────────────────┘
    ↓
Output: (batch, 50, 32)
```

**Key Features**:
- Self-attention: All positions attend to all
- Adaptive LayerNorm: Modulated by timestep
- Position-aware: Absolute position embeddings
- Parameters: ~18M (512 dim, 8 layers)

### U-Net

```
Input: (batch, 50, 32)
    ↓ Transpose to (batch, 32, 50)
┌─────────────────────────┐
│   Input Projection      │  Conv1d: 32 → 128
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Timestep Embedding    │  Sinusoidal → MLP
│   (batch, 512)          │
└─────────────────────────┘
    ↓
╔═════════════════════════╗
║      ENCODER            ║
╠═════════════════════════╣
║  Level 1 (128 ch)      ║──┐ Skip 1
║  ├─ ResBlock + Attn    ║  │
║  ├─ ResBlock + Attn    ║  │
║  └─ Downsample ÷2      ║  │
╠═════════════════════════╣  │
║  Level 2 (256 ch)      ║──┼─┐ Skip 2
║  ├─ ResBlock + Attn    ║  │ │
║  ├─ ResBlock + Attn    ║  │ │
║  └─ Downsample ÷2      ║  │ │
╠═════════════════════════╣  │ │
║  Level 3 (512 ch)      ║──┼─┼─┐ Skip 3
║  ├─ ResBlock + Attn    ║  │ │ │
║  ├─ ResBlock + Attn    ║  │ │ │
║  └─ Downsample ÷2      ║  │ │ │
╚═════════════════════════╝  │ │ │
    ↓                        │ │ │
┌─────────────────────────┐  │ │ │
│    BOTTLENECK (1024)    │  │ │ │
│  ├─ ResBlock            │  │ │ │
│  ├─ Attention           │  │ │ │
│  └─ ResBlock            │  │ │ │
└─────────────────────────┘  │ │ │
    ↓                        │ │ │
╔═════════════════════════╗  │ │ │
║      DECODER            ║  │ │ │
╠═════════════════════════╣  │ │ │
║  Level 3 (512 ch)      ║  │ │ │
║  ├─ Upsample ×2         ║  │ │ │
║  ├─ Concat(Skip 3) ◄────┼──┘ │ │
║  ├─ ResBlock + Attn    ║    │ │
║  └─ ResBlock + Attn    ║    │ │
╠═════════════════════════╣    │ │
║  Level 2 (256 ch)      ║    │ │
║  ├─ Upsample ×2         ║    │ │
║  ├─ Concat(Skip 2) ◄────┼────┘ │
║  ├─ ResBlock + Attn    ║      │
║  └─ ResBlock + Attn    ║      │
╠═════════════════════════╣      │
║  Level 1 (128 ch)      ║      │
║  ├─ Upsample ×2         ║      │
║  ├─ Concat(Skip 1) ◄────┼──────┘
║  ├─ ResBlock + Attn    ║
║  └─ ResBlock + Attn    ║
╚═════════════════════════╝
    ↓
┌─────────────────────────┐
│   GroupNorm + SiLU      │
│   + Output Projection   │  128 → 32
└─────────────────────────┘
    ↓ Transpose back
Output: (batch, 50, 32)
```

**Key Features**:
- Multi-scale: 4 resolution levels
- Skip connections: Direct encoder→decoder paths
- Local attention: Convolutional receptive fields
- Parameters: ~8M (128 base, 4 levels)

## Detailed Component Breakdown

### DiT Components

#### AdaLN (Adaptive Layer Norm)
```
Input x: (batch, seq, dim)
Timestep t_emb: (batch, cond_dim)
    ↓
scale, shift = Linear(t_emb).chunk(2)
    ↓
x_norm = LayerNorm(x)
    ↓
output = x_norm * (1 + scale) + shift
```

#### DiT Block
```
┌─────────────────────────┐
│  Input x                │
├─────────────────────────┤
│  h = AdaLN(x, t)       │
│  h = MultiHeadAttn(h)  │
│  x = x + h (residual)  │
├─────────────────────────┤
│  h = AdaLN(x, t)       │
│  h = MLP(h)            │  [Linear → GELU → Dropout → Linear]
│  x = x + h (residual)  │
└─────────────────────────┘
```

### U-Net Components

#### ResBlock
```
┌─────────────────────────┐
│  Input x: (B, C, L)     │
├─────────────────────────┤
│  h = GroupNorm(x)       │
│  h = SiLU(h)            │
│  h = Conv1d(h)          │
├─────────────────────────┤
│  t_emb = MLP(timestep)  │
│  h = h + t_emb[:,:,None]│  Add time
├─────────────────────────┤
│  h = GroupNorm(h)       │
│  h = SiLU(h)            │
│  h = Dropout(h)         │
│  h = Conv1d(h)          │
├─────────────────────────┤
│  if C_in != C_out:      │
│    x = Conv1d_1x1(x)    │
│  output = x + h         │
└─────────────────────────┘
```

#### Attention Block
```
┌─────────────────────────┐
│  Input x: (B, C, L)     │
├─────────────────────────┤
│  h = GroupNorm(x)       │
│  Q, K, V = Conv1d(h)    │  Split into 3
├─────────────────────────┤
│  Reshape for multi-head │
│  Q: (B, H, C/H, L)      │  H = num_heads
│  K: (B, H, C/H, L)      │
│  V: (B, H, C/H, L)      │
├─────────────────────────┤
│  attn = softmax(QK^T/√d)│
│  h = attn @ V           │
├─────────────────────────┤
│  Reshape back           │
│  h = Conv1d_1x1(h)      │
│  output = x + h         │
└─────────────────────────┘
```

## Diffusion Process

```
Training:
    Clean x₀ ────┐
                  │
    Noise ε  ────┼──▶ Forward Diffusion ──▶ Noisy x_t
                  │         q(x_t|x_0)
    Timestep t ───┘
                  
                  ├──▶ Model(x_t, t) ──▶ Predicted ε̂
                  │
                  └──▶ Loss = MSE(ε, ε̂)


Sampling:
    Random noise x_T
           │
           ▼
    ┌────────────────┐
    │ Model(x_t, t)  │ ──▶ Predict ε̂
    └────────────────┘
           │
           ▼
    Remove some noise
           │
           ▼
        x_{t-1}
           │
           ▼ (repeat 1000 times for DDPM)
        x_0 (clean sample)
```

## Memory and Computation

### DiT
```
Forward Pass:
- Input projection: O(L × d × D)
- Attention (per layer): O(L² × D)
- MLP (per layer): O(L × D²)
- Total per layer: O(L² × D + L × D²)

With L=50, D=512:
- Attention: 50² × 512 = 1.3M ops
- MLP: 50 × 512² × 4 = 52M ops
- 8 layers ≈ 420M FLOPs per forward pass

Memory:
- Model weights: ~18M params × 4 bytes = 72 MB
- Activations: ~6 GB (batch 128, mixed precision)
```

### U-Net
```
Forward Pass:
- Convolutions: O(L × C² × k)
- Attention (at some levels): O(L² × C)
- Skip concatenations: O(L × C)

With L=50, C=128, k=3:
- Conv per block: 50 × 128² × 3 = 2.5M ops
- Attention (2 levels): 2 × (50² × 128) = 640K ops
- 4 levels × 2 blocks ≈ 20M FLOPs per forward pass

Memory:
- Model weights: ~8M params × 4 bytes = 32 MB
- Activations: ~4 GB (batch 128, mixed precision)
```

## Training Flow

```
┌─────────────────────────────────────────────────────┐
│                  Training Loop                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Load batch of clean sequences x₀                │
│     ↓                                                │
│  2. Sample random timesteps t ~ Uniform(0, T)       │
│     ↓                                                │
│  3. Sample noise ε ~ N(0, I)                        │
│     ↓                                                │
│  4. Create noisy x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε      │
│     ↓                                                │
│  5. Predict noise: ε̂ = Model(x_t, t)               │
│     ↓                                                │
│  6. Compute loss: L = MSE(ε, ε̂)                     │
│     ↓                                                │
│  7. Backprop and update weights                     │
│     ↓                                                │
│  8. [Every N epochs] Cache intermediate samples     │
│     ↓                                                │
│  9. [Every M epochs] Validate and save checkpoint   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Sampling Flow (DDIM)

```
┌─────────────────────────────────────────────────────┐
│                  DDIM Sampling                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Start with pure noise x_T ~ N(0, I)             │
│     ↓                                                │
│  2. Select subset of timesteps (e.g., 50 of 1000)  │
│     [1000, 980, 960, ..., 40, 20, 0]               │
│     ↓                                                │
│  FOR each timestep t in reverse:                    │
│     │                                                │
│     3. Predict noise: ε̂ = Model(x_t, t)            │
│     ↓                                                │
│     4. Predict x₀: x̂₀ = (x_t - √(1-ᾱ_t)ε̂) / √(ᾱ_t) │
│     ↓                                                │
│     5. Denoise: x_{t-1} = √(ᾱ_{t-1}) x̂₀ +          │
│                           √(1-ᾱ_{t-1}) ε̂           │
│     ↓                                                │
│  END FOR                                             │
│     ↓                                                │
│  6. Return x_0 (clean sample)                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
diffusion_transformer/
│
├── data/
│   └── dataset.py          ◄─── Loads embeddings, creates batches
│
├── models/
│   ├── dit.py              ◄─── Transformer denoising model
│   ├── unet.py             ◄─── CNN denoising model
│   └── diffusion.py        ◄─── Forward/reverse diffusion logic
│
├── training/
│   └── trainer.py          ◄─── Training loop, optimization
│
├── evaluation/
│   ├── evaluator.py        ◄─── Metrics computation
│   └── generator.py        ◄─── Sample generation
│
└── visualization/
    └── visualizer.py       ◄─── Plotting and visualization
```

## Decision Tree: Which Model?

```
                    Start
                      │
                      ▼
            Have > 16GB GPU memory?
                 /          \
               Yes           No
                │             │
                ▼             ▼
    Want best quality?    Use U-Net
         /        \        (efficient)
       Yes        No
        │          │
        ▼          ▼
     Use DiT    Try both,
    (large)    benchmark
```

## Computational Graph Example

### DiT Forward Pass
```
x₀ [B,50,32] ──┬──► Linear [32→512] ──┐
               │                       │
t [B] ─────────┴──► TimestepMLP ──────┼──► x [B,50,512]
                                       │        │
                                       │        ▼
                                       │   ┌─────────┐
                                       │   │ Block 1 │
                                       │   └────┬────┘
                                       │        │
                                       │   ┌────▼────┐
                                       └──▶│ Block 2 │
                                            └────┬────┘
                                                 ⋮
                                            ┌────▼────┐
                                            │ Block 8 │
                                            └────┬────┘
                                                 │
                                            LayerNorm
                                                 │
                                           Linear [512→32]
                                                 │
                                                 ▼
                                           ε̂ [B,50,32]
```

### U-Net Forward Pass
```
x₀ [B,50,32] ──► Conv1d [32→128] ──┐
                                    │
t [B] ──► TimestepMLP ─────────────┼──────────┐
                                    │          │
                                    ▼          │
                              ┌──────────┐    │
                              │ Encoder  │    │
                              │  Level 1 │────┼──skip1
                              │  ↓ ÷2    │    │
                              │  Level 2 │────┼──skip2
                              │  ↓ ÷2    │    │
                              │  Level 3 │────┼──skip3
                              │  ↓ ÷2    │    │
                              └────┬─────┘    │
                                   │          │
                              ┌────▼─────┐    │
                              │Bottleneck│    │
                              └────┬─────┘    │
                                   │          │
                              ┌────▼─────┐    │
                              │ Decoder  │    │
                              │  ↑ ×2    │◄───┘ (uses t)
                              │  +skip3  │
                              │  ↑ ×2    │
                              │  +skip2  │
                              │  ↑ ×2    │
                              │  +skip1  │
                              └────┬─────┘
                                   │
                              Conv1d [128→32]
                                   │
                                   ▼
                              ε̂ [B,50,32]
```

This architecture supports efficient diffusion-based generation of football event sequences with two complementary approaches!
