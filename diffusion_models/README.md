# Diffusion Transformer for Football Event Sequence Generation

A comprehensive pipeline for training and evaluating diffusion models on football event sequences. Supports two architectures:
- **DiT (Diffusion Transformer)**: Transformer-based denoising with adaptive layer normalization
- **U-Net**: CNN-based denoising with 1D convolutions (DDPM-style)

This implementation uses latent diffusion on pre-computed event embeddings to generate realistic sequences of football events.

## Features

- **Two Model Architectures**: Compare transformer-based (DiT) vs convolutional (U-Net) approaches
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Multiple Noise Schedules**: Support for linear, cosine, and quadratic noise schedules
- **Fast Sampling**: DDIM sampling for faster generation at inference time
- **Comprehensive Evaluation**: Statistical metrics, diversity metrics, and xG-based realism measures
- **Benchmarking Tools**: Built-in scripts to compare model performance
- **Rich Visualizations**: t-SNE/PCA embeddings, diffusion trajectories, pitch visualizations, and more
- **Efficient Caching**: Caches intermediate diffusion states during training (separate for each model)
- **Mixed Precision Training**: Automatic mixed precision for faster training on GPU
- **Configurable**: Extensive CLI arguments for all hyperparameters

## Model Architectures

### Diffusion Transformer (DiT)
- Transformer-based denoising model
- Adaptive layer normalization conditioned on timestep
- Self-attention across sequence positions
- Best for: Capturing long-range dependencies

### U-Net (DDPM-style)
- 1D convolutional architecture
- Encoder-decoder with skip connections
- Residual blocks with time conditioning
- Multi-scale processing with downsampling/upsampling
- Best for: Efficient training and local pattern modeling

Both models can be trained and evaluated with the same pipeline!

## Installation

```bash
# Clone the repositories
git clone https://github.com/Zendellll/EmbedBall
git clone https://github.com/Amit-bt-technion/xG_prediction_model

# Install dependencies
pip install torch numpy matplotlib seaborn scikit-learn scipy tqdm
```

## Project Structure

```
diffusion_transformer/
├── main.py                          # Main entry point
├── data/
│   └── dataset.py                   # Dataset and dataloader utilities
├── models/
│   ├── dit.py                       # Diffusion Transformer architecture
│   └── diffusion.py                 # Diffusion process utilities
├── training/
│   └── trainer.py                   # Training loop and optimization
├── evaluation/
│   ├── evaluator.py                 # Evaluation metrics
│   └── generator.py                 # Sample generation
└── visualization/
    └── visualizer.py                # Visualization utilities

cache/
├── embeddings/                      # Pre-computed embeddings
└── diffusion/                       # Cached diffusion samples

models/                              # Saved model checkpoints

outputs/                             # Generated samples and visualizations
```

## Usage

### 1. Training

Train a diffusion model (choose DiT or U-Net):

**Diffusion Transformer (DiT)**:
```bash
python -m diffusion_transformer.main \
    --task train \
    --model_type dit \
    --embeddings_path cache/embeddings/embeddings.pkl \
    --num_epochs 100 \
    --batch_size 128 \
    --model_dim 512 \
    --num_layers 8 \
    --num_heads 8
```

**U-Net**:
```bash
python -m diffusion_transformer.main \
    --task train \
    --model_type unet \
    --embeddings_path cache/embeddings/embeddings.pkl \
    --num_epochs 100 \
    --batch_size 128 \
    --model_dim 128 \
    --channel_multipliers "1,2,4,8" \
    --num_res_blocks 2
```

Key training arguments:
- `--model_type`: Choose 'dit' or 'unet'
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_timesteps`: Number of diffusion timesteps (default: 1000)
- `--noise_schedule`: Noise schedule type (linear/cosine/quadratic)

### 2. Naive Generation

Generate sequences from random noise:

```bash
python -m diffusion_transformer.main \
    --task naive_generation \
    --model_type dit \  # or 'unet'
    --checkpoint models/dit_best_model.pt \
    --num_gen_samples 100 \
    --ddim_steps 50
```

This will:
- Generate samples from pure noise
- Create visualizations of the diffusion process
- Save generated samples and trajectories
- Plot embedding space visualizations

### 3. Evaluation

Evaluate model with comprehensive metrics:

```bash
python -m diffusion_transformer.main \
    --task evaluate \
    --model_type unet \  # or 'dit'
    --checkpoint models/unet_best_model.pt \
    --num_eval_samples 1000 \
    --use_statistical_metrics \
    --use_xg_metrics \
    --diversity_weight 0.5
```

Evaluation metrics include:
- **Statistical Metrics**: Mean/std differences, Fréchet Distance
- **Diversity Metrics**: Pairwise distances, unique ratio
- **Coverage**: How well generated samples cover real distribution
- **xG Metrics** (optional): Realism based on xG predictions
- **Combined Score**: Weighted combination of realism and diversity

### 4. Visualization

Create all visualizations from saved data:

```bash
python -m diffusion_transformer.main \
    --task visualize \
    --output_dir outputs
```

Generated visualizations:
- Diffusion trajectory (noise → clean)
- Sample grid (heatmaps of generated sequences)
- Embedding space (t-SNE and PCA)
- Metrics bar charts
- Pitch visualizations (if decoded)

### 5. Benchmark Both Models

Compare DiT vs U-Net performance:

```bash
python examples/benchmark_models.py \
    --num_epochs 50 \
    --num_eval_samples 1000
```

This will:
- Train both models with identical settings
- Compare training time and memory usage
- Compare generation speed (DDPM vs DDIM)
- Compare sample quality metrics
- Generate a comprehensive comparison report

Expected output:
```
COMPARISON REPORT
================================================================================
[Training Performance]
Metric                          DiT                  U-Net                Winner
--------------------------------------------------------------------------------
Time per epoch (s)              45.23                38.67                U-Net
Peak memory (GB)                6.52                 4.81                 U-Net

[Generation Speed]
Metric                          DiT                  U-Net                Winner
--------------------------------------------------------------------------------
DDIM samples/sec                12.45                15.78                U-Net

[Quality Metrics]
Metric                          DiT                  U-Net                Winner
--------------------------------------------------------------------------------
Fréchet Distance                32.45                35.12                DiT
Avg Pairwise Distance           67.89                64.23                DiT
Combined Score                  0.7234               0.6891               DiT
```

## Configuration

### Model Selection

Choose between two architectures:

**DiT (Diffusion Transformer)**:
```bash
--model_type dit
--model_dim 512           # Hidden dimension
--num_layers 8            # Number of transformer layers
--num_heads 8             # Number of attention heads
--mlp_ratio 4.0          # MLP expansion ratio
--dropout 0.1            # Dropout rate
```

**U-Net**:
```bash
--model_type unet
--model_dim 128           # Base channel count
--channel_multipliers "1,2,4,8"  # Channel multipliers per level
--num_res_blocks 2        # Residual blocks per resolution
--attention_resolutions "8,16"   # Where to apply attention
--dropout 0.1            # Dropout rate
```

### Diffusion Parameters

```bash
--num_timesteps 1000     # Number of diffusion steps
--noise_schedule cosine  # linear/cosine/quadratic
--beta_start 0.0001     # Starting beta (linear schedule)
--beta_end 0.02         # Ending beta (linear schedule)
```

### Data Parameters

```bash
--sequence_length 50     # Length of event sequences
--embedding_dim 32       # Dimension of embeddings
--train_split 0.8       # Training data ratio
--val_split 0.1         # Validation data ratio
```

## How It Works

### 1. Data Processing

The pipeline uses pre-computed event embeddings from the EmbedBall project and sampling functions from the xG prediction model:

- Loads pickled embeddings (50 events × 32 dimensions)
- Masks the last event (shot) during training
- Creates train/val/test splits

### 2. Training

The diffusion transformer learns to denoise corrupted sequences:

1. Sample a clean sequence x₀
2. Sample timestep t ~ Uniform(0, T)
3. Add noise: x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε
4. Predict noise: ε̂ = DiT(x_t, t)
5. Minimize MSE: L = ||ε - ε̂||²

The model uses:
- Adaptive Layer Normalization conditioned on timestep
- Sinusoidal timestep embeddings
- Residual transformer blocks
- Mixed precision training (optional)

### 3. Generation

Generate samples using reverse diffusion:

**Standard Sampling (DDPM)**:
- Start from pure noise x_T ~ N(0, I)
- Iteratively denoise: x_{t-1} = μ(x_t, t) + σ_t z
- T steps required

**Fast Sampling (DDIM)**:
- Deterministic sampling with fewer steps
- 50 steps instead of 1000 (20× speedup)
- Maintains sample quality

### 4. Evaluation

**Realism Metrics**:
- Fréchet Distance between generated and real distributions
- Mean/std matching
- xG prediction distribution comparison (optional)

**Diversity Metrics**:
- Average pairwise distance
- Minimum distance (mode collapse indicator)
- Unique sample ratio

**Combined Score**:
```
score = (1 - w) × realism + w × diversity
```
where w is the diversity weight (0-1).

## Cache Structure

The pipeline uses caching extensively:

```
cache/
├── embeddings/
│   └── embeddings.pkl              # Pre-computed event embeddings
└── diffusion/
    ├── cached_samples_epoch_5.pkl  # Intermediate samples
    ├── cached_samples_epoch_10.pkl
    └── ...
```

Cached diffusion samples include:
- Samples at t = [0, 250, 500, 750, 999]
- Final denoised samples
- Used for visualization and debugging

## Advanced Usage

### Resume Training

```bash
python -m diffusion_transformer.main \
    --task train \
    --checkpoint models/checkpoint_epoch_50.pt
```

### Custom Noise Schedule

```bash
python -m diffusion_transformer.main \
    --task train \
    --noise_schedule quadratic \
    --beta_start 0.0001 \
    --beta_end 0.05
```

### High-Resolution Sampling

```bash
python -m diffusion_transformer.main \
    --task naive_generation \
    --num_timesteps 2000 \
    --ddim_steps 100
```

## Future Extensions

The architecture is designed for easy extension:

1. **Conditional Generation**: Add conditioning on:
   - Team/player context
   - Game state (score, time)
   - Tactical instructions

2. **Classifier-Free Guidance**: Improve sample quality with CFG

3. **Inpainting**: Generate missing events in a sequence

4. **Super-Resolution**: Generate high-frequency details

5. **Multi-Modal**: Condition on text descriptions

## Troubleshooting

**CUDA Out of Memory**:
```bash
--batch_size 64          # Reduce batch size
--model_dim 256          # Smaller model
--num_layers 4           # Fewer layers
```

**Slow Training**:
```bash
--num_workers 8          # More data loading workers
--ddim_steps 20          # Fewer eval steps
```

**Poor Sample Quality**:
```bash
--num_epochs 200         # Train longer
--num_timesteps 2000     # More diffusion steps
--noise_schedule cosine  # Try different schedule
```

## Citation

If you use this code, please cite:

```bibtex
@software{diffusion_transformer_football,
  title={Diffusion Transformer for Football Event Sequence Generation},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- Based on the EmbedBall and xG prediction model projects
- Inspired by the Diffusion Transformer (DiT) paper
- Uses techniques from DDPM and DDIM papers
