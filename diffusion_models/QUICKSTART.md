# Quick Start Guide

Get started with the Diffusion Transformer pipeline in 5 minutes!

## Prerequisites

1. **Python Environment**: Python 3.8+
2. **GPU** (recommended): CUDA-capable GPU for faster training
3. **Data**: Pre-computed embeddings from EmbedBall project

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd diffusion_transformer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Prepare Data

Ensure your embeddings are in the correct location:

```
cache/
└── embeddings/
    └── embeddings.pkl
```

The pickle file should contain a dictionary with:
- `'embeddings'`: Event embeddings
- `'sequences'`: List of sequences (each of shape [50, 32])
- `'metadata'`: Optional metadata

## Quick Run

### Option 1: Run Everything (Recommended for First Time)

```bash
python examples/train_and_generate.py
```

This will:
1. Train for 50 epochs (~30 minutes on GPU)
2. Generate 100 samples
3. Evaluate with metrics
4. Create all visualizations

### Option 2: Step-by-Step

#### Step 1: Train a Model (Small for Testing)

```bash
python -m diffusion_transformer.main \
    --task train \
    --num_epochs 10 \
    --batch_size 64 \
    --model_dim 256 \
    --num_layers 4 \
    --save_every 5
```

**Expected output:**
- Training progress bars
- Loss values decreasing
- Checkpoints saved to `models/`
- Cached samples in `cache/diffusion/`

#### Step 2: Generate Samples

```bash
python -m diffusion_transformer.main \
    --task naive_generation \
    --checkpoint models/best_model.pt \
    --num_gen_samples 50 \
    --ddim_steps 50
```

**Expected output:**
- Sampling progress bar
- Generated samples saved to `outputs/generated_samples.pkl`
- Visualizations in `outputs/`

#### Step 3: Evaluate

```bash
python -m diffusion_transformer.main \
    --task evaluate \
    --checkpoint models/best_model.pt \
    --num_eval_samples 500 \
    --use_statistical_metrics
```

**Expected output:**
- Evaluation progress
- Metrics printed to console
- Results saved to `outputs/evaluation_results.json`

#### Step 4: Visualize Everything

```bash
python -m diffusion_transformer.main \
    --task visualize
```

**Expected output:**
- Multiple PNG files in `outputs/`
- GIF animation of diffusion process

## Expected Directory Structure After Running

```
.
├── cache/
│   ├── embeddings/
│   │   └── embeddings.pkl           # Your input data
│   └── diffusion/
│       ├── cached_samples_epoch_5.pkl
│       └── cached_samples_epoch_10.pkl
│
├── models/
│   ├── best_model.pt                # Best model by validation loss
│   ├── checkpoint_epoch_10.pt
│   └── final_model.pt
│
└── outputs/
    ├── generated_samples.pkl        # Generated samples
    ├── evaluation_results.json      # Metrics
    ├── diffusion_trajectory.png     # Noise → Clean visualization
    ├── trajectory_animation.gif     # Animated denoising
    ├── generated_samples_grid.png   # Sample heatmaps
    ├── embedding_space.png          # t-SNE and PCA plots
    └── metrics_visualization.png    # Metrics bar charts
```

## Verify Everything Worked

After running, check:

1. **Training worked**: `models/best_model.pt` exists
2. **Generation worked**: `outputs/generated_samples.pkl` exists
3. **Evaluation worked**: `outputs/evaluation_results.json` exists
4. **Visualizations worked**: Multiple PNG files in `outputs/`

## Common Issues

### Issue: "File not found: embeddings.pkl"

**Solution**: Make sure your embeddings are in `cache/embeddings/embeddings.pkl`

```bash
# Check if file exists
ls -lh cache/embeddings/embeddings.pkl

# If not, create the directory and place your file there
mkdir -p cache/embeddings
cp /path/to/your/embeddings.pkl cache/embeddings/
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or model size

```bash
python -m diffusion_transformer.main \
    --task train \
    --batch_size 32 \
    --model_dim 256 \
    --num_layers 4
```

### Issue: "No module named 'diffusion_transformer'"

**Solution**: Run from the repository root or install as package

```bash
# Option 1: Run from root
cd /path/to/diffusion_transformer
python -m diffusion_transformer.main --task train

# Option 2: Install as package
pip install -e .
```

### Issue: Training is very slow

**Solution**: 
1. Use GPU if available
2. Reduce number of timesteps for faster epochs
3. Use DDIM for evaluation (fewer steps)

```bash
python -m diffusion_transformer.main \
    --task train \
    --num_timesteps 500 \
    --ddim_steps 25 \
    --device cuda
```

## Next Steps

Once you've verified everything works:

1. **Train Longer**: Increase `--num_epochs` to 100+
2. **Bigger Model**: Increase `--model_dim` to 512 or 768
3. **More Samples**: Generate thousands of samples for better evaluation
4. **Tune Hyperparameters**: Experiment with learning rate, noise schedule, etc.

### Hyperparameter Tuning

Try these configurations:

**Fast Training (Testing)**:
```bash
--num_epochs 20 --model_dim 256 --num_layers 4 --num_timesteps 500
```

**Balanced (Recommended)**:
```bash
--num_epochs 100 --model_dim 512 --num_layers 8 --num_timesteps 1000
```

**High Quality (Slow)**:
```bash
--num_epochs 200 --model_dim 768 --num_layers 12 --num_timesteps 2000
```

## Understanding the Output

### Generated Samples

The generated samples are numpy arrays of shape `(N, 50, 32)`:
- N: Number of samples
- 50: Sequence length (events)
- 32: Embedding dimension

### Metrics

**Realism Metrics** (lower is better):
- `frechet_distance`: FID-like metric for distribution matching
- `mean_difference`: How close generated means are to real means
- `std_difference`: How close generated stds are to real stds

**Diversity Metrics** (higher is better):
- `avg_pairwise_distance`: Average distance between samples
- `min_pairwise_distance`: Minimum distance (mode collapse indicator)
- `unique_ratio`: Percentage of unique samples

**Combined Score** (higher is better):
- Weighted combination of realism and diversity
- Controlled by `--diversity_weight` (0.0 = all realism, 1.0 = all diversity)

### Visualizations

- **trajectory_animation.gif**: Shows denoising process over time
- **embedding_space.png**: t-SNE and PCA of generated vs real samples
- **generated_samples_grid.png**: Heatmaps of sample sequences
- **metrics_visualization.png**: Bar charts of all metrics

## Tips for Best Results

1. **Train Long Enough**: Monitor validation loss, train until it plateaus
2. **Use Cosine Schedule**: Generally works better than linear
3. **Tune Diversity Weight**: Start with 0.5, adjust based on your needs
4. **Cache Samples**: Review cached samples during training to monitor quality
5. **Use DDIM for Inference**: 50 steps is usually sufficient (vs 1000 DDPM steps)

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review this guide and the main README
3. Verify your data format matches expected structure
4. Try reducing model/batch size if memory issues
5. Check GitHub issues for similar problems

## Example Session

Here's what a typical first-time session looks like:

```bash
# 1. Quick test (5 minutes)
python examples/train_and_generate.py --num_epochs 5 --skip_training

# 2. Check outputs
ls outputs/

# 3. If looks good, train properly (30 minutes)
python examples/train_and_generate.py --num_epochs 50

# 4. Generate more samples (5 minutes)
python -m diffusion_transformer.main \
    --task naive_generation \
    --num_gen_samples 500

# 5. Full evaluation (10 minutes)
python -m diffusion_transformer.main \
    --task evaluate \
    --num_eval_samples 2000 \
    --use_statistical_metrics
```

## Success Indicators

You'll know everything is working when:

✅ Training loss decreases smoothly  
✅ Validation loss plateaus (not increasing)  
✅ Generated samples look diverse (in embedding space plot)  
✅ Trajectory animation shows clear denoising  
✅ Frechet distance < 50 (for 32D embeddings)  
✅ Unique ratio > 0.8  
✅ No NaN values in metrics  

Happy diffusion! 🚀
