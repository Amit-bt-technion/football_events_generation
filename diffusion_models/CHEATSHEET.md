# Diffusion Models Quick Reference

## Model Selection

| Model | Best For | Speed | Memory | Quality |
|-------|----------|-------|--------|---------|
| **DiT** | Long-range patterns | Slower | Higher | Better |
| **U-Net** | Efficiency & speed | **Faster** | **Lower** | Good |

## Common Commands

### Training

```bash
# DiT
python -m diffusion_transformer.main --task train --model_type dit

# U-Net  
python -m diffusion_transformer.main --task train --model_type unet
```

### Generation

```bash
# From checkpoint
python -m diffusion_transformer.main --task naive_generation \
    --model_type [dit|unet] --checkpoint models/[model]_best_model.pt
```

### Evaluation

```bash
python -m diffusion_transformer.main --task evaluate \
    --model_type [dit|unet] --use_statistical_metrics
```

### Benchmark

```bash
python examples/benchmark_models.py --num_epochs 50
```

## Model Configurations

### DiT Presets

**Small (Fast)**
```bash
--model_type dit --model_dim 256 --num_layers 4 --num_heads 4
# ~4M params, ~25s/epoch
```

**Medium (Balanced)**
```bash
--model_type dit --model_dim 512 --num_layers 8 --num_heads 8
# ~18M params, ~45s/epoch
```

**Large (Quality)**
```bash
--model_type dit --model_dim 768 --num_layers 12 --num_heads 12
# ~45M params, ~90s/epoch
```

### U-Net Presets

**Small (Very Fast)**
```bash
--model_type unet --model_dim 64 --channel_multipliers "1,2,4" --num_res_blocks 1
# ~2M params, ~15s/epoch
```

**Medium (Balanced)**
```bash
--model_type unet --model_dim 128 --channel_multipliers "1,2,4,8" --num_res_blocks 2
# ~8M params, ~35s/epoch
```

**Large (Quality)**
```bash
--model_type unet --model_dim 192 --channel_multipliers "1,2,4,8" --num_res_blocks 3
# ~15M params, ~50s/epoch
```

## File Outputs

### Models
```
models/
├── dit_best_model.pt              # Best DiT checkpoint
├── dit_checkpoint_epoch_N.pt      # DiT periodic saves
├── unet_best_model.pt             # Best U-Net checkpoint
└── unet_checkpoint_epoch_N.pt     # U-Net periodic saves
```

### Cached Samples
```
cache/diffusion/
├── dit_cached_samples_epoch_N.pkl
└── unet_cached_samples_epoch_N.pkl
```

### Results
```
outputs/
├── dit_evaluation_results.json
├── dit_generated_samples.pkl
├── unet_evaluation_results.json
├── unet_generated_samples.pkl
└── benchmark_report.json
```

## Important Parameters

### Training
```bash
--num_epochs 100          # Training epochs
--batch_size 128          # Batch size
--learning_rate 1e-4      # Learning rate
--num_timesteps 1000      # Diffusion steps
--noise_schedule cosine   # linear|cosine|quadratic
```

### Generation
```bash
--num_gen_samples 100     # Number of samples
--ddim_steps 50          # DDIM steps (faster)
```

### Evaluation
```bash
--num_eval_samples 1000   # Samples for metrics
--diversity_weight 0.5    # Realism vs diversity (0-1)
```

## Noise Schedules

| Schedule | Best For | Notes |
|----------|----------|-------|
| `linear` | Baseline | Simple, works okay |
| `cosine` | **Recommended** | More stable, better quality |
| `quadratic` | Experimentation | Sometimes better for specific data |

## Sampling Methods

| Method | Steps | Speed | Quality |
|--------|-------|-------|---------|
| DDPM | 1000 | Slow | Best |
| DDIM | 50 | **20x faster** | ~Same |

Use DDIM for inference: `--ddim_steps 50`

## Metrics

### Realism (lower is better)
- **Fréchet Distance**: Distribution similarity
- **Mean/Std Difference**: Moment matching

### Diversity (higher is better)
- **Avg Pairwise Distance**: Sample variety
- **Unique Ratio**: Mode collapse indicator

### Combined Score
- `score = (1-w) * realism + w * diversity`
- `w = 0.5` balances both (adjust with `--diversity_weight`)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch_size` or `--model_dim` |
| Slow training | Use U-Net or smaller model |
| Loss is NaN | Lower `--learning_rate`, use cosine schedule |
| Poor quality | Train longer, larger model |
| Mode collapse | Check diversity metrics, retrain |

## GPU Memory Guide

| Config | Memory | Speed |
|--------|--------|-------|
| DiT Small (256, 4) | ~3 GB | Fast |
| DiT Medium (512, 8) | ~6 GB | Medium |
| DiT Large (768, 12) | ~12 GB | Slow |
| U-Net Small (64) | ~2 GB | Very Fast |
| U-Net Medium (128) | ~4 GB | Fast |
| U-Net Large (192) | ~7 GB | Medium |

## Quick Decision Tree

```
Need best quality?
├─ Yes → Use DiT (large config)
└─ No → Need fast training?
    ├─ Yes → Use U-Net (medium config)
    └─ No → Try both and benchmark
```

## One-Liners

```bash
# Quick train (5 epochs, small model)
python -m diffusion_transformer.main --task train --model_type unet --model_dim 64 --num_epochs 5

# Generate 10 samples fast
python -m diffusion_transformer.main --task naive_generation --model_type unet --num_gen_samples 10 --ddim_steps 20

# Full benchmark
python examples/benchmark_models.py --num_epochs 50

# Visualize everything
python -m diffusion_transformer.main --task visualize
```

## Environment Setup

```bash
# Install
pip install -r requirements.txt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test install
python -m diffusion_transformer.models.dit
python -m diffusion_transformer.models.unet
```

## Default Settings (Good Starting Point)

```bash
python -m diffusion_transformer.main \
    --task train \
    --model_type unet \
    --num_epochs 100 \
    --batch_size 128 \
    --model_dim 128 \
    --learning_rate 1e-4 \
    --noise_schedule cosine \
    --num_timesteps 1000
```

## Pro Tips

💡 **Use U-Net first** for prototyping, then DiT for final quality  
💡 **Cosine schedule** works best in most cases  
💡 **DDIM sampling** is 20x faster with similar quality  
💡 **Monitor diversity** metrics to catch mode collapse early  
💡 **Cache samples** during training to visualize progress  
💡 **Benchmark both** models on your specific data  

## Getting Help

📚 **Full docs**: See README.md  
🚀 **Quick start**: See QUICKSTART.md  
🔧 **Problems**: See TROUBLESHOOTING.md  
⚖️ **Comparison**: See DiT_vs_UNet_GUIDE.md  
