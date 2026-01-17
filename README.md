# Diffusion Transformer for Football Event Sequence Generation

Production-ready implementation of diffusion models (DiT and U-Net) for generating realistic football event sequences using latent diffusion on pre-computed event embeddings.

## Features

- **Two Model Architectures**: Transformer-based (DiT) and convolutional (U-Net) approaches
- **Production-Ready**: Comprehensive logging, error handling, and monitoring
- **Multiple Noise Schedules**: Linear, cosine, and quadratic schedules
- **Fast Sampling**: DDIM sampling for accelerated inference
- **Comprehensive Evaluation**: Statistical, diversity, and coverage metrics
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision
- **Modular Design**: Clean separation of concerns for easy extension

## Quick Start

### Installation with UV (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Sync dependencies
uv sync

# Install in editable mode
uv pip install -e .
```

### Installation with pip

```bash
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation
pip install -e .
```

## Usage

### Training

```bash
# Train DiT model
python main.py --step train --model_type dit --num_epochs 100 --batch_size 128

# Train U-Net model
python main.py --step train --model_type unet --num_epochs 100 --batch_size 128

# With custom configuration
python main.py --step train --model_type dit \
    --model_dim 512 --num_layers 8 --num_heads 8 \
    --learning_rate 1e-4 --batch_size 128 \
    --noise_schedule cosine --num_timesteps 1000
```

### Evaluation

```bash
# Evaluate trained model
python main.py --step evaluate --model_type dit \
    --checkpoint models/dit_best_model.pt \
    --use_statistical_metrics \
    --num_eval_samples 1000

# Use DDIM for faster evaluation
python main.py --step evaluate --model_type dit \
    --checkpoint models/dit_best_model.pt \
    --ddim_steps 50 --use_statistical_metrics
```

### Generation

```bash
# Generate samples
python main.py --step naive_generation --model_type dit \
    --checkpoint models/dit_best_model.pt \
    --num_gen_samples 100
```

### Visualization

```bash
# Visualize training progress and results
python main.py --step visualize
```

## Model Comparison

| Model | Parameters | Speed | Memory | Best For |
|-------|-----------|-------|--------|----------|
| **DiT** | ~18M (default) | Slower | Higher | Long-range dependencies, quality |
| **U-Net** | ~8M (default) | **Faster** | **Lower** | Efficiency, local patterns |

## Architecture Options

### DiT Presets

**Small** (Fast training):
```bash
--model_type dit --model_dim 256 --num_layers 4 --num_heads 4
```

**Medium** (Balanced):
```bash
--model_type dit --model_dim 512 --num_layers 8 --num_heads 8  # Default
```

**Large** (Maximum quality):
```bash
--model_type dit --model_dim 768 --num_layers 12 --num_heads 12
```

### U-Net Presets

**Small**:
```bash
--model_type unet --model_dim 64 --channel_multipliers "1,2,4"
```

**Medium**:
```bash
--model_type unet --model_dim 128 --channel_multipliers "1,2,4,8"  # Default
```

**Large**:
```bash
--model_type unet --model_dim 192 --channel_multipliers "1,2,4,8"
```

## Project Structure

```
football_events_generation/
├── main.py                          # Entry point
├── diffusion_transformer/
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # DiT, U-Net, and diffusion process
│   ├── training/                   # Training loop and optimization
│   ├── evaluation/                 # Evaluation metrics and generation
│   ├── visualization/              # Plotting and visualization
│   └── utils/                      # Logging and utilities
├── cache/                          # Cached embeddings and samples
├── models/                         # Saved checkpoints
├── outputs/                        # Evaluation results and visualizations
└── csv/                           # Input CSV files
```

## Key Arguments

### Common Arguments

- `--step`: Task to perform (`train`, `evaluate`, `naive_generation`, `visualize`)
- `--model_type`: Model architecture (`dit` or `unet`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Enable debug logging

### Training Arguments

- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_epochs`: Warmup epochs (default: 5)
- `--grad_clip`: Gradient clipping value (default: 1.0)

### Diffusion Arguments

- `--num_timesteps`: Number of diffusion steps (default: 1000)
- `--noise_schedule`: Noise schedule type (`linear`, `cosine`, `quadratic`)
- `--ddim_steps`: DDIM sampling steps for fast generation (default: 50)

### Evaluation Arguments

- `--num_eval_samples`: Number of samples for evaluation (default: 1000)
- `--use_statistical_metrics`: Enable statistical metrics
- `--use_xg_metrics`: Enable xG-based metrics

## Logging

All runs create detailed logs:
- Console output with INFO level messages
- File logs saved to `outputs/run.log`
- Use `--verbose` for DEBUG level logging

## Development

### Build Package

```bash
# With uv
uv build

# With pip
python -m build
```

### Install Wheel

```bash
# With uv
uv pip install dist/diffusion_transformer-1.0.0-py3-none-any.whl

# With pip
pip install dist/diffusion_transformer-1.0.0-py3-none-any.whl
```

### Code Quality

```bash
# Format code
black diffusion_transformer/ main.py

# Lint code
ruff check diffusion_transformer/ main.py
```

## Citation

```bibtex
@software{diffusion_transformer_2025,
  author = {Ben-Tzvi, Amit and Zendel, Ilan},
  title = {Diffusion Transformer for Football Event Sequence Generation},
  year = {2025},
  url = {https://github.com/Amit-bt-technion/football_events_generation}
}
```

## License

MIT License - see LICENSE file for details.

## Authors

- Amit Ben-Tzvi (amit-ben@campus.technion.ac.il)
- Ilan Zendel (ilan.zendel@campus.technion.ac.il)

