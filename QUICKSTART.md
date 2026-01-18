# Quick Start Guide

This guide covers installation, usage, and development setup for the Football Event Sequence Generation project.

## Installation

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

The main entry point for all tasks is `main.py`.

### Training

```bash
# Train DiT model
python main.py --step train --model_type dit --num_epochs 100 --batch_size 128

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
python main.py --step generate --model_type dit \
    --checkpoint models/dit_best_model.pt \
    --num_gen_samples 100
```

### Visualization

```bash
# Visualize training progress and results
python main.py --step visualize
```

## Running on Slurm (HPC)

You can use the provided `sbatch_run.sh` script to run tasks on a cluster.

```bash
# Start training
sbatch -c 2 --gres=gpu:1 -J train_dit sbatch_run.sh train

# Start generation
sbatch -c 2 --gres=gpu:1 -J gen_dit sbatch_run.sh generate
```

> **Note**: Update the paths in `sbatch_run.sh` to match your environment before running.

## Key Arguments

### Common Arguments

- `--step`: Task to perform (`train`, `evaluate`, `generate`, `visualize`)
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

## Development

### Code Quality

```bash
# Format code
black diffusion_transformer/ main.py

# Lint code
ruff check diffusion_transformer/ main.py
```

### Setup Verification

Use the provided verification script to check your installation:

```bash
python verify_setup.py
```
