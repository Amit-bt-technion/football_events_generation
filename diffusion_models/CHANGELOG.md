# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-01-12

### Added - U-Net Support & Benchmarking

#### New Model Architecture
- **U-Net implementation** (`models/unet.py`)
  - 1D convolutional architecture for sequence data
  - Residual blocks with time conditioning
  - Multi-scale processing with encoder-decoder
  - Skip connections between encoder and decoder
  - Self-attention at configurable resolutions
  - ~500 lines of well-documented code

#### Model Selection
- **`--model_type` parameter** to choose between 'dit' and 'unet'
- Separate checkpoints for each model:
  - `dit_best_model.pt`, `dit_checkpoint_epoch_N.pt`
  - `unet_best_model.pt`, `unet_checkpoint_epoch_N.pt`
- Separate cached samples:
  - `dit_cached_samples_epoch_N.pkl`
  - `unet_cached_samples_epoch_N.pkl`
- Automatic model type detection from checkpoints

#### Benchmarking Tools
- **`examples/benchmark_models.py`**
  - Trains both models with identical settings
  - Compares training time and GPU memory
  - Compares generation speed (DDPM and DDIM)
  - Compares quality metrics (FD, diversity, combined score)
  - Generates comprehensive comparison report
  - Outputs `benchmark_report.json`

#### Documentation
- **`DiT_vs_UNet_GUIDE.md`**
  - Detailed architecture comparison
  - When to use each model
  - Performance benchmarks
  - Configuration examples
  - Common questions answered

- **`UPDATE_SUMMARY.md`**
  - Summary of all changes
  - Migration guide (none needed - backward compatible)
  - Quick start for new features

- **`CHEATSHEET.md`**
  - Quick reference for common commands
  - Model presets
  - File structure
  - Troubleshooting tips

#### Configuration
- New U-Net specific parameters:
  - `--channel_multipliers`: e.g., "1,2,4,8"
  - `--num_res_blocks`: Residual blocks per resolution
  - `--attention_resolutions`: e.g., "8,16"

### Changed

#### Training
- Model creation now supports both architectures
- Checkpoint naming includes model type
- Cache naming includes model type

#### Evaluation
- Auto-detects model type from checkpoint
- Loads appropriate architecture
- Saves results with model type prefix

#### Generation
- Supports both model types
- Saves samples with model type prefix

#### Package
- Version bumped to 0.2.0
- `models/__init__.py` exports `UNet1D`
- Main package `__init__.py` exports both models

### Maintained

#### Backward Compatibility
- ✅ All existing DiT functionality preserved
- ✅ Old checkpoints still work (treated as DiT)
- ✅ No breaking changes to API
- ✅ All tests pass

#### Core Features
- ✅ Diffusion process (DDPM, DDIM)
- ✅ Multiple noise schedules
- ✅ Training pipeline with mixed precision
- ✅ Comprehensive evaluation metrics
- ✅ Rich visualizations
- ✅ Data pipeline and caching

## [0.1.0] - 2024-01-11

### Initial Release

#### Core Components
- **Diffusion Transformer (DiT)** model
  - Transformer-based denoising
  - Adaptive layer normalization
  - Positional embeddings
  - Configurable layers, heads, dimensions

- **Diffusion Process**
  - Forward diffusion (noise addition)
  - Reverse diffusion (denoising)
  - Multiple noise schedules (linear, cosine, quadratic)
  - DDPM sampling (1000 steps)
  - DDIM sampling (50 steps, 20x faster)

- **Data Pipeline**
  - EventSequenceDataset
  - Loads pre-computed embeddings
  - Automatic sequence masking
  - Train/val/test splitting

- **Training**
  - Full training loop with validation
  - Mixed precision training (AMP)
  - Gradient clipping
  - Learning rate warmup + cosine decay
  - Checkpoint management
  - Sample caching during training

- **Evaluation**
  - Statistical metrics (Fréchet distance, mean/std matching)
  - Diversity metrics (pairwise distance, unique ratio)
  - Coverage metrics
  - xG-based metrics (framework ready)
  - Combined scoring with tunable weights

- **Generation**
  - Naive generation from random noise
  - Trajectory recording
  - DDPM and DDIM sampling
  - Batch generation

- **Visualization**
  - Diffusion trajectory plots
  - Animated GIF of denoising
  - Sample heatmaps
  - t-SNE and PCA embeddings
  - Football pitch visualization (ready)
  - Metrics bar charts

#### User Interface
- **CLI with argparse**
  - 50+ configurable parameters
  - Four tasks: train, evaluate, naive_generation, visualize
  - Sensible defaults

- **Example Scripts**
  - `examples/train_and_generate.py`: Full pipeline
  - Step-by-step workflow

#### Documentation
- **README.md**: Comprehensive usage guide
- **QUICKSTART.md**: 5-minute getting started
- **PROJECT_SUMMARY.md**: Implementation overview
- **TROUBLESHOOTING.md**: Common issues and solutions

#### Testing
- **Unit tests** (`tests/test_pipeline.py`)
  - Model architecture tests
  - Diffusion process tests
  - Dataset tests
  - Integration tests

#### Packaging
- `setup.py` for installation
- `requirements.txt` for dependencies
- `configs/` for configuration templates
- Proper Python package structure

---

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Upgrade Guide

### From 0.1.0 to 0.2.0

**No action required!** Version 0.2.0 is fully backward compatible.

**Optional**: To adopt new naming convention:
```bash
# Rename old checkpoints
mv models/best_model.pt models/dit_best_model.pt
```

**To use U-Net**:
```bash
# Just add --model_type unet
python -m diffusion_transformer.main --task train --model_type unet
```

**To benchmark**:
```bash
# Run comparison script
python examples/benchmark_models.py
```

## Future Roadmap

### [0.3.0] - Planned
- Conditional generation (team, player, game state)
- Classifier-Free Guidance (CFG)
- Pre-trained model zoo
- Web UI for generation

### [0.4.0] - Planned
- Inpainting (fill missing events)
- Latent space interpolation
- Multi-task training
- Advanced sampling (DPM-Solver)

### [1.0.0] - Planned
- Production-ready API
- Model serving
- Real-time generation
- Comprehensive tutorials

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details
