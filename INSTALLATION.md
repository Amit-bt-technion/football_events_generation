# Installation Guide

Complete installation instructions for the Diffusion Transformer package.

## Quick Installation

### Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation

# Install in editable mode
pip install -e .
```

### Using pip with development dependencies

```bash
# Install with dev tools (pytest, black, flake8, etc.)
pip install -e ".[dev]"
```

### Using pip with all dependencies

```bash
# Install everything (dev + docs)
pip install -e ".[all]"
```

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **GPU**: CUDA-capable GPU recommended (but CPU works too)

### Python Version Check
```bash
python --version  # Should be 3.8+
```

## Installation Methods

### Method 1: Editable Install (Development)

Best for development and experimentation:

```bash
# Clone repository
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Verify installation
python -c "import diffusion_transformer; print(diffusion_transformer.__version__)"
```

**Editable mode** means changes to the source code are immediately reflected without reinstalling.

### Method 2: Standard Install

For production use:

```bash
pip install git+https://github.com/Amit-bt-technion/football_events_generation.git
```

Or from a local copy:

```bash
pip install /path/to/football_events_generation
```

### Method 3: From Source (Build Wheel)

```bash
# Clone and build
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation

# Build wheel
pip install build
python -m build

# Install the wheel
pip install dist/diffusion_transformer-0.2.0-py3-none-any.whl
```

## Dependencies

### Core Dependencies

Automatically installed:
- **torch** (>=2.0.0): Deep learning framework
- **numpy** (>=1.21.0): Numerical computing
- **matplotlib** (>=3.5.0): Plotting
- **seaborn** (>=0.12.0): Statistical visualization
- **scikit-learn** (>=1.0.0): ML utilities
- **scipy** (>=1.7.0): Scientific computing
- **tqdm** (>=4.62.0): Progress bars
- **Pillow** (>=9.0.0): Image processing

### Optional Dependencies

#### Development Tools
```bash
pip install -e ".[dev]"
```

Includes:
- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- flake8 (linting)
- isort (import sorting)
- mypy (type checking)
- pre-commit (git hooks)

#### Documentation Tools
```bash
pip install -e ".[docs]"
```

Includes:
- sphinx (documentation generator)
- sphinx-rtd-theme (ReadTheDocs theme)
- sphinx-autodoc-typehints (type hints in docs)

## GPU Support

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### CPU-Only Installation

PyTorch CPU version is installed by default. No special steps needed.

## Verification

### Quick Test

```bash
# Test imports
python -c "import diffusion_transformer"
python -c "from diffusion_transformer import DiffusionTransformer, UNet1D"

# Test models
python -m diffusion_transformer.models.dit
python -m diffusion_transformer.models.unet

# Check version
python -c "import diffusion_transformer; print(diffusion_transformer.__version__)"
```

### Run Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=diffusion_transformer --cov-report=html
```

### Command-Line Tools

After installation, these commands should be available:

```bash
dit-train --help
dit-generate --help
dit-evaluate --help
dit-benchmark --help
```

## Directory Setup

Create necessary directories:

```bash
# Manual setup
mkdir -p cache/embeddings cache/diffusion models outputs configs

# Or use Makefile
make setup-dirs
```

## Common Installation Issues

### Issue: "No module named 'diffusion_transformer'"

**Solution**:
```bash
# Ensure you're in the right directory
cd /path/to/football_events_generation

# Reinstall
pip install -e .
```

### Issue: "torch not found" or CUDA errors

**Solution**:
```bash
# Uninstall current torch
pip uninstall torch

# Reinstall with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied" during installation

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .

# Or install for user only
pip install --user -e .
```

### Issue: "Building wheel for X failed"

**Solution**:
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Try again
pip install -e .
```

## Using Makefile

For convenience, use the provided Makefile:

```bash
# Install package
make install

# Install with dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean build files
make clean

# Quick benchmark
make benchmark

# Check installation
make check-install
```

## Virtual Environments

### Using venv (Recommended)

```bash
# Create environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install -e .

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create environment
conda create -n diffusion python=3.10
conda activate diffusion

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install package
pip install -e .
```

## Development Setup

For contributors:

```bash
# Clone repository
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd football_events_generation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run quick test
make quick-test

# Format code before committing
make format
```

## Docker Installation (Optional)

Create a `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "diffusion_transformer.main", "--help"]
```

Build and run:

```bash
docker build -t diffusion-transformer .
docker run --gpus all -it diffusion-transformer
```

## Updating

### Update from Git

```bash
cd football_events_generation
git pull
pip install -e . --upgrade
```

### Update Dependencies

```bash
pip install -e . --upgrade --upgrade-strategy eager
```

## Uninstallation

```bash
pip uninstall diffusion-transformer
```

## Next Steps

After installation:

1. **Verify**: Run `make check-install`
2. **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
3. **Test**: Run `make quick-test`
4. **Train**: Try `make train-dit` or `make train-unet`
5. **Benchmark**: Run `make benchmark`

## Getting Help

- **Installation issues**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **General usage**: See [README.md](README.md)
- **API reference**: Run `python -c "help(diffusion_transformer)"`

## System-Specific Notes

### Linux
- Recommended for production
- Best GPU support
- Use system package manager for system deps

### macOS
- Works well, but no CUDA support (MPS available on Apple Silicon)
- Use Homebrew for system dependencies
- May need to install Xcode Command Line Tools

### Windows
- Works with WSL2 recommended
- Use PowerShell or Command Prompt
- Activate venv: `venv\Scripts\activate`

## Minimal Installation

For the smallest footprint:

```bash
# Core dependencies only
pip install torch numpy matplotlib scipy tqdm Pillow

# Install package (no extras)
pip install -e . --no-deps
```

---

**Successfully installed?** Head to [QUICKSTART.md](QUICKSTART.md) to start training! 🚀