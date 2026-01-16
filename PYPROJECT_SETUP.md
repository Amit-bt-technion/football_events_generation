# Modern Python Project Setup

This project now uses modern Python packaging standards with `pyproject.toml`.

## What's New

### Files Added

1. **`pyproject.toml`** - Modern Python project configuration (PEP 517/518)
2. **`MANIFEST.in`** - Package data inclusion rules
3. **`.gitignore`** - Git ignore patterns
4. **`Makefile`** - Convenient development commands
5. **`.pre-commit-config.yaml`** - Git pre-commit hooks
6. **`LICENSE`** - MIT License
7. **`INSTALLATION.md`** - Comprehensive installation guide

### Files Updated

- **`setup.py`** - Now minimal (defers to pyproject.toml)
- **`__init__.py`** - Version bumped to 0.2.0

## Quick Start

### Installation

```bash
# Standard installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With everything
pip install -e ".[all]"
```

### Using Make Commands

```bash
make install      # Install package
make test         # Run tests
make format       # Format code (black + isort)
make lint         # Check code quality
make clean        # Clean build files
make benchmark    # Quick benchmark
```

## Project Structure

```
diffusion-transformer/
├── pyproject.toml              # Modern Python configuration ⭐
├── setup.py                    # Minimal setup (backward compat)
├── MANIFEST.in                 # Package data rules ⭐
├── Makefile                    # Convenient commands ⭐
├── .gitignore                  # Git ignore patterns ⭐
├── .pre-commit-config.yaml     # Code quality hooks ⭐
├── LICENSE                     # MIT License ⭐
├── INSTALLATION.md             # Install guide ⭐
│
├── requirements.txt            # Core dependencies
├── README.md                   # Main documentation
├── QUICKSTART.md              # Getting started
├── CHANGELOG.md               # Version history
│
├── diffusion_transformer/     # Main package
│   ├── __init__.py
│   ├── main.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
│
├── examples/                   # Example scripts
├── tests/                     # Test suite
├── configs/                   # Config templates
│
├── cache/                     # Cached data (gitignored)
├── models/                    # Model checkpoints (gitignored)
└── outputs/                   # Results (gitignored)
```

## pyproject.toml Features

### 1. Project Metadata

```toml
[project]
name = "diffusion-transformer"
version = "0.2.0"
description = "Diffusion models for football event generation"
requires-python = ">=3.8"
```

### 2. Dependencies

```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    # ... more
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", ...]
docs = ["sphinx>=4.5.0", ...]
```

Install with: `pip install -e ".[dev]"`

### 3. Entry Points (CLI Commands)

```toml
[project.scripts]
dit-train = "diffusion_transformer.main:main"
dit-generate = "diffusion_transformer.evaluation.generator:main"
dit-evaluate = "diffusion_transformer.evaluation.evaluator:main"
dit-benchmark = "examples.benchmark_models:main"
```

After install, these commands work globally:

```bash
dit-train --help
dit-benchmark --num_epochs 50
```

### 4. Tool Configurations

All dev tool configs in one place:

```toml
[tool.black]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.8"

[tool.coverage.run]
source = ["diffusion_transformer"]
```

## Development Workflow

### Setup

```bash
# Clone
git clone https://github.com/Amit-bt-technion/football_events_generation.git
cd diffusion-transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Daily Development

```bash
# Make changes to code
vim diffusion_transformer/models/dit.py

# Format code
make format

# Run tests
make test

# Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add feature X"
```

### Running Code Quality Tools

```bash
# All at once
make lint        # flake8 + mypy

# Individually
flake8 diffusion_transformer/
mypy diffusion_transformer/
black --check diffusion_transformer/
isort --check diffusion_transformer/
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest tests/test_pipeline.py::TestDiffusionTransformer -v

# Run only fast tests
pytest -m "not slow"
```

## Pre-commit Hooks

Automatically run on `git commit`:

- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **black**: Format Python code
- **isort**: Sort imports
- **flake8**: Lint code
- **mypy**: Type check
- **bandit**: Security checks

Run manually:

```bash
# All files
pre-commit run --all-files

# Specific hook
pre-commit run black --all-files
```

## Makefile Commands

### Installation

```bash
make install         # pip install -e .
make install-dev     # pip install -e ".[dev]"
```

### Testing

```bash
make test            # Run pytest
make test-cov        # Run with coverage report
make quick-test      # Quick sanity check
```

### Code Quality

```bash
make format          # black + isort
make lint            # flake8 + mypy
make clean           # Remove cache files
```

### Training

```bash
make train-dit       # Quick DiT training (5 epochs)
make train-unet      # Quick U-Net training (5 epochs)
make benchmark       # Compare both models
```

### Utilities

```bash
make check-install   # Verify installation
make setup-dirs      # Create cache/models/outputs
make help            # Show all commands
```

## Configuration Files Explained

### pyproject.toml

Central configuration for:
- Package metadata
- Dependencies
- Build system
- All dev tools (black, pytest, mypy, etc.)

### MANIFEST.in

Controls which files are included in distribution:
- Documentation files
- Config templates
- Tests (optional)
- Excludes: cache, models, outputs

### .gitignore

Prevents committing:
- Python cache files (`__pycache__`, `*.pyc`)
- Virtual environments
- Build artifacts
- Large data files (`*.pkl`, `*.pt`)
- IDE files

### .pre-commit-config.yaml

Defines git hooks that run before commit:
- Code formatting
- Linting
- Type checking
- Security checks

## Building Distribution

### Build Wheel

```bash
# Install build tool
pip install build

# Build
python -m build

# Result
ls dist/
# diffusion_transformer-0.2.0-py3-none-any.whl
# diffusion_transformer-0.2.0.tar.gz
```

### Install Wheel

```bash
pip install dist/diffusion_transformer-0.2.0-py3-none-any.whl
```

### Publish to PyPI (Future)

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*
```

## Migration from Old Setup

If you had the old version:

```bash
# Uninstall old
pip uninstall diffusion-transformer

# Pull latest
git pull

# Install new
pip install -e ".[dev]"

# Verify
make check-install
```

## Benefits of Modern Setup

### For Users

✅ **Simple installation**: `pip install -e .`  
✅ **Clear dependencies**: All in pyproject.toml  
✅ **CLI commands**: Installed globally  
✅ **Standard structure**: Follows Python best practices  

### For Developers

✅ **One config file**: Everything in pyproject.toml  
✅ **Pre-commit hooks**: Catch issues before commit  
✅ **Makefile**: Convenient commands  
✅ **Type checking**: Better IDE support  
✅ **Automated testing**: CI/CD ready  

### For Contributors

✅ **Clear setup**: INSTALLATION.md  
✅ **Code standards**: Black + isort + flake8  
✅ **Easy testing**: `make test`  
✅ **Documentation**: Inline and external  

## Compatibility

### Python Versions

Tested on:
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12

### Operating Systems

Works on:
- ✅ Linux (Ubuntu, CentOS, etc.)
- ✅ macOS (Intel & Apple Silicon)
- ✅ Windows (native & WSL)

### Package Managers

Compatible with:
- ✅ pip
- ✅ conda (via pip install)
- ✅ poetry (add to pyproject.toml)
- ✅ pipenv (add to Pipfile)

## Troubleshooting

### "No module named 'build'"

```bash
pip install build
```

### "pre-commit: command not found"

```bash
pip install pre-commit
pre-commit install
```

### "Make: command not found" (Windows)

Use individual commands instead:

```bash
# Instead of: make install
pip install -e .

# Instead of: make test
pytest tests/ -v
```

Or install Make for Windows.

### Import errors after installation

```bash
# Reinstall in editable mode
pip install -e . --force-reinstall --no-deps
```

## Next Steps

1. ✅ Install: `pip install -e ".[dev]"`
2. ✅ Setup hooks: `pre-commit install`
3. ✅ Test: `make test`
4. ✅ Start coding!

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

---

**Questions?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue!