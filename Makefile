.PHONY: help install install-dev test lint format clean benchmark train-dit train-unet

help:
	@echo "Diffusion Transformer - Makefile Commands"
	@echo "=========================================="
	@echo "install          Install package"
	@echo "install-dev      Install package with dev dependencies"
	@echo "test             Run tests"
	@echo "test-cov         Run tests with coverage"
	@echo "lint             Run linters (flake8, mypy)"
	@echo "format           Format code (black, isort)"
	@echo "clean            Clean cache and build files"
	@echo "benchmark        Run DiT vs U-Net benchmark"
	@echo "train-dit        Train DiT model (quick test)"
	@echo "train-unet       Train U-Net model (quick test)"
	@echo "docs             Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=diffusion_transformer --cov-report=html --cov-report=term

lint:
	flake8 diffusion_transformer/ tests/ examples/
	mypy diffusion_transformer/ --ignore-missing-imports

format:
	black diffusion_transformer/ tests/ examples/
	isort diffusion_transformer/ tests/ examples/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	rm -rf outputs/*.pkl outputs/*.json outputs/*.png outputs/*.gif
	@echo "Cleaned cache and build files"

benchmark:
	python examples/benchmark_models.py --num_epochs 10 --batch_size 64

train-dit:
	python -m diffusion_transformer.main --task train --model_type dit \
		--num_epochs 5 --batch_size 64 --model_dim 256 --num_layers 4

train-unet:
	python -m diffusion_transformer.main --task train --model_type unet \
		--num_epochs 5 --batch_size 64 --model_dim 64 --channel_multipliers "1,2,4"

docs:
	@echo "Building documentation..."
	@echo "TODO: Set up Sphinx documentation"

check-install:
	python -c "import diffusion_transformer; print(f'Version: {diffusion_transformer.__version__}')"
	python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

setup-dirs:
	mkdir -p cache/embeddings cache/diffusion models outputs configs

quick-test: setup-dirs
	@echo "Running quick test..."
	python -m diffusion_transformer.models.dit
	python -m diffusion_transformer.models.unet
	@echo "Quick test passed!"