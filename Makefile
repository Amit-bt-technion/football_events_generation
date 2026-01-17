# Makefile for Diffusion Transformer project
# Uses uv for modern Python package management

.PHONY: help install install-dev sync lock clean test format lint check build run-train run-eval all

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Create venv and install package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make sync         - Sync dependencies from lock file"
	@echo "  make lock         - Update lock file"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo "  make test         - Run tests"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make check        - Run format and lint checks"
	@echo "  make build        - Build wheel package"
	@echo "  make run-train    - Run training with DiT model"
	@echo "  make run-eval     - Run evaluation"
	@echo "  make all          - Run checks and build"

# Installation and setup
install:
	uv venv
	uv pip install -e .

install-dev:
	uv venv
	uv pip install -e ".[dev]"

sync:
	uv sync

lock:
	uv lock

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Testing
test:
	pytest tests/ -v --cov=diffusion_transformer --cov-report=term-missing

# Code quality
format:
	black diffusion_transformer/ main.py --line-length 100

lint:
	ruff check diffusion_transformer/ main.py

check: format lint
	@echo "All checks passed!"

# Building
build:
	uv build

# Running
run-train:
	python main.py --step train --model_type dit --num_epochs 100 --batch_size 128 --verbose

run-eval:
	python main.py --step evaluate --model_type dit --use_statistical_metrics --num_eval_samples 1000

# Complete workflow
all: check build
	@echo "Build complete!"

