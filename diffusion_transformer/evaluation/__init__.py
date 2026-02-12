"""Evaluation module with metrics and sample generation."""

from diffusion_transformer.evaluation.evaluator import Evaluator

# Lazy import Generator to avoid import errors when only Evaluator is needed
try:
    from diffusion_transformer.evaluation.generator import Generator
    __all__ = ["Evaluator", "Generator"]
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Could not import Generator: {e}")
    Generator = None
    __all__ = ["Evaluator"]
