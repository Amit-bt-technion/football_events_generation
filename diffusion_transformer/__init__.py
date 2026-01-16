"""
Diffusion Transformer for Football Event Sequence Generation.

A modular pipeline for training and evaluating diffusion models
on football event sequences. Supports two architectures:
- DiT (Diffusion Transformer): Transformer-based denoising
- U-Net: CNN-based denoising with 1D convolutions
"""

__version__ = "0.2.0"
__author__ = "Amit Ben-Tzvi"

from diffusion_transformer.models.dit import DiffusionTransformer
from diffusion_transformer.models.unet import UNet1D
from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.data.dataset import EventSequenceDataset, create_dataloaders
from diffusion_transformer.training.trainer import Trainer
from diffusion_transformer.evaluation.evaluator import Evaluator
from diffusion_transformer.evaluation.generator import Generator
from diffusion_transformer.visualization.visualizer import Visualizer

__all__ = [
    "DiffusionTransformer",
    "UNet1D",
    "DiffusionProcess",
    "EventSequenceDataset",
    "create_dataloaders",
    "Trainer",
    "Evaluator",
    "Generator",
    "Visualizer",
]
