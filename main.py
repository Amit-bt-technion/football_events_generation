"""
Main entry point for Diffusion Transformer pipeline.
Supports multiple tasks including naive generation, training, and evaluation.

Run: nohup srun -c 2 --gres=gpu:2 python3 main.py --csv_dir ../match_csv/ --autoencoder_path ../encoder > log.out 2> log.out &
"""

import argparse
import os

import torch
import numpy as np
import random

from diffusion_transformer import create_dataloaders
from diffusion_transformer.data.preprocessing import load_and_embed_matches


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Diffusion Transformer for Football Event Sequence Generation"
    )
    
    # Task and modes paths
    parser.add_argument("--step", type=str, default="train", choices=["train", "evaluate", "naive_generation", "visualize"], help="Step to perform")
    parser.add_argument("--task", type=str, default="full", choices=["full", "shot_only"], help="Task to perform")
    parser.add_argument("--task_params", type=str, default=None)

    # Paths
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory for cached data")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory for model checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for outputs (visualizations, results)")
    parser.add_argument("--autoencoder_path", type=str, default="diffusion_transformer/models/autoencoder.pt", help="Path to pre-trained autoencoder")
    parser.add_argument("--xg_model_path", type=str, default="diffusion_transformer/models/xg_model.pt", help="Path to pre-trained xG model")
    parser.add_argument("--csv_dir", type=str, default="csv")

    # Data settings
    parser.add_argument("--min_gap", type=int, default=1)
    parser.add_argument("--max_gap", type=int, default=None)
    parser.add_argument("--max_samples_per_match", type=int, default=1000)
    parser.add_argument("--max_samples_total", type=int, default=1000000)
    parser.add_argument("--force_recompute", action="store_true", default=False)
    parser.add_argument("--sequence_length", type=int, default=50, help="Length of event sequences")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of event embeddings")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training data split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation data split ratio")
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="dit", choices=["dit", "unet"], help="Model architecture: 'dit' (Diffusion Transformer) or 'unet' (U-Net)")
    
    # Model architecture - DiT
    parser.add_argument("--model_dim", type=int, default=512, help="Model hidden dimension (DiT) or base channels (U-Net)")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers (DiT only)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (DiT only)")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP hidden dimension ratio (DiT only)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Model architecture - U-Net
    parser.add_argument("--channel_multipliers", type=str, default="1,2,4,8", help="Channel multipliers for U-Net (comma-separated, e.g., '1,2,4,8')")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per resolution (U-Net only)")
    parser.add_argument("--attention_resolutions", type=str, default="8,16", help="Resolutions to apply attention (U-Net, comma-separated, e.g., '8,16')")
    
    # Diffusion parameters
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="cosine", choices=["linear", "cosine", "quadratic"], help="Noise schedule type")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Starting beta for linear schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta for linear schedule")
    
    # Training parameters
    parser.add_argument("--skip_training", action="store_true", help="Skip training and use existing checkpoint")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--shuffle", action="store_true", default=True)
    
    # Caching parameters
    parser.add_argument("--num_cached_samples", type=int, default=10, help="Number of samples to cache during training")
    parser.add_argument("--cache_interval", type=int, default=5, help="Cache samples every N epochs")
    
    # Evaluation parameters
    parser.add_argument("--num_eval_samples", type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument("--use_statistical_metrics", action="store_true", help="Use statistical metrics for evaluation")
    parser.add_argument("--use_xg_metrics", action="store_true", help="Use xG model for evaluation")
    parser.add_argument("--diversity_weight", type=float, default=0.5, help="Weight for diversity in combined metrics (0=all realism, 1=all diversity)")
    
    # Generation parameters
    parser.add_argument("--num_gen_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps (for faster generation)")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true", default=False)
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "diffusion"), exist_ok=True)

    print("Loading match events...")
    events_dict, embeddings_dict = load_and_embed_matches(args)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(args, events_dict, embeddings_dict)

    print(f"Running step: {args.step}, task: {args.task}")
    print(f"Device: {args.device}")
    
    if args.step == "train":
        from diffusion_transformer.training.trainer import Trainer
        trainer = Trainer(args, train_loader, val_loader)
        trainer.train()
        
    elif args.step == "evaluate":
        from diffusion_transformer.evaluation.evaluator import Evaluator
        evaluator = Evaluator(args, test_loader)
        evaluator.evaluate()
        
    elif args.step == "naive_generation":
        from diffusion_transformer.evaluation.generator import Generator
        generator = Generator(args, events_dict, embeddings_dict )
        generator.generate_and_visualize()
        
    elif args.step == "visualize":
        from diffusion_transformer.visualization.visualizer import Visualizer
        visualizer = Visualizer(args)
        visualizer.visualize_all()
    
    print(f"{args.step} step completed successfully!")


if __name__ == "__main__":
    main()
