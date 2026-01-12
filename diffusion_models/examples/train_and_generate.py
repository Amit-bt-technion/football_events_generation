"""
Example script demonstrating the full pipeline:
1. Train a diffusion transformer
2. Generate samples
3. Evaluate and visualize

Usage:
    python examples/train_and_generate.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from diffusion_transformer import (
    DiffusionTransformer,
    DiffusionProcess,
    create_dataloaders,
    Trainer,
    Generator,
    Evaluator,
    Visualizer
)


def create_config():
    """Create configuration for the pipeline."""
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--embeddings_path", type=str, 
                       default="cache/embeddings/embeddings.pkl")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # Training settings
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # Model settings
    parser.add_argument("--sequence_length", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Diffusion settings
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    
    # Generation settings
    parser.add_argument("--num_gen_samples", type=int, default=100)
    parser.add_argument("--ddim_steps", type=int, default=50)
    
    # Evaluation settings
    parser.add_argument("--num_eval_samples", type=int, default=1000)
    parser.add_argument("--use_statistical_metrics", action="store_true", default=True)
    parser.add_argument("--use_xg_metrics", action="store_true", default=False)
    parser.add_argument("--diversity_weight", type=float, default=0.5)
    
    # Other settings
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--num_cached_samples", type=int, default=10)
    parser.add_argument("--cache_interval", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Mode
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and use existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    return parser.parse_args()


def main():
    """Run the full pipeline."""
    args = create_config()
    
    print("="*80)
    print("Diffusion Transformer Pipeline")
    print("="*80)
    print(f"\nDevice: {args.device}")
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Model dim: {args.model_dim}, Layers: {args.num_layers}, Heads: {args.num_heads}")
    print(f"Diffusion steps: {args.num_timesteps}, Schedule: {args.noise_schedule}")
    print()
    
    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "diffusion"), exist_ok=True)
    
    # Step 1: Training (optional)
    if not args.skip_training:
        print("\n" + "="*80)
        print("STEP 1: Training")
        print("="*80)
        
        trainer = Trainer(args)
        trainer.train()
        
        # Update checkpoint path to best model
        args.checkpoint = os.path.join(args.models_dir, "best_model.pt")
    else:
        print("\n" + "="*80)
        print("STEP 1: Training (SKIPPED)")
        print("="*80)
        
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.models_dir, "best_model.pt")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            return
        
        print(f"Using checkpoint: {args.checkpoint}")
    
    # Step 2: Generation
    print("\n" + "="*80)
    print("STEP 2: Generation")
    print("="*80)
    
    generator = Generator(args)
    samples, trajectory = generator.generate_and_visualize()
    
    print(f"\nGenerated {len(samples)} samples")
    print(f"Sample shape: {samples.shape}")
    
    # Step 3: Evaluation
    print("\n" + "="*80)
    print("STEP 3: Evaluation")
    print("="*80)
    
    evaluator = Evaluator(args)
    results = evaluator.evaluate()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    if 'statistical' in results:
        print("\nStatistical Metrics:")
        for key, value in results['statistical'].items():
            print(f"  {key}: {value:.4f}")
    
    if 'diversity' in results:
        print("\nDiversity Metrics:")
        for key, value in results['diversity'].items():
            print(f"  {key}: {value:.4f}")
    
    if 'coverage' in results:
        print(f"\nCoverage: {results['coverage']:.4f}")
    
    if 'combined_score' in results:
        print(f"\nCombined Score: {results['combined_score']:.4f}")
    
    # Step 4: Final Visualizations
    print("\n" + "="*80)
    print("STEP 4: Creating Visualizations")
    print("="*80)
    
    visualizer = Visualizer(args)
    visualizer.visualize_all()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"Models saved to: {args.models_dir}")
    print(f"Cache saved to: {args.cache_dir}")
    print()


if __name__ == "__main__":
    main()
