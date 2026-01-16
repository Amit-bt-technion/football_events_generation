"""
Benchmark script to compare DiT vs U-Net performance.

Trains both models with same settings and compares:
- Training time
- Memory usage
- Sample quality (metrics)
- Generation speed

Usage:
    python examples/benchmark_models.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import time
import json
import numpy as np
from pathlib import Path

from diffusion_transformer import (
    DiffusionTransformer,
    UNet1D,
    DiffusionProcess,
    create_dataloaders,
    Trainer,
    Evaluator
)


def create_shared_config():
    """Create configuration shared by both models."""
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--embeddings_path", type=str, 
                       default="cache/embeddings/embeddings.pkl")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--models_dir", type=str, default="models/benchmark")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    
    # Shared settings
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs for each model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sequence_length", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Diffusion settings
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    
    # Benchmark settings
    parser.add_argument("--num_eval_samples", type=int, default=1000)
    parser.add_argument("--num_gen_samples", type=int, default=100)
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only evaluate existing models")
    
    # Other
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
    parser.add_argument("--cache_interval", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--use_statistical_metrics", action="store_true", default=True)
    parser.add_argument("--use_xg_metrics", action="store_true", default=False)
    parser.add_argument("--diversity_weight", type=float, default=0.5)
    
    # Model-specific settings
    parser.add_argument("--model_dim", type=int, default=512,
                       help="Hidden dim for DiT / Base channels for U-Net")
    parser.add_argument("--dit_num_layers", type=int, default=8)
    parser.add_argument("--dit_num_heads", type=int, default=8)
    parser.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--unet_channel_mults", type=str, default="1,2,4,8")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2)
    parser.add_argument("--unet_attn_resolutions", type=str, default="8,16")
    
    return parser.parse_args()


def benchmark_model(args, model_type):
    """Benchmark a single model."""
    print("\n" + "="*80)
    print(f"BENCHMARKING {model_type.upper()}")
    print("="*80)
    
    # Update args for this model
    args.model_type = model_type
    
    # Set model-specific parameters
    if model_type == "dit":
        args.num_layers = args.dit_num_layers
        args.num_heads = args.dit_num_heads
        args.mlp_ratio = args.dit_mlp_ratio
    else:  # unet
        args.channel_multipliers = args.unet_channel_mults
        args.num_res_blocks = args.unet_num_res_blocks
        args.attention_resolutions = args.unet_attn_resolutions
    
    results = {
        'model_type': model_type,
        'config': {}
    }
    
    # Training benchmark
    if not args.skip_training:
        print("\n[1] Training Benchmark")
        print("-" * 80)
        
        trainer = Trainer(args)
        
        # Record training time and memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        else:
            peak_memory = 0
        
        results['training'] = {
            'total_time_seconds': training_time,
            'time_per_epoch': training_time / args.num_epochs,
            'peak_memory_gb': peak_memory
        }
        
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Time per epoch: {training_time/args.num_epochs:.2f} seconds")
        if peak_memory > 0:
            print(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    # Generation benchmark
    print("\n[2] Generation Speed Benchmark")
    print("-" * 80)
    
    from diffusion_transformer.evaluation.generator import Generator
    generator = Generator(args)
    
    # Warmup
    _ = generator.generate_samples(10, use_ddim=True, return_trajectory=False)
    
    # Benchmark DDPM
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    _ = generator.generate_samples(args.num_gen_samples, use_ddim=False, return_trajectory=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ddpm_time = time.time() - start_time
    
    # Benchmark DDIM
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    _ = generator.generate_samples(args.num_gen_samples, use_ddim=True, return_trajectory=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ddim_time = time.time() - start_time
    
    results['generation'] = {
        'ddpm_time_seconds': ddpm_time,
        'ddpm_samples_per_second': args.num_gen_samples / ddpm_time,
        'ddim_time_seconds': ddim_time,
        'ddim_samples_per_second': args.num_gen_samples / ddim_time,
        'speedup_factor': ddpm_time / ddim_time
    }
    
    print(f"DDPM: {ddpm_time:.2f}s ({args.num_gen_samples/ddpm_time:.2f} samples/s)")
    print(f"DDIM: {ddim_time:.2f}s ({args.num_gen_samples/ddim_time:.2f} samples/s)")
    print(f"DDIM speedup: {ddpm_time/ddim_time:.2f}x")
    
    # Evaluation benchmark
    print("\n[3] Quality Metrics")
    print("-" * 80)
    
    evaluator = Evaluator(args)
    eval_results = evaluator.evaluate()
    
    results['quality'] = eval_results
    
    # Model parameters
    from diffusion_transformer.models.dit import count_parameters
    model = trainer.model if not args.skip_training else generator.model
    num_params = count_parameters(model)
    
    results['model'] = {
        'parameters': num_params,
        'parameters_millions': num_params / 1e6
    }
    
    print(f"\nModel parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return results


def create_comparison_report(dit_results, unet_results, output_dir):
    """Create a comparison report."""
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    
    # Training comparison
    if 'training' in dit_results and 'training' in unet_results:
        print("\n[Training Performance]")
        print("-" * 80)
        dit_train = dit_results['training']
        unet_train = unet_results['training']
        
        print(f"{'Metric':<30} {'DiT':<20} {'U-Net':<20} {'Winner':<10}")
        print("-" * 80)
        
        time_dit = dit_train['time_per_epoch']
        time_unet = unet_train['time_per_epoch']
        winner = "DiT" if time_dit < time_unet else "U-Net"
        print(f"{'Time per epoch (s)':<30} {time_dit:<20.2f} {time_unet:<20.2f} {winner:<10}")
        
        if 'peak_memory_gb' in dit_train and dit_train['peak_memory_gb'] > 0:
            mem_dit = dit_train['peak_memory_gb']
            mem_unet = unet_train['peak_memory_gb']
            winner = "DiT" if mem_dit < mem_unet else "U-Net"
            print(f"{'Peak memory (GB)':<30} {mem_dit:<20.2f} {mem_unet:<20.2f} {winner:<10}")
    
    # Generation comparison
    print("\n[Generation Speed]")
    print("-" * 80)
    dit_gen = dit_results['generation']
    unet_gen = unet_results['generation']
    
    print(f"{'Metric':<30} {'DiT':<20} {'U-Net':<20} {'Winner':<10}")
    print("-" * 80)
    
    dit_speed = dit_gen['ddim_samples_per_second']
    unet_speed = unet_gen['ddim_samples_per_second']
    winner = "DiT" if dit_speed > unet_speed else "U-Net"
    print(f"{'DDIM samples/sec':<30} {dit_speed:<20.2f} {unet_speed:<20.2f} {winner:<10}")
    
    # Quality comparison
    print("\n[Quality Metrics]")
    print("-" * 80)
    
    dit_quality = dit_results['quality']
    unet_quality = unet_results['quality']
    
    print(f"{'Metric':<30} {'DiT':<20} {'U-Net':<20} {'Winner':<10}")
    print("-" * 80)
    
    if 'statistical' in dit_quality and 'statistical' in unet_quality:
        dit_fd = dit_quality['statistical'].get('frechet_distance', 0)
        unet_fd = unet_quality['statistical'].get('frechet_distance', 0)
        winner = "DiT" if dit_fd < unet_fd else "U-Net"
        print(f"{'Fréchet Distance':<30} {dit_fd:<20.4f} {unet_fd:<20.4f} {winner:<10}")
    
    if 'diversity' in dit_quality and 'diversity' in unet_quality:
        dit_div = dit_quality['diversity'].get('avg_pairwise_distance', 0)
        unet_div = unet_quality['diversity'].get('avg_pairwise_distance', 0)
        winner = "DiT" if dit_div > unet_div else "U-Net"
        print(f"{'Avg Pairwise Distance':<30} {dit_div:<20.4f} {unet_div:<20.4f} {winner:<10}")
    
    if 'combined_score' in dit_quality and 'combined_score' in unet_quality:
        dit_score = dit_quality['combined_score']
        unet_score = unet_quality['combined_score']
        winner = "DiT" if dit_score > unet_score else "U-Net"
        print(f"{'Combined Score':<30} {dit_score:<20.4f} {unet_score:<20.4f} {winner:<10}")
    
    # Model size comparison
    print("\n[Model Size]")
    print("-" * 80)
    
    dit_params = dit_results['model']['parameters_millions']
    unet_params = unet_results['model']['parameters_millions']
    winner = "U-Net" if unet_params < dit_params else "DiT"
    
    print(f"{'Model':<30} {'Parameters (M)':<20}")
    print("-" * 80)
    print(f"{'DiT':<30} {dit_params:<20.2f}")
    print(f"{'U-Net':<30} {unet_params:<20.2f}")
    print(f"{'Smaller model':<30} {winner:<20}")
    
    # Save full report
    report = {
        'dit': dit_results,
        'unet': unet_results,
        'summary': {
            'training_faster': 'dit' if dit_results.get('training', {}).get('time_per_epoch', float('inf')) < 
                                        unet_results.get('training', {}).get('time_per_epoch', float('inf')) else 'unet',
            'generation_faster': 'dit' if dit_gen['ddim_samples_per_second'] > unet_gen['ddim_samples_per_second'] else 'unet',
            'better_quality': 'dit' if dit_quality.get('combined_score', 0) > unet_quality.get('combined_score', 0) else 'unet',
            'smaller_model': 'unet' if unet_params < dit_params else 'dit'
        }
    }
    
    output_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to {output_path}")
    
    return report


def main():
    """Run benchmark comparison."""
    args = create_shared_config()
    
    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("DIFFUSION MODEL BENCHMARK: DiT vs U-Net")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model dim: {args.model_dim}")
    print(f"  Device: {args.device}")
    print(f"  Dataset: {args.embeddings_path}")
    print()
    
    # Benchmark DiT
    dit_results = benchmark_model(args, 'dit')
    
    # Benchmark U-Net
    unet_results = benchmark_model(args, 'unet')
    
    # Create comparison report
    report = create_comparison_report(dit_results, unet_results, args.output_dir)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - dit_evaluation_results.json")
    print(f"  - unet_evaluation_results.json")
    print(f"  - benchmark_report.json")


if __name__ == "__main__":
    main()
