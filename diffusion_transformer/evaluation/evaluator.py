"""
Evaluation module with realism and diversity metrics.
"""

import os
import torch
import numpy as np
from scipy import linalg
from sklearn.metrics import pairwise_distances
import pickle
import json

from torch.utils.data import DataLoader

from diffusion_transformer.models.diffusion import DiffusionProcess


class Evaluator:
    """Evaluator for generated sequences."""
    
    def __init__(self, args, test_dataset: DataLoader):
        """
        Initialize evaluator.
        
        Args:
            args: Argument namespace with configuration
        """
        self.args = args
        self.device = torch.device(args.device)
        self.test_dataset = test_dataset
        
        # Load model based on checkpoint or args
        print("Loading model...")
        
        # Try to load checkpoint to get model type
        model_type = args.model_type
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            model_type = checkpoint.get('model_type', args.model_type)
            print(f"Detected model type from checkpoint: {model_type}")
        
        # Create model
        if model_type == "dit":
            from diffusion_transformer.models.dit import DiffusionTransformer
            self.model = DiffusionTransformer(
                input_dim=args.embedding_dim,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout,
                max_seq_len=args.sequence_length
            ).to(self.device)
        
        elif model_type == "unet":
            from diffusion_transformer.models.unet import UNet1D
            channel_mults = tuple(int(x) for x in args.channel_multipliers.split(','))
            attn_resolutions = tuple(int(x) for x in args.attention_resolutions.split(','))
            
            self.model = UNet1D(
                input_dim=args.embedding_dim,
                model_channels=args.model_dim,
                channel_multipliers=channel_mults,
                num_res_blocks=args.num_res_blocks,
                attention_resolutions=attn_resolutions,
                dropout=args.dropout,
                max_seq_len=args.sequence_length
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        
        # Load checkpoint
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            # Try to load best model
            best_model_path = os.path.join(args.models_dir, f"{model_type}_best_model.pt")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from {best_model_path}")
            else:
                raise ValueError(f"No checkpoint provided and no {model_type}_best_model.pt found!")
        
        self.model.eval()
        
        # Create diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=args.num_timesteps,
            schedule_type=args.noise_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device
        )
        
        # Load xG model if needed
        self.xg_model = None
        if args.use_xg_metrics and os.path.exists(args.xg_model_path):
            print("Loading xG model...")
            self.xg_model = self.load_xg_model(args.xg_model_path)
    
    def load_xg_model(self, model_path):
        """Load pre-trained xG model."""
        try:
            # This is a placeholder - adjust based on actual xG model structure
            checkpoint = torch.load(model_path, map_location=self.device)
            # TODO: Initialize xG model architecture and load weights
            # xg_model = XGModel(...)
            # xg_model.load_state_dict(checkpoint['model_state_dict'])
            # xg_model.eval()
            # return xg_model
            print("Warning: xG model loading not implemented yet")
            return None
        except Exception as e:
            print(f"Error loading xG model: {e}")
            return None
    
    @torch.no_grad()
    def generate_samples(self, num_samples, use_ddim=False):
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            use_ddim: Whether to use DDIM sampling
            
        Returns:
            Generated samples (num_samples, seq_len, dim)
        """
        shape = (num_samples, self.args.sequence_length, self.args.embedding_dim)
        
        if use_ddim:
            samples = self.diffusion.ddim_sample(
                self.model,
                shape,
                ddim_steps=self.args.ddim_steps,
                progress=True
            )
        else:
            samples = self.diffusion.p_sample_loop(
                self.model,
                shape,
                progress=True
            )
        
        return samples.cpu().numpy()


    def get_real_samples(self, num_samples):
        """Get real samples from test set."""

        num_samples = min(num_samples, len(self.test_dataset))
        print(f"Using {num_samples} real samples for evaluation")
        indices = np.random.choice(len(self.test_dataset), num_samples, replace=False)
        samples = []
        
        for idx in indices:
            sample = self.test_dataset.dataset[int(idx)]
            samples.append(sample.numpy())
        
        return np.stack(samples)
    
    def calculate_statistical_metrics(self, generated, real):
        """
        Calculate statistical metrics for realism.
        
        Args:
            generated: Generated samples (N, seq_len, dim)
            real: Real samples (M, seq_len, dim)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Flatten sequences for distribution comparison
        gen_flat = generated.reshape(-1, generated.shape[-1])
        real_flat = real.reshape(-1, real.shape[-1])
        
        # Mean and std comparison
        gen_mean = np.mean(gen_flat, axis=0)
        real_mean = np.mean(real_flat, axis=0)
        gen_std = np.std(gen_flat, axis=0)
        real_std = np.std(real_flat, axis=0)
        
        metrics['mean_difference'] = np.mean(np.abs(gen_mean - real_mean))
        metrics['std_difference'] = np.mean(np.abs(gen_std - real_std))
        
        # Frechet Distance (simplified version)
        mu_gen = np.mean(gen_flat, axis=0)
        mu_real = np.mean(real_flat, axis=0)
        sigma_gen = np.cov(gen_flat, rowvar=False)
        sigma_real = np.cov(real_flat, rowvar=False)
        
        metrics['frechet_distance'] = self.calculate_frechet_distance(
            mu_gen, sigma_gen, mu_real, sigma_real
        )
        
        return metrics
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Frechet Distance between two Gaussians.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    def calculate_xg_metrics(self, generated, real):
        """
        Calculate xG-based realism metrics.
        
        Args:
            generated: Generated samples
            real: Real samples
            
        Returns:
            Dictionary of metrics
        """
        if self.xg_model is None:
            return {}
        
        metrics = {}
        
        # TODO: Implement xG-based metrics
        # 1. Pass sequences through xG model
        # 2. Compare xG distributions
        # 3. Check if xG values are reasonable
        
        print("Warning: xG metrics not fully implemented yet")
        
        return metrics
    
    def calculate_diversity_metrics(self, generated):
        """
        Calculate diversity metrics.
        
        Args:
            generated: Generated samples (N, seq_len, dim)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Flatten sequences
        gen_flat = generated.reshape(generated.shape[0], -1)
        
        # Pairwise distances
        distances = pairwise_distances(gen_flat, metric='euclidean')
        
        # Average pairwise distance
        n = len(distances)
        metrics['avg_pairwise_distance'] = np.sum(distances) / (n * (n - 1))
        
        # Minimum pairwise distance (mode collapse indicator)
        np.fill_diagonal(distances, np.inf)
        metrics['min_pairwise_distance'] = np.min(distances)
        
        # Standard deviation of pairwise distances
        metrics['std_pairwise_distance'] = np.std(distances[distances != np.inf])
        
        # Entropy-based diversity (approximate)
        # Compute unique samples (with tolerance)
        unique_ratio = len(np.unique(gen_flat, axis=0)) / len(gen_flat)
        metrics['unique_ratio'] = unique_ratio
        
        return metrics
    
    def calculate_coverage(self, generated, real, k=5):
        """
        Calculate coverage: how well generated samples cover real distribution.
        
        Args:
            generated: Generated samples
            real: Real samples
            k: Number of nearest neighbors
            
        Returns:
            Coverage score
        """
        gen_flat = generated.reshape(generated.shape[0], -1)
        real_flat = real.reshape(real.shape[0], -1)
        
        # For each real sample, check if it has a close generated sample
        distances = pairwise_distances(real_flat, gen_flat, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Coverage: percentage of real samples within threshold
        threshold = np.percentile(min_distances, 95)
        coverage = np.mean(min_distances < threshold)
        
        return coverage
    
    def evaluate(self):
        """Run full evaluation."""
        print(f"\nGenerating {self.args.num_eval_samples} samples for evaluation...")
        
        # Generate samples
        generated = self.generate_samples(
            self.args.num_eval_samples,
            use_ddim=(self.args.ddim_steps < self.args.num_timesteps)
        )
        
        # Get real samples
        real = self.get_real_samples(self.args.num_eval_samples)
        
        print("\nCalculating metrics...")
        results = {}
        
        # Statistical metrics
        if self.args.use_statistical_metrics:
            print("Computing statistical metrics...")
            stat_metrics = self.calculate_statistical_metrics(generated, real)
            results['statistical'] = stat_metrics
            
            print("\nStatistical Metrics:")
            for key, value in stat_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # xG metrics
        if self.args.use_xg_metrics:
            print("\nComputing xG-based metrics...")
            xg_metrics = self.calculate_xg_metrics(generated, real)
            results['xg'] = xg_metrics
            
            if xg_metrics:
                print("\nxG Metrics:")
                for key, value in xg_metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        # Diversity metrics
        print("\nComputing diversity metrics...")
        diversity_metrics = self.calculate_diversity_metrics(generated)
        results['diversity'] = diversity_metrics
        
        print("\nDiversity Metrics:")
        for key, value in diversity_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Coverage
        print("\nComputing coverage...")
        coverage = self.calculate_coverage(generated, real)
        results['coverage'] = coverage
        print(f"Coverage: {coverage:.4f}")
        
        # Combined score
        combined_score = self.calculate_combined_score(results)
        results['combined_score'] = combined_score
        print(f"\nCombined Score: {combined_score:.4f}")
        
        # Save results
        output_path = os.path.join(self.args.output_dir, f"{self.model_type}_evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        
        # Save samples
        samples_path = os.path.join(self.args.output_dir, f"{self.model_type}_generated_samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump({
                'generated': generated,
                'real': real
            }, f)
        print(f"Samples saved to {samples_path}")
        
        return results
    
    def calculate_combined_score(self, results):
        """
        Calculate combined score from realism and diversity.
        
        Args:
            results: Dictionary of all metrics
            
        Returns:
            Combined score (higher is better)
        """
        # Normalize metrics (lower is better for most metrics)
        realism_score = 0
        diversity_score = 0
        
        if 'statistical' in results:
            # Invert Frechet distance (lower is better)
            fd = results['statistical'].get('frechet_distance', 0)
            realism_score += 1 / (1 + fd)
        
        if 'diversity' in results:
            # Higher is better
            diversity_score += results['diversity'].get('avg_pairwise_distance', 0) / 10
            diversity_score += results['diversity'].get('unique_ratio', 0)
        
        # Weight realism and diversity
        w = self.args.diversity_weight
        combined = (1 - w) * realism_score + w * diversity_score
        
        return combined
