"""
Evaluation module with realism and diversity metrics.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from scipy import linalg
from sklearn.metrics import pairwise_distances
import pickle
import json

from torch.utils.data import DataLoader

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for generated sequences."""
    
    def __init__(self, args, test_dataset: DataLoader):
        """
        Initialize evaluator.
        
        Args:
            args: Argument namespace with configuration
            test_dataset: DataLoader for test data
        """
        self.args = args
        self.device = torch.device(args.device)
        self.test_dataset = test_dataset
        
        # Load model based on checkpoint or args
        logger.info("Loading model...")
        
        # Try to load checkpoint to get model type
        model_type = args.model_type
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            model_type = checkpoint.get('model_type', args.model_type)
            logger.info(f"Detected model type from checkpoint: {model_type}")
        
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
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        else:
            # Try to load best model
            best_model_path = os.path.join(args.models_dir, f"{model_type}_best_model.pt")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded best model from {best_model_path}")
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
            logger.info("Loading xG model...")
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
            logger.warning("xG model loading not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error loading xG model: {e}")
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
        logger.info(f"Using {num_samples} real samples for evaluation")
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
        
        logger.warning("xG metrics not fully implemented yet")
        
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
        logger.info(f"Generating {self.args.num_eval_samples} samples for evaluation...")

        # Generate samples
        generated = self.generate_samples(
            self.args.num_eval_samples,
            use_ddim=(self.args.ddim_steps < self.args.num_timesteps)
        )
        
        # Get real samples
        real = self.get_real_samples(self.args.num_eval_samples)
        
        logger.info("Calculating evaluation metrics...")
        results = {}
        
        # Statistical metrics
        if self.args.use_statistical_metrics:
            logger.info("Computing statistical metrics...")
            stat_metrics = self.calculate_statistical_metrics(generated, real)
            results['statistical'] = stat_metrics
            
            logger.info("Statistical Metrics:")
            for key, value in stat_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        # xG metrics
        if self.args.use_xg_metrics:
            logger.info("Computing xG-based metrics...")
            xg_metrics = self.calculate_xg_metrics(generated, real)
            results['xg'] = xg_metrics
            
            if xg_metrics:
                logger.info("xG Metrics:")
                for key, value in xg_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")

        # Diversity metrics
        logger.info("Computing diversity metrics...")
        diversity_metrics = self.calculate_diversity_metrics(generated)
        results['diversity'] = diversity_metrics
        
        logger.info("Diversity Metrics:")
        for key, value in diversity_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Coverage
        logger.info("Computing coverage...")
        coverage = self.calculate_coverage(generated, real)
        results['coverage'] = coverage
        logger.info(f"Coverage: {coverage:.4f}")

        # Combined score
        combined_score = self.calculate_combined_score(results)
        results['combined_score'] = combined_score
        logger.info(f"Combined Score: {combined_score:.4f}")

        # Save results
        output_path = os.path.join(self.args.output_dir, f"{self.model_type}_evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        logger.info(f"Results saved to {output_path}")

        # Save samples
        samples_path = os.path.join(self.args.output_dir, f"{self.model_type}_generated_samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump({
                'generated': generated,
                'real': real
            }, f)
        logger.info(f"Samples saved to {samples_path}")

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


class SequenceEvaluator:
    """Evaluates sequences for illegal transitions and non-monotonic timestamps."""

    def __init__(self, cache_dir=None, transition_tolerance=0.0, time_tolerance=0):
        """
        Initialize sequence evaluator.

        Args:
            cache_dir: Directory containing cache files (defaults to project_root/cache)
            transition_tolerance: Minimum probability for valid transitions (0 = strict)
            time_tolerance: Allowed time tolerance in seconds (0 = strict monotonic)
        """
        # Get project root (football_events_generation directory)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Set cache directory relative to project root
        if cache_dir is None:
            self.cache_dir = project_root / 'cache'
        else:
            # If provided as relative path, make it relative to project root
            cache_path = Path(cache_dir)
            if cache_path.is_absolute():
                self.cache_dir = cache_path
            else:
                self.cache_dir = project_root / cache_dir

        self.cache_dir = str(self.cache_dir)
        self.transition_tolerance = transition_tolerance
        self.time_tolerance = time_tolerance

        # Load or create transition matrix
        self.transition_matrix_dir = os.path.join(self.cache_dir, 'transition_matrix')
        self._load_or_create_transition_matrix()

        # Feature indices from config (common features)
        # Based on tokenizer config: type, play_pattern, location[0], location[1],
        # duration, under_pressure, out, counterpress, period, second, position
        # Plus special parsers: minute, team, possession_team, player
        self.event_type_idx = 0
        self.period_idx = 8
        self.second_idx = 9
        # Minute is a special parser at index 11 (after common categorical features)
        self.minute_idx = 11

    def _load_or_create_transition_matrix(self):
        """Load transition matrix from cache or create if not found."""
        prob_matrix_path = os.path.join(self.transition_matrix_dir, 'event_transition_probabilities.pkl')

        if os.path.exists(prob_matrix_path):
            logger.info(f"Loading transition matrix from {prob_matrix_path}")
            import pandas as pd
            self.prob_matrix = pd.read_pickle(prob_matrix_path)

            with open(os.path.join(self.transition_matrix_dir, 'event_type_mapping.pkl'), 'rb') as f:
                self.event_mapping = pickle.load(f)
        else:
            logger.info("Transition matrix not found, creating...")
            self._create_transition_matrix()

    def _create_transition_matrix(self):
        """Create transition matrix using the create_transition_matrix module."""
        from diffusion_transformer.evaluation.create_transition_matrix import create_transition_matrix

        # Create matrix with custom output directory
        prob_df, _, mapping = create_transition_matrix(
            events_df_path=os.path.join(self.cache_dir, 'events_df.pkl')
        )

        self.prob_matrix = prob_df
        self.event_mapping = mapping
        logger.info("Transition matrix created successfully")

    def _denormalize_event_type(self, normalized_value):
        """Convert normalized event type back to event ID."""
        from tokenizer.config import event_ids

        all_event_ids = sorted(event_ids.values())
        num_categories = len(all_event_ids)
        original_index = int(round(normalized_value * (num_categories - 1)))
        original_index = max(0, min(original_index, num_categories - 1))
        return all_event_ids[original_index]

    def _get_event_name(self, event_id):
        """Get event name from event ID."""
        idx = self.event_mapping['id_to_index'].get(event_id)
        if idx is not None:
            return self.event_mapping['index_to_name'][idx]
        return None

    def _extract_timestamp(self, event_vector):
        """Extract timestamp from event vector (period, minute, second)."""
        # Denormalize features
        period = int(round(event_vector[self.period_idx] * 4)) + 1  # 1-5
        minute = int(round(event_vector[self.minute_idx] * 59))  # 0-59
        second = int(round(event_vector[self.second_idx] * 59))  # 0-59

        # Convert to total seconds
        # Each period is ~45 minutes (regulation) or ~15 minutes (extra time)
        # Simplification: treat each period as 45 min for ordering purposes
        total_seconds = (period - 1) * 45 * 60 + minute * 60 + second
        return total_seconds, period, minute, second

    def evaluate_sequence(self, sequence):
        """
        Evaluate a sequence of event vectors.

        Args:
            sequence: Array of shape (seq_len, feature_dim)

        Returns:
            dict with:
                - passed: bool, whether sequence passes all checks
                - violations: list of violation messages
                - transition_violations: int, number of illegal transitions
                - time_violations: int, number of time violations
        """
        violations = []
        transition_violations = 0
        time_violations = 0

        # Extract event types
        event_types_normalized = sequence[:, self.event_type_idx]
        event_ids = [self._denormalize_event_type(val) for val in event_types_normalized]
        event_names = [self._get_event_name(eid) for eid in event_ids]

        # Check transitions
        for i in range(len(sequence) - 1):
            current_event = event_names[i]
            next_event = event_names[i + 1]

            if current_event is None or next_event is None:
                # Ignored event type
                continue

            # Get transition probability
            prob = self.prob_matrix.loc[current_event, next_event]

            if prob < self.transition_tolerance:
                transition_violations += 1
                violations.append(
                    f"Illegal transition at step {i}: {current_event} → {next_event} "
                    f"(probability: {prob:.4f}, threshold: {self.transition_tolerance})"
                )

        # Check timestamps (monotonic increasing)
        prev_timestamp = -1
        for i in range(len(sequence)):
            total_seconds, period, minute, second = self._extract_timestamp(sequence[i])

            if total_seconds < prev_timestamp - self.time_tolerance:
                time_violations += 1
                violations.append(
                    f"Non-monotonic timestamp at step {i}: "
                    f"P{period} {minute:02d}:{second:02d} ({total_seconds}s) < "
                    f"previous ({prev_timestamp}s), tolerance: {self.time_tolerance}s"
                )

            prev_timestamp = total_seconds

        # Summary
        passed = (transition_violations == 0 and time_violations == 0)

        return {
            'passed': passed,
            'violations': violations,
            'transition_violations': transition_violations,
            'time_violations': time_violations,
            'total_violations': len(violations)
        }

    def evaluate_batch(self, sequences):
        """
        Evaluate a batch of sequences.

        Args:
            sequences: Array of shape (batch_size, seq_len, feature_dim)

        Returns:
            dict with aggregated results
        """
        results = {
            'total_sequences': len(sequences),
            'passed_sequences': 0,
            'failed_sequences': 0,
            'total_transition_violations': 0,
            'total_time_violations': 0,
            'all_violations': []
        }

        for i, sequence in enumerate(sequences):
            result = self.evaluate_sequence(sequence)

            if result['passed']:
                results['passed_sequences'] += 1
            else:
                results['failed_sequences'] += 1

            results['total_transition_violations'] += result['transition_violations']
            results['total_time_violations'] += result['time_violations']

            if not result['passed']:
                results['all_violations'].append({
                    'sequence_idx': i,
                    'violations': result['violations']
                })

        results['pass_rate'] = results['passed_sequences'] / results['total_sequences']

        logger.info(f"Evaluated {results['total_sequences']} sequences:")
        logger.info(f"  Passed: {results['passed_sequences']} ({results['pass_rate']:.2%})")
        logger.info(f"  Failed: {results['failed_sequences']}")
        logger.info(f"  Transition violations: {results['total_transition_violations']}")
        logger.info(f"  Time violations: {results['total_time_violations']}")

        return results

