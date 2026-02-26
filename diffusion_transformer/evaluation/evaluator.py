"""
Evaluation module with realism and diversity metrics.
Includes comprehensive vector validation for decoded event vectors.
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
from collections import defaultdict

from torch.utils.data import DataLoader

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.data.event_autoencoder_model import EventAutoencoder
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Event Vector Structure Definition (128 dimensions)
# Based on tokenizer configuration
# ============================================================================

# Feature indices and their types
FEATURE_DEFINITIONS = {
    # Common features (0-14)
    0: {'name': 'type.id', 'type': 'categorical', 'num_categories': 29},
    1: {'name': 'play_pattern.id', 'type': 'categorical', 'num_categories': 9},
    2: {'name': 'location_x', 'type': 'range', 'min': 0, 'max': 120},
    3: {'name': 'location_y', 'type': 'range', 'min': 0, 'max': 80},
    4: {'name': 'duration', 'type': 'range', 'min': 0, 'max': 3},
    5: {'name': 'under_pressure', 'type': 'binary'},
    6: {'name': 'out', 'type': 'binary'},
    7: {'name': 'counterpress', 'type': 'binary'},
    8: {'name': 'unified_time', 'type': 'range', 'min': 0, 'max': 1},
    9: {'name': 'second_zeroed', 'type': 'zero'},
    10: {'name': 'position.id', 'type': 'categorical', 'num_categories': 25},
    11: {'name': 'minute_zeroed', 'type': 'zero'},
    12: {'name': 'team.id', 'type': 'binary'},
    13: {'name': 'possession_team.id', 'type': 'binary'},
    14: {'name': 'player_position', 'type': 'range', 'min': 0, 'max': 1},
    
    # Event-specific features (15-70) - sparse for most events
    # Ball recovery (15-16)
    15: {'name': 'ball_recovery.offensive', 'type': 'binary', 'sparse': True},
    16: {'name': 'ball_recovery.recovery_failure', 'type': 'binary', 'sparse': True},
    
    # Duel (17-18)
    17: {'name': 'duel.type.id', 'type': 'categorical', 'num_categories': 2, 'sparse': True},
    18: {'name': 'duel.outcome.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    
    # Block (19-21)
    19: {'name': 'block.deflection', 'type': 'binary', 'sparse': True},
    20: {'name': 'block.offensive', 'type': 'binary', 'sparse': True},
    21: {'name': 'block.save_block', 'type': 'binary', 'sparse': True},
    
    # Clearance (22-23)
    22: {'name': 'clearance.aerial_won', 'type': 'binary', 'sparse': True},
    23: {'name': 'clearance.body_part.id', 'type': 'categorical', 'num_categories': 4, 'sparse': True},
    
    # Interception (24)
    24: {'name': 'interception.outcome.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    
    # Dribble (25-28)
    25: {'name': 'dribble.overrun', 'type': 'binary', 'sparse': True},
    26: {'name': 'dribble.nutmeg', 'type': 'binary', 'sparse': True},
    27: {'name': 'dribble.outcome.id', 'type': 'categorical', 'num_categories': 2, 'sparse': True},
    28: {'name': 'dribble.no_touch', 'type': 'binary', 'sparse': True},
    
    # Substitution (29)
    29: {'name': 'substitution.outcome.id', 'type': 'categorical', 'num_categories': 2, 'sparse': True},
    
    # Foul won (30-32)
    30: {'name': 'foul_won.defensive', 'type': 'binary', 'sparse': True},
    31: {'name': 'foul_won.advantage', 'type': 'binary', 'sparse': True},
    32: {'name': 'foul_won.penalty', 'type': 'binary', 'sparse': True},
    
    # Foul committed (33-37)
    33: {'name': 'foul_committed.type.id', 'type': 'categorical', 'num_categories': 6, 'sparse': True},
    34: {'name': 'foul_committed.offensive', 'type': 'binary', 'sparse': True},
    35: {'name': 'foul_committed.advantage', 'type': 'binary', 'sparse': True},
    36: {'name': 'foul_committed.penalty', 'type': 'binary', 'sparse': True},
    37: {'name': 'foul_committed.card.id', 'type': 'categorical', 'num_categories': 3, 'sparse': True},
    
    # Goalkeeper (38-44)
    38: {'name': 'goalkeeper.type.id', 'type': 'categorical', 'num_categories': 14, 'sparse': True},
    39: {'name': 'goalkeeper.outcome.id', 'type': 'categorical', 'num_categories': 19, 'sparse': True},
    40: {'name': 'goalkeeper.position.id', 'type': 'categorical', 'num_categories': 3, 'sparse': True},
    41: {'name': 'goalkeeper.technique.id', 'type': 'categorical', 'num_categories': 2, 'sparse': True},
    42: {'name': 'goalkeeper.body_part.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    43: {'name': 'goalkeeper.end_location_x', 'type': 'range', 'min': 0, 'max': 120, 'sparse': True},
    44: {'name': 'goalkeeper.end_location_y', 'type': 'range', 'min': 0, 'max': 80, 'sparse': True},
    
    # Bad behavior (45)
    45: {'name': 'bad_behavior.card.id', 'type': 'categorical', 'num_categories': 3, 'sparse': True},
    
    # Player off (46)
    46: {'name': 'player_off.permanent', 'type': 'binary', 'sparse': True},
    
    # Pass (47-64)
    47: {'name': 'pass.type.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    48: {'name': 'pass.length', 'type': 'range', 'min': 0, 'max': 120, 'sparse': True},
    49: {'name': 'pass.angle', 'type': 'range', 'min': -3.15, 'max': 3.15, 'sparse': True},
    50: {'name': 'pass.height.id', 'type': 'categorical', 'num_categories': 3, 'sparse': True},
    51: {'name': 'pass.end_location_x', 'type': 'range', 'min': 0, 'max': 120, 'sparse': True},
    52: {'name': 'pass.end_location_y', 'type': 'range', 'min': 0, 'max': 80, 'sparse': True},
    53: {'name': 'pass.backheel', 'type': 'binary', 'sparse': True},
    54: {'name': 'pass.deflected', 'type': 'binary', 'sparse': True},
    55: {'name': 'pass.miscommunication', 'type': 'binary', 'sparse': True},
    56: {'name': 'pass.cross', 'type': 'binary', 'sparse': True},
    57: {'name': 'pass.cut_back', 'type': 'binary', 'sparse': True},
    58: {'name': 'pass.switch', 'type': 'binary', 'sparse': True},
    59: {'name': 'pass.shot_assist', 'type': 'binary', 'sparse': True},
    60: {'name': 'pass.goal_assist', 'type': 'binary', 'sparse': True},
    61: {'name': 'pass.body_part.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    62: {'name': 'pass.outcome.id', 'type': 'categorical', 'num_categories': 5, 'sparse': True},
    63: {'name': 'pass.technique.id', 'type': 'categorical', 'num_categories': 4, 'sparse': True},
    64: {'name': 'pass.recipient_position', 'type': 'range', 'min': 0, 'max': 1, 'sparse': True},
    
    # 50-50 (65)
    65: {'name': '50_50.outcome.id', 'type': 'categorical', 'num_categories': 4, 'sparse': True},
    
    # Miscontrol (66)
    66: {'name': 'miscontrol.aerial_won', 'type': 'binary', 'sparse': True},
    
    # Injury stoppage (67)
    67: {'name': 'injury_stoppage.in_chain', 'type': 'binary', 'sparse': True},
    
    # Ball receipt (68)
    68: {'name': 'ball_receipt.outcome.id', 'type': 'categorical', 'num_categories': 1, 'sparse': True},
    
    # Carry (69-70)
    69: {'name': 'carry.end_location_x', 'type': 'range', 'min': 0, 'max': 120, 'sparse': True},
    70: {'name': 'carry.end_location_y', 'type': 'range', 'min': 0, 'max': 80, 'sparse': True},
    
    # Shot features (71-83)
    71: {'name': 'shot.type.id', 'type': 'categorical', 'num_categories': 4, 'sparse': True},
    72: {'name': 'shot.end_location_x', 'type': 'range', 'min': 0, 'max': 120, 'sparse': True},
    73: {'name': 'shot.end_location_y', 'type': 'range', 'min': 0, 'max': 80, 'sparse': True},
    74: {'name': 'shot.end_location_z', 'type': 'range', 'min': 0, 'max': 5, 'sparse': True},
    75: {'name': 'shot.aerial_won', 'type': 'binary', 'sparse': True},
    76: {'name': 'shot.follows_dribble', 'type': 'binary', 'sparse': True},
    77: {'name': 'shot.first_time', 'type': 'binary', 'sparse': True},
    78: {'name': 'shot.open_goal', 'type': 'binary', 'sparse': True},
    79: {'name': 'shot.statsbomb_xg', 'type': 'range', 'min': 0, 'max': 1, 'sparse': True},
    80: {'name': 'shot.deflected', 'type': 'binary', 'sparse': True},
    81: {'name': 'shot.technique.id', 'type': 'categorical', 'num_categories': 7, 'sparse': True},
    82: {'name': 'shot.body_part.id', 'type': 'categorical', 'num_categories': 4, 'sparse': True},
    83: {'name': 'shot.outcome.id', 'type': 'categorical', 'num_categories': 8, 'sparse': True},
    
    # Freeze frame features (84-127) - 22 players × 2 values (x, y locations)
    **{i: {'name': f'freeze_frame.player_{(i-84)//2}.{"x" if (i-84)%2==0 else "y"}', 
           'type': 'range', 'min': 0, 'max': 120 if (i-84)%2==0 else 80, 'sparse': True} 
       for i in range(84, 128)}
}

# Event type ID to normalized value mapping (for validation)
# Shot event type.id = 16, which is at position 9 in the sorted list of 29 event types
SHOT_EVENT_TYPE_NORMALIZED = 9 / 29  # ≈ 0.3103


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
        
        # Load autoencoder for decoding (vector validation)
        self.autoencoder = None
        if os.path.exists(args.autoencoder_path):
            try:
                logger.info("Loading autoencoder for vector validation...")
                self.autoencoder = self._load_autoencoder(args.autoencoder_path)
            except Exception as e:
                logger.warning(f"Could not load autoencoder for validation: {e}")
    
    def _load_autoencoder(self, model_path):
        """Load the autoencoder for decoding embeddings."""
        # Determine input_dim from the model file
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Try to infer input_dim from the checkpoint
        if isinstance(checkpoint, dict) and 'encoder.0.weight' in checkpoint:
            input_dim = checkpoint['encoder.0.weight'].shape[1]
        else:
            input_dim = 128  # Default assumption
        
        model = EventAutoencoder(input_dim=input_dim, latent_dim=self.args.embedding_dim)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        return model
    
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

    # ========================================================================
    # Vector Validation Methods (for decoded 128-dim event vectors)
    # ========================================================================
    
    @torch.no_grad()
    def decode_embeddings(self, embeddings):
        """
        Decode latent embeddings to full event vectors using the autoencoder.
        
        Args:
            embeddings: Latent embeddings (N, seq_len, 32)
            
        Returns:
            Decoded vectors (N, seq_len, 128) or None if autoencoder unavailable
        """
        if self.autoencoder is None:
            logger.warning("Autoencoder not loaded - cannot decode embeddings")
            return None
        
        # Flatten for batch processing
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        # Convert to tensor and decode
        tensor_embeddings = torch.tensor(flat_embeddings, dtype=torch.float32).to(self.device)
        decoded = self.autoencoder.decoder(tensor_embeddings)
        
        # Reshape back
        decoded_np = decoded.cpu().numpy()
        decoded_np = decoded_np.reshape(original_shape[0], original_shape[1], -1)
        
        return decoded_np
    
    def calculate_vector_validation_metrics(self, decoded_vectors, 
                                            epsilon_values=[0, 1e-10, 1e-6, 1e-3, 1e-2]):
        """
        Comprehensive validation of decoded event vectors.
        
        Args:
            decoded_vectors: Decoded vectors (N, seq_len, 128)
            epsilon_values: List of epsilon tolerances for range checks
            
        Returns:
            Dictionary containing all validation metrics
        """
        if decoded_vectors is None:
            return {'error': 'No decoded vectors available'}
        
        logger.info("="*60)
        logger.info("VECTOR VALIDATION METRICS")
        logger.info("="*60)
        
        metrics = {}
        
        # Flatten to (N*seq_len, 128) for per-event analysis
        flat_vectors = decoded_vectors.reshape(-1, decoded_vectors.shape[-1])
        total_events = flat_vectors.shape[0]
        total_values = flat_vectors.size
        
        logger.info(f"Validating {total_events} events ({total_values} total values)")
        
        # 1. Basic Range Validation with Epsilon Tolerance
        metrics['range_validation'] = self._validate_range_with_epsilon(
            flat_vectors, epsilon_values
        )
        
        # 2. Per-Feature Statistics
        metrics['feature_statistics'] = self._calculate_feature_statistics(flat_vectors)
        
        # 3. Categorical Feature Validation
        metrics['categorical_validation'] = self._validate_categorical_features(flat_vectors)
        
        # 4. Binary Feature Validation
        metrics['binary_validation'] = self._validate_binary_features(flat_vectors)
        
        # 5. Sparsity Analysis
        metrics['sparsity_analysis'] = self._analyze_sparsity(flat_vectors)
        
        # 6. Event Type Consistency
        metrics['event_type_consistency'] = self._validate_event_type_consistency(flat_vectors)
        
        # 7. Location Distribution Analysis
        metrics['location_analysis'] = self._analyze_locations(flat_vectors)
        
        return metrics
    
    def _validate_range_with_epsilon(self, vectors, epsilon_values):
        """
        Check how many values fall outside [0, 1] range with various epsilon tolerances.
        
        Args:
            vectors: Flattened vectors (N, 128)
            epsilon_values: List of epsilon tolerances
            
        Returns:
            Dictionary with range validation results per epsilon
        """
        results = {}
        total_values = vectors.size
        
        logger.info("\n--- Range Validation (with epsilon tolerance) ---")
        
        for eps in epsilon_values:
            below_zero = np.sum(vectors < (0 - eps))
            above_one = np.sum(vectors > (1 + eps))
            in_range = total_values - below_zero - above_one
            
            eps_key = f'eps_{eps}'
            results[eps_key] = {
                'epsilon': eps,
                'total_values': int(total_values),
                'below_zero_count': int(below_zero),
                'above_one_count': int(above_one),
                'in_range_count': int(in_range),
                'below_zero_pct': float(below_zero / total_values * 100),
                'above_one_pct': float(above_one / total_values * 100),
                'in_range_pct': float(in_range / total_values * 100),
            }
            
            # Also track the magnitude of violations
            if below_zero > 0:
                violations_below = vectors[vectors < (0 - eps)]
                results[eps_key]['min_violation_below'] = float(np.min(violations_below))
                results[eps_key]['mean_violation_below'] = float(np.mean(violations_below))
            
            if above_one > 0:
                violations_above = vectors[vectors > (1 + eps)]
                results[eps_key]['max_violation_above'] = float(np.max(violations_above))
                results[eps_key]['mean_violation_above'] = float(np.mean(violations_above))
            
            logger.info(f"  eps={eps}: in_range={results[eps_key]['in_range_pct']:.4f}%, "
                       f"below_0={results[eps_key]['below_zero_pct']:.4f}%, "
                       f"above_1={results[eps_key]['above_one_pct']:.4f}%")
        
        return results
    
    def _calculate_feature_statistics(self, vectors):
        """
        Calculate statistics for each of the 128 features.
        
        Args:
            vectors: Flattened vectors (N, 128)
            
        Returns:
            Dictionary with per-feature statistics
        """
        logger.info("\n--- Per-Feature Statistics ---")
        
        results = {
            'per_feature': {},
            'summary': {}
        }
        
        num_features = vectors.shape[1]
        
        for idx in range(num_features):
            feature_values = vectors[:, idx]
            feature_def = FEATURE_DEFINITIONS.get(idx, {'name': f'unknown_{idx}', 'type': 'unknown'})
            
            stats = {
                'name': feature_def['name'],
                'type': feature_def.get('type', 'unknown'),
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values)),
                'median': float(np.median(feature_values)),
                'zero_pct': float(np.sum(feature_values == 0) / len(feature_values) * 100),
                'negative_pct': float(np.sum(feature_values < 0) / len(feature_values) * 100),
                'above_one_pct': float(np.sum(feature_values > 1) / len(feature_values) * 100),
            }
            
            results['per_feature'][idx] = stats
        
        # Summary statistics
        all_means = [results['per_feature'][i]['mean'] for i in range(num_features)]
        all_stds = [results['per_feature'][i]['std'] for i in range(num_features)]
        
        results['summary'] = {
            'num_features': num_features,
            'avg_feature_mean': float(np.mean(all_means)),
            'avg_feature_std': float(np.mean(all_stds)),
            'features_with_negatives': sum(1 for i in range(num_features) 
                                          if results['per_feature'][i]['negative_pct'] > 0),
            'features_with_above_one': sum(1 for i in range(num_features) 
                                          if results['per_feature'][i]['above_one_pct'] > 0),
        }
        
        logger.info(f"  Features with negative values: {results['summary']['features_with_negatives']}")
        logger.info(f"  Features with values > 1: {results['summary']['features_with_above_one']}")
        
        return results
    
    def _validate_categorical_features(self, vectors, tolerance=0.05):
        """
        Validate that categorical features have values close to valid discrete values.
        
        For a categorical with n categories, valid values are: 0, 1/n, 2/n, ..., n/n
        
        Args:
            vectors: Flattened vectors (N, 128)
            tolerance: Maximum allowed deviation from valid discrete value
            
        Returns:
            Dictionary with categorical validation results
        """
        logger.info("\n--- Categorical Feature Validation ---")
        
        results = {'per_feature': {}, 'summary': {}}
        total_valid = 0
        total_checked = 0
        
        for idx, feature_def in FEATURE_DEFINITIONS.items():
            if feature_def.get('type') != 'categorical':
                continue
            
            num_categories = feature_def.get('num_categories', 1)
            feature_values = vectors[:, idx]
            
            # Valid discrete values for this categorical
            valid_values = [i / num_categories for i in range(num_categories + 1)]
            
            # Check each value against valid values
            valid_count = 0
            for val in feature_values:
                is_valid = any(abs(val - valid_val) <= tolerance for valid_val in valid_values)
                if is_valid:
                    valid_count += 1
            
            valid_pct = valid_count / len(feature_values) * 100
            
            results['per_feature'][idx] = {
                'name': feature_def['name'],
                'num_categories': num_categories,
                'valid_values': valid_values,
                'valid_count': int(valid_count),
                'total_count': int(len(feature_values)),
                'valid_pct': float(valid_pct),
                'tolerance': tolerance,
            }
            
            total_valid += valid_count
            total_checked += len(feature_values)
        
        results['summary'] = {
            'total_categorical_features': len(results['per_feature']),
            'overall_valid_pct': float(total_valid / total_checked * 100) if total_checked > 0 else 0,
            'tolerance_used': tolerance,
        }
        
        logger.info(f"  Categorical features valid: {results['summary']['overall_valid_pct']:.2f}% "
                   f"(tolerance={tolerance})")
        
        return results
    
    def _validate_binary_features(self, vectors, tolerance=0.05):
        """
        Validate that binary features have values close to 0, 0.5, or 1.
        
        Binary features use CategoricalFeatureParser with categories [0, 1],
        so valid normalized values are: 0 (not present), 0.5 (False), 1.0 (True)
        
        Args:
            vectors: Flattened vectors (N, 128)
            tolerance: Maximum allowed deviation
            
        Returns:
            Dictionary with binary validation results
        """
        logger.info("\n--- Binary Feature Validation ---")
        
        results = {'per_feature': {}, 'summary': {}}
        valid_binary_values = [0.0, 0.5, 1.0]
        total_valid = 0
        total_checked = 0
        
        for idx, feature_def in FEATURE_DEFINITIONS.items():
            if feature_def.get('type') != 'binary':
                continue
            
            feature_values = vectors[:, idx]
            
            # Check each value
            valid_count = 0
            value_distribution = {0.0: 0, 0.5: 0, 1.0: 0, 'other': 0}
            
            for val in feature_values:
                matched = False
                for valid_val in valid_binary_values:
                    if abs(val - valid_val) <= tolerance:
                        valid_count += 1
                        value_distribution[valid_val] += 1
                        matched = True
                        break
                if not matched:
                    value_distribution['other'] += 1
            
            valid_pct = valid_count / len(feature_values) * 100
            
            results['per_feature'][idx] = {
                'name': feature_def['name'],
                'valid_count': int(valid_count),
                'total_count': int(len(feature_values)),
                'valid_pct': float(valid_pct),
                'distribution': {k: int(v) for k, v in value_distribution.items()},
            }
            
            total_valid += valid_count
            total_checked += len(feature_values)
        
        results['summary'] = {
            'total_binary_features': len(results['per_feature']),
            'overall_valid_pct': float(total_valid / total_checked * 100) if total_checked > 0 else 0,
            'tolerance_used': tolerance,
        }
        
        logger.info(f"  Binary features valid: {results['summary']['overall_valid_pct']:.2f}% "
                   f"(tolerance={tolerance})")
        
        return results
    
    def _analyze_sparsity(self, vectors, zero_threshold=0.01):
        """
        Analyze sparsity patterns in the vectors.
        
        Args:
            vectors: Flattened vectors (N, 128)
            zero_threshold: Values below this are considered zero
            
        Returns:
            Dictionary with sparsity analysis
        """
        logger.info("\n--- Sparsity Analysis ---")
        
        results = {'per_feature': {}, 'summary': {}}
        
        expected_sparse_indices = [idx for idx, feat in FEATURE_DEFINITIONS.items() 
                                   if feat.get('sparse', False)]
        
        for idx in range(vectors.shape[1]):
            feature_values = vectors[:, idx]
            zero_count = np.sum(np.abs(feature_values) < zero_threshold)
            sparsity = zero_count / len(feature_values) * 100
            
            feature_def = FEATURE_DEFINITIONS.get(idx, {})
            expected_sparse = feature_def.get('sparse', False)
            
            results['per_feature'][idx] = {
                'name': feature_def.get('name', f'feature_{idx}'),
                'sparsity_pct': float(sparsity),
                'expected_sparse': expected_sparse,
                'zero_count': int(zero_count),
            }
        
        # Summary
        avg_sparsity = np.mean([results['per_feature'][i]['sparsity_pct'] 
                               for i in range(vectors.shape[1])])
        sparse_feature_sparsity = np.mean([results['per_feature'][i]['sparsity_pct'] 
                                          for i in expected_sparse_indices]) if expected_sparse_indices else 0
        
        results['summary'] = {
            'overall_avg_sparsity_pct': float(avg_sparsity),
            'expected_sparse_features_avg_sparsity': float(sparse_feature_sparsity),
            'num_expected_sparse_features': len(expected_sparse_indices),
        }
        
        logger.info(f"  Overall average sparsity: {avg_sparsity:.2f}%")
        logger.info(f"  Sparse features (expected) avg sparsity: {sparse_feature_sparsity:.2f}%")
        
        return results
    
    def _validate_event_type_consistency(self, vectors, tolerance=0.05):
        """
        Validate that event type determines which features are non-zero.
        
        For example, shot events (type.id ≈ 0.3) should have non-zero shot features (71-83).
        
        Args:
            vectors: Flattened vectors (N, 128)
            tolerance: Tolerance for matching event type
            
        Returns:
            Dictionary with consistency analysis
        """
        logger.info("\n--- Event Type Consistency ---")
        
        results = {}
        
        # Get event types (index 0)
        event_types = vectors[:, 0]
        
        # Count events by type (binned)
        type_bins = np.round(event_types * 34) / 34  # Round to nearest valid categorical
        unique_types, type_counts = np.unique(type_bins, return_counts=True)
        
        results['event_type_distribution'] = {
            float(t): int(c) for t, c in zip(unique_types, type_counts)
        }
        
        # Check shot events specifically
        shot_type_value = SHOT_EVENT_TYPE_NORMALIZED
        is_shot = np.abs(event_types - shot_type_value) < tolerance
        shot_count = np.sum(is_shot)
        
        if shot_count > 0:
            shot_vectors = vectors[is_shot]
            # Check if shot features (71-83) are non-zero for shot events
            shot_features = shot_vectors[:, 71:84]
            shot_features_active = np.sum(np.abs(shot_features) > 0.01, axis=1)
            avg_active_shot_features = np.mean(shot_features_active)
            
            results['shot_events'] = {
                'count': int(shot_count),
                'percentage': float(shot_count / len(vectors) * 100),
                'avg_active_shot_features': float(avg_active_shot_features),
            }
        else:
            results['shot_events'] = {
                'count': 0,
                'percentage': 0.0,
                'avg_active_shot_features': 0.0,
            }
        
        # Check non-shot events don't have shot features active
        non_shot_vectors = vectors[~is_shot]
        if len(non_shot_vectors) > 0:
            non_shot_shot_features = non_shot_vectors[:, 71:84]
            non_shot_active = np.sum(np.abs(non_shot_shot_features) > 0.01)
            
            results['non_shot_events_with_shot_features'] = {
                'count': int(non_shot_active),
                'percentage': float(non_shot_active / non_shot_shot_features.size * 100),
            }
        
        logger.info(f"  Shot events: {results['shot_events']['count']} "
                   f"({results['shot_events']['percentage']:.2f}%)")
        
        return results
    
    def _analyze_locations(self, vectors):
        """
        Analyze location feature distributions (should be spread across the pitch).
        
        Args:
            vectors: Flattened vectors (N, 128)
            
        Returns:
            Dictionary with location analysis
        """
        logger.info("\n--- Location Distribution Analysis ---")
        
        results = {}
        
        # Primary location (indices 2, 3)
        x_locations = vectors[:, 2]
        y_locations = vectors[:, 3]
        
        results['primary_location'] = {
            'x': {
                'mean': float(np.mean(x_locations)),
                'std': float(np.std(x_locations)),
                'min': float(np.min(x_locations)),
                'max': float(np.max(x_locations)),
                'in_range_pct': float(np.sum((x_locations >= 0) & (x_locations <= 1)) / len(x_locations) * 100),
            },
            'y': {
                'mean': float(np.mean(y_locations)),
                'std': float(np.std(y_locations)),
                'min': float(np.min(y_locations)),
                'max': float(np.max(y_locations)),
                'in_range_pct': float(np.sum((y_locations >= 0) & (y_locations <= 1)) / len(y_locations) * 100),
            }
        }
        
        # Check if locations are reasonably distributed (not all zeros or all ones)
        x_spread = results['primary_location']['x']['std']
        y_spread = results['primary_location']['y']['std']
        
        results['distribution_quality'] = {
            'x_well_distributed': x_spread > 0.1,  # Reasonable spread
            'y_well_distributed': y_spread > 0.1,
            'x_spread': float(x_spread),
            'y_spread': float(y_spread),
        }
        
        logger.info(f"  Location X: mean={results['primary_location']['x']['mean']:.3f}, "
                   f"std={x_spread:.3f}")
        logger.info(f"  Location Y: mean={results['primary_location']['y']['mean']:.3f}, "
                   f"std={y_spread:.3f}")
        
        return results
    
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
        
        # ================================================================
        # Vector Validation (decode and validate 128-dim vectors)
        # ================================================================
        if self.autoencoder is not None:
            logger.info("\nDecoding embeddings for vector validation...")
            
            # Decode generated samples
            decoded_generated = self.decode_embeddings(generated)
            
            if decoded_generated is not None:
                # Run comprehensive vector validation
                vector_metrics = self.calculate_vector_validation_metrics(
                    decoded_generated,
                    epsilon_values=[0, 1e-10, 1e-6, 1e-3, 1e-2, 0.05]
                )
                results['vector_validation'] = vector_metrics
                
                # Also validate real samples for comparison
                logger.info("\nValidating real samples for comparison...")
                decoded_real = self.decode_embeddings(real)
                if decoded_real is not None:
                    real_vector_metrics = self.calculate_vector_validation_metrics(
                        decoded_real,
                        epsilon_values=[0, 1e-10, 1e-6, 1e-3, 1e-2, 0.05]
                    )
                    results['vector_validation_real'] = real_vector_metrics
                
                # Save decoded samples
                decoded_path = os.path.join(self.args.output_dir, 
                                           f"{self.model_type}_decoded_samples.pkl")
                with open(decoded_path, 'wb') as f:
                    pickle.dump({
                        'decoded_generated': decoded_generated,
                        'decoded_real': decoded_real
                    }, f)
                logger.info(f"Decoded samples saved to {decoded_path}")
        else:
            logger.warning("Autoencoder not available - skipping vector validation")

        # Save results
        output_path = os.path.join(self.args.output_dir, f"{self.model_type}_evaluation_results.json")
        with open(output_path, 'w') as f:
            # Custom JSON encoder for nested dicts with numpy types
            json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
        logger.info(f"Results saved to {output_path}")

        # Save samples
        samples_path = os.path.join(self.args.output_dir, f"{self.model_type}_generated_samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump({
                'generated': generated,
                'real': real
            }, f)
        logger.info(f"Samples saved to {samples_path}")
        
        # Print summary
        self._print_validation_summary(results)

        return results
    
    def _print_validation_summary(self, results):
        """Print a concise summary of validation results."""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        if 'statistical' in results:
            logger.info(f"Fréchet Distance: {results['statistical'].get('frechet_distance', 'N/A'):.4f}")
        
        logger.info(f"Coverage: {results.get('coverage', 'N/A'):.4f}")
        logger.info(f"Combined Score: {results.get('combined_score', 'N/A'):.4f}")
        
        if 'vector_validation' in results:
            vv = results['vector_validation']
            
            # Range validation summary
            if 'range_validation' in vv:
                eps_0 = vv['range_validation'].get('eps_0', {})
                logger.info(f"\nVector Range (eps=0):")
                logger.info(f"  In range [0,1]: {eps_0.get('in_range_pct', 0):.2f}%")
                logger.info(f"  Below 0: {eps_0.get('below_zero_pct', 0):.2f}%")
                logger.info(f"  Above 1: {eps_0.get('above_one_pct', 0):.2f}%")
            
            # Categorical/Binary validation
            if 'categorical_validation' in vv:
                cat_valid = vv['categorical_validation'].get('summary', {}).get('overall_valid_pct', 0)
                logger.info(f"  Categorical features valid: {cat_valid:.2f}%")
            
            if 'binary_validation' in vv:
                bin_valid = vv['binary_validation'].get('summary', {}).get('overall_valid_pct', 0)
                logger.info(f"  Binary features valid: {bin_valid:.2f}%")
        
        logger.info("="*60)
    
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
        # duration, under_pressure, out, counterpress, unified_time, second_zeroed, position
        # Plus special parsers: minute_zeroed, team, possession_team, player
        self.event_type_idx = 0
        self.unified_time_idx = 8
        # Maximum total match seconds used for normalization (from UnifiedTimeParser)
        self.MAX_MATCH_SECONDS = 9059

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
        """
        Convert normalized event type back to event ID.

        CategoricalFeatureParser encodes as: position / num_categories
        where position is 1-indexed.  Denormalize: position = round(val * N),
        then index = position - 1.
        """
        from tokenizer.config import event_ids

        all_event_ids = sorted(event_ids.values())
        num_categories = len(all_event_ids)
        position = int(round(normalized_value * num_categories))
        # Clamp to valid 1-indexed range
        position = max(1, min(num_categories, position))
        return all_event_ids[position - 1]

    def _get_event_name(self, event_id):
        """Get event name from event ID."""
        idx = self.event_mapping['id_to_index'].get(event_id)
        if idx is not None:
            return self.event_mapping['index_to_name'][idx]
        return None

    def _extract_timestamp(self, event_vector):
        """
        Extract timestamp from event vector using the unified time feature.

        The unified time at index 8 is normalised total match seconds:
            total_seconds = normalised_value * MAX_MATCH_SECONDS

        Returns:
            total_seconds (float)
        """
        normalised_time = event_vector[self.unified_time_idx]
        total_seconds = normalised_time * self.MAX_MATCH_SECONDS
        return total_seconds

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
            total_seconds = self._extract_timestamp(sequence[i])
            approx_min = int(total_seconds // 60)
            approx_sec = int(total_seconds % 60)

            if total_seconds < prev_timestamp - self.time_tolerance:
                time_violations += 1
                violations.append(
                    f"Non-monotonic timestamp at step {i}: "
                    f"{approx_min:02d}:{approx_sec:02d} ({total_seconds:.0f}s) < "
                    f"previous ({prev_timestamp:.0f}s), tolerance: {self.time_tolerance}s"
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

