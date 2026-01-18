"""
Generator module for naive generation task.
"""

import os
import torch
import numpy as np
import pickle
from tqdm import tqdm

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.visualization.visualizer import Visualizer
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)


class Generator:
    """
    Generator for naive generation task.
    Handles loading models, sampling from noise, and creating visualizations.
    """
    
    def __init__(self, args, events_dict: dict[str, np.ndarray], embeddings_dict: dict[str, np.ndarray]):
        """
        Initialize generator.
        
        Args:
            args: Argument namespace with configuration
            events_dict: Dictionary of event data by match_id
            embeddings_dict: Dictionary of embeddings by match_id
        """
        self.args = args
        self.device = torch.device(args.device)
        self.events_dict = events_dict
        self.embeddings_dict = embeddings_dict
        
        # Load model based on checkpoint or args
        logger.info("Loading model...")
        
        # Try to load checkpoint to get model type
        model_type = args.model_type
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
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
                logger.warning("No checkpoint found. Using randomly initialized model.")
        
        self.model.eval()
        
        # Create diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=args.num_timesteps,
            schedule_type=args.noise_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device
        )
        
        # Load autoencoder if available for decoding
        self.autoencoder = None
        if os.path.exists(args.autoencoder_path):
            try:
                logger.info("Loading autoencoder for decoding...")
                self.autoencoder = self.load_autoencoder(args.autoencoder_path)
            except Exception as e:
                logger.error(f"Could not load autoencoder: {e}")
    
    def load_autoencoder(self, model_path):
        """
        Load pre-trained autoencoder.
        
        Args:
            model_path: Path to the autoencoder checkpoint
        """
        # This is a placeholder - adjust based on actual autoencoder structure
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # TODO: Initialize autoencoder architecture and load weights
            logger.warning("Autoencoder loading not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error loading autoencoder: {e}")
            return None
    
    @torch.no_grad()
    def generate_samples(self, num_samples, use_ddim=False, return_trajectory=False):
        """
        Generate samples from random noise.
        
        Args:
            num_samples: Number of samples to generate
            use_ddim: Whether to use DDIM sampling (faster)
            return_trajectory: Whether to return intermediate steps
            
        Returns:
            Generated samples (and optionally trajectory)
        """
        shape = (num_samples, self.args.sequence_length, self.args.embedding_dim)
        
        logger.info(f"Generating {num_samples} samples...")
        
        if return_trajectory:
            # Generate with trajectory for visualization
            samples, trajectory = self.generate_with_trajectory(shape)
            return samples, trajectory
        
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
    
    @torch.no_grad()
    def generate_with_trajectory(self, shape):
        """
        Generate samples while recording intermediate steps.
        
        Args:
            shape: Shape of samples to generate
            
        Returns:
            final_samples, trajectory (list of samples at different timesteps)
        """
        device = next(self.model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectory = []
        
        # Record at specific timesteps
        record_timesteps = [
            self.diffusion.num_timesteps - 1,
            3 * self.diffusion.num_timesteps // 4,
            self.diffusion.num_timesteps // 2,
            self.diffusion.num_timesteps // 4,
            0
        ]
        
        timesteps = list(range(self.diffusion.num_timesteps))[::-1]
        
        for t in tqdm(timesteps, desc="Generating with trajectory", disable=not self.args.verbose):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t_batch)
            
            # Record trajectory
            if t in record_timesteps:
                trajectory.append({
                    'timestep': t,
                    'samples': x.cpu().numpy()
                })
        
        return x.cpu().numpy(), trajectory
    
    def decode_samples(self, samples):
        """
        Decode latent samples using autoencoder.
        
        Args:
            samples: Latent samples (N, seq_len, embedding_dim)
            
        Returns:
            Decoded events
        """
        if self.autoencoder is None:
            logger.warning("No autoencoder available for decoding")
            return None

        decoded = self.autoencoder.decode(torch.tensor(samples).to(self.device))
        return decoded.cpu().numpy()

    
    def generate_and_visualize(self):
        """Generate samples and create visualizations."""
        logger.info("="*60)
        logger.info("NAIVE GENERATION TASK")
        logger.info("="*60)
        logger.info(f"Generating {self.args.num_gen_samples} samples from random noise...")
        
        # Generate samples
        use_ddim = self.args.ddim_steps < self.args.num_timesteps
        
        # Generate a few samples with trajectory for visualization
        num_traj_samples = min(5, self.args.num_gen_samples)
        samples_with_traj, trajectory = self.generate_samples(
            num_traj_samples,
            use_ddim=use_ddim,
            return_trajectory=True
        )
        
        # Generate remaining samples without trajectory (faster)
        if self.args.num_gen_samples > num_traj_samples:
            remaining_samples = self.generate_samples(
                self.args.num_gen_samples - num_traj_samples,
                use_ddim=use_ddim,
                return_trajectory=False
            )
            all_samples = np.concatenate([samples_with_traj, remaining_samples], axis=0)
        else:
            all_samples = samples_with_traj
        
        # Save samples
        output_path = os.path.join(self.args.output_dir, f"{self.model_type}_generated_samples.pkl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'samples': all_samples,
                'trajectory': trajectory,
                'model_type': self.model_type,
                'args': vars(self.args)
            }, f)
        
        logger.info(f"Generated samples saved to {output_path}")
        logger.info(f"Sample shape: {all_samples.shape}")

        # Decode samples if autoencoder is available
        decoded = self.decode_samples(all_samples[:num_traj_samples])
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = Visualizer(self.args, self.events_dict, self.embeddings_dict)
        
        # Visualize trajectory
        visualizer.visualize_diffusion_trajectory(trajectory)
        
        # Visualize generated samples
        visualizer.visualize_generated_samples(all_samples[:50])  # Visualize first 50
        
        # t-SNE/UMAP visualization if we have enough samples
        if len(all_samples) >= 50:
            visualizer.visualize_embedding_space(all_samples)
        
        logger.info("Visualization completed! Check output directory for results.")
        
        return all_samples, trajectory
