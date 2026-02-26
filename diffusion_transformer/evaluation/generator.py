"""
Generator module for naive generation task.
"""

import os
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.visualization.visualizer import Visualizer
from diffusion_transformer.utils import get_logger
from diffusion_transformer.data.event_autoencoder_model import EventAutoencoder

logger = get_logger(__name__)


class Generator:
    """
    Generator for naive generation task.
    Handles loading models, sampling from noise, and creating visualizations.
    """
    
    def __init__(self, args, events_dict: dict[str, np.ndarray] = None, embeddings_dict: dict[str, np.ndarray] = None):
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
            checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
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
            model = EventAutoencoder()
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            model = model.to(self.device)
            model.eval()
            return model
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

        decoded = self.autoencoder.decoder(torch.tensor(samples).to(self.device))
        return decoded.detach().cpu().numpy()

    
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
        
        # Decode ALL samples if autoencoder is available
        decoded = self.decode_samples(all_samples)
        
        # Save samples (including decoded samples for extract_event_sequences.py)
        output_path = os.path.join(self.args.output_dir, f"{self.model_type}_generated_samples.pkl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        save_data = {
            'samples': all_samples,
            'trajectory': trajectory,
            'model_type': self.model_type,
            'args': vars(self.args)
        }
        
        # Add decoded samples if available (for extract_event_sequences.py)
        if decoded is not None:
            save_data['decoded_generated'] = decoded
            logger.info(f"Decoded samples shape: {decoded.shape}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Generated samples saved to {output_path}")
        logger.info(f"Sample shape: {all_samples.shape}")

        # Save decoded samples as CSVs
        if decoded is not None:
            self.save_decoded_samples_csv(decoded, all_samples)
        
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

    def generate_valid_sequences(self):
        """
        Generate sequences in batches, evaluate each for validity using
        SequenceEvaluator, and keep only valid ones until the target count
        is reached.  Then run t-SNE visualisation and event-type distribution
        analysis on the collected valid sequences.

        Validity criteria:
            - 0 transition violations (strict)
            - time_violations <= args.max_time_violations (configurable)
        """
        from diffusion_transformer.evaluation.evaluator import SequenceEvaluator

        logger.info("=" * 60)
        logger.info("GENERATE VALID SEQUENCES")
        logger.info("=" * 60)

        # ---- pre-flight checks ------------------------------------------------
        if self.autoencoder is None:
            raise RuntimeError(
                "Autoencoder is required for generate_valid (decoding + evaluation). "
                "Please provide a valid --autoencoder_path."
            )

        target = self.args.num_valid_sequences
        batch_size = self.args.num_gen_samples
        max_time_violations = self.args.max_time_violations
        use_ddim = self.args.ddim_steps < self.args.num_timesteps

        logger.info(f"Target valid sequences : {target}")
        logger.info(f"Batch size             : {batch_size}")
        logger.info(f"Max time violations    : {max_time_violations}")
        logger.info("Transition tolerance   : 0 (strict)")
        logger.info(f"Using DDIM sampling    : {use_ddim}")

        # ---- initialise evaluator ---------------------------------------------
        evaluator = SequenceEvaluator(
            cache_dir=self.args.cache_dir,
            transition_tolerance=0.0,        # strict – no illegal transitions
            time_tolerance=0,                # per-event; we do per-sequence filtering below
        )

        # ---- collection loop ---------------------------------------------------
        valid_latent = []      # latent embeddings  (N, seq_len, emb_dim)
        valid_decoded = []     # decoded 128-dim     (N, seq_len, 128)
        total_generated = 0
        round_idx = 0

        while len(valid_latent) < target:
            round_idx += 1
            needed = target - len(valid_latent)
            current_batch = max(batch_size, needed)  # generate at least batch_size

            logger.info(
                f"\n--- Round {round_idx}: generating {current_batch} samples "
                f"(collected {len(valid_latent)}/{target}) ---"
            )

            # generate latent samples
            latent_samples = self.generate_samples(
                current_batch, use_ddim=use_ddim, return_trajectory=False
            )
            total_generated += current_batch

            # decode to 128-dim event vectors
            decoded_samples = self.decode_samples(latent_samples)
            if decoded_samples is None:
                raise RuntimeError("Decoding returned None – autoencoder issue.")

            # evaluate each sequence individually
            for i in range(len(decoded_samples)):
                result = evaluator.evaluate_sequence(decoded_samples[i])
                is_valid = (
                    result['transition_violations'] == 0
                    and result['time_violations'] <= max_time_violations
                )
                if is_valid:
                    valid_latent.append(latent_samples[i])
                    valid_decoded.append(decoded_samples[i])
                    if len(valid_latent) >= target:
                        break

            logger.info(
                f"Round {round_idx} done – kept "
                f"{len(valid_latent)}/{target} valid sequences "
                f"(total generated so far: {total_generated})"
            )

        # stack into arrays
        valid_latent = np.stack(valid_latent[:target], axis=0)
        valid_decoded = np.stack(valid_decoded[:target], axis=0)

        logger.info(f"\nCollection complete: {target} valid sequences "
                     f"out of {total_generated} total generated")
        logger.info(f"Valid latent shape : {valid_latent.shape}")
        logger.info(f"Valid decoded shape: {valid_decoded.shape}")

        # ---- save artefacts ---------------------------------------------------
        output_dir = os.path.join(self.args.output_dir, 'valid_sequences')
        os.makedirs(output_dir, exist_ok=True)

        # pickle for downstream use
        pkl_path = os.path.join(output_dir, f"{self.model_type}_valid_samples.pkl")
        save_data = {
            'latent': valid_latent,
            'decoded': valid_decoded,
            'model_type': self.model_type,
            'total_generated': total_generated,
            'target': target,
            'max_time_violations': max_time_violations,
            'args': vars(self.args),
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved valid samples to {pkl_path}")

        # save decoded CSVs
        self.save_decoded_samples_csv(valid_decoded, valid_latent,
                                       output_subdir='valid_sequences/decoded_csvs')

        # ---- post-analysis: t-SNE visualisation -------------------------------
        logger.info("Running t-SNE visualisation on valid sequences...")
        vis = Visualizer(self.args, self.events_dict, self.embeddings_dict)
        # Temporarily redirect visualizer output into the valid_sequences subdir
        original_output_dir = vis.output_dir
        vis.output_dir = output_dir
        vis.visualize_embedding_space(valid_latent)
        vis.output_dir = original_output_dir

        # ---- post-analysis: event type distribution ----------------------------
        self.analyze_event_type_distribution(valid_decoded, output_dir)

        logger.info("=" * 60)
        logger.info(f"GENERATE VALID SEQUENCES completed – results in {output_dir}")
        logger.info("=" * 60)

        return valid_latent, valid_decoded

    # ------------------------------------------------------------------
    # Event-type distribution analysis
    # ------------------------------------------------------------------
    def analyze_event_type_distribution(self, decoded_sequences, output_dir):
        """
        Decode event types from 128-dim vectors using the boundary-based
        mapping from ``extract_event_sequences``, compute counts and
        percentages, and save as CSV.

        Args:
            decoded_sequences: np.ndarray of shape (N, seq_len, 128)
            output_dir: directory to write the CSV into
        """
        # Import mapping utilities from extract_event_sequences
        import sys
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from extract_event_sequences import build_event_mapping, get_event_name

        logger.info("Analysing event-type distribution of valid sequences...")

        _, _, _, boundaries = build_event_mapping()

        # Count events
        from collections import Counter
        event_counter = Counter()
        total_events = 0

        for seq in decoded_sequences:
            for event_vec in seq:
                raw_val = event_vec[0]  # feature 0 = event type
                event_name, _, _ = get_event_name(raw_val, boundaries)
                event_counter[event_name] += 1
                total_events += 1

        # Build dataframe sorted by count descending
        rows = []
        for event_name, count in event_counter.most_common():
            rows.append({
                'event_type': event_name,
                'count': count,
                'percentage': count / total_events * 100.0 if total_events else 0.0,
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'event_type_distribution.csv')
        df.to_csv(csv_path, index=False)

        logger.info(f"Event type distribution ({len(df)} types, {total_events} total events) "
                     f"saved to {csv_path}")
        logger.info(f"\n{df.to_string(index=False)}")

        return df


    def save_decoded_samples_csv(self, decoded_samples, latent_samples, output_subdir='decoded_sequences'):
        """
        Save decoded samples as CSV files in a logical structure.

        Args:
            decoded_samples: Decoded event sequences (N, seq_len, event_features)
            latent_samples: Original latent embeddings (N, seq_len, embedding_dim)
            output_subdir: Subdirectory under output_dir to write CSVs into
        """

        logger.info("Saving decoded samples as CSVs...")

        # Create output directory for decoded samples
        decoded_dir = os.path.join(self.args.output_dir, output_subdir)
        os.makedirs(decoded_dir, exist_ok=True)

        num_sequences, seq_len, num_features = decoded_samples.shape

        # Save each sequence as a separate CSV
        for seq_idx in range(num_sequences):
            sequence = decoded_samples[seq_idx]  # Shape: (seq_len, num_features)

            # Create DataFrame with event features
            df = pd.DataFrame(sequence, columns=[f'feature_{i}' for i in range(num_features)])
            df.insert(0, 'event_index', range(seq_len))

            # Save individual sequence
            sequence_path = os.path.join(decoded_dir, f'sequence_{seq_idx:04d}.csv')
            df.to_csv(sequence_path, index=False)

        logger.info(f"Saved {num_sequences} decoded sequences to {decoded_dir}")

        # Also save a summary file with all sequences combined
        summary_data = []
        for seq_idx in range(num_sequences):
            sequence = decoded_samples[seq_idx]
            for event_idx in range(seq_len):
                row = {
                    'sequence_id': seq_idx,
                    'event_index': event_idx,
                }
                # Add features
                for feat_idx in range(num_features):
                    row[f'feature_{feat_idx}'] = sequence[event_idx, feat_idx]
                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.args.output_dir, 'decoded_sequences_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved combined summary to {summary_path}")

        # Save statistics about the decoded sequences
        stats = {
            'num_sequences': num_sequences,
            'sequence_length': seq_len,
            'num_features': num_features,
            'mean_per_feature': sequence.mean(axis=0).tolist(),
            'std_per_feature': sequence.std(axis=0).tolist(),
        }

        stats_df = pd.DataFrame([stats])
        stats_path = os.path.join(self.args.output_dir, 'decoded_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved statistics to {stats_path}")
