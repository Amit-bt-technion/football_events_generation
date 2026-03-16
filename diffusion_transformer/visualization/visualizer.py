"""
Visualization module for diffusion transformer outputs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pickle
from pathlib import Path
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)

# Ensure project root is on sys.path so extract_event_sequences can be imported
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

FIELD_IMAGE_PATH = os.path.join(parent_dir, 'football_field.jpg')

# Pixel boundaries of the playing field inside football_field.jpg (612x408)
FIELD_BOUNDS = {'left': 24, 'right': 588, 'top': 18, 'bottom': 374}

TEAM_COLORS = {0: '#1f77b4', 1: '#d62728'}  # blue / red

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class Visualizer:
    """Visualizer for diffusion transformer results."""
    
    def __init__(self, args, events_dict=None, embeddings_dict=None):
        """
        Initialize visualizer.
                Args:
            args: Argument namespace with configuration
            events_dict: Optional dictionary of event data by match_id
            embeddings_dict: Optional dictionary of embeddings by match_id
        """
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.events_dict = events_dict
        self.embeddings_dict = embeddings_dict
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def visualize_diffusion_trajectory(self, trajectory):
        """
        Visualize the diffusion process from noise to clean samples.
        
        Args:
            trajectory: List of dicts with 'timestep' and 'samples'
        """
        if not trajectory:
            logger.warning("No trajectory data to visualize")
            return
        
        logger.info("Visualizing diffusion trajectory...")
        
        # Create figure with multiple subplots
        n_steps = len(trajectory)
        fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))
        
        if n_steps == 1:
            axes = [axes]
        
        for i, step in enumerate(trajectory):
            ax = axes[i]
            samples = step['samples']
            timestep = step['timestep']
            
            # Visualize first sample
            sample = samples[0]  # Shape: (seq_len, embedding_dim)
            
            # Plot as heatmap
            im = ax.imshow(sample.T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_title(f't = {timestep}')
            ax.set_xlabel('Event Index')
            ax.set_ylabel('Embedding Dim')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'diffusion_trajectory.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trajectory visualization saved to {output_path}")
        
        # Create animated trajectory
        self.create_trajectory_animation(trajectory)
    
    def create_trajectory_animation(self, trajectory):
        """
        Create animated visualization of diffusion trajectory.
        
        Args:
            trajectory: List of dicts with 'timestep' and 'samples'
        """
        logger.info("Creating trajectory animation...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get first sample from each timestep
        samples = [step['samples'][0] for step in trajectory]
        timesteps = [step['timestep'] for step in trajectory]
        
        # Initialize plot
        im = ax.imshow(samples[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Event Index')
        ax.set_ylabel('Embedding Dim')
        title = ax.set_title(f'Timestep: {timesteps[0]}')
        plt.colorbar(im, ax=ax)
        
        def update(frame):
            im.set_array(samples[frame].T)
            title.set_text(f'Timestep: {timesteps[frame]}')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=len(samples), interval=500, blit=True)
        
        output_path = os.path.join(self.output_dir, 'trajectory_animation.gif')
        writer = PillowWriter(fps=2)
        anim.save(output_path, writer=writer)
        plt.close()
        
        logger.info(f"Animation saved to {output_path}")
    
    def visualize_generated_samples(self, samples):
        """
        Visualize generated samples as heatmaps.
        
        Args:
            samples: Generated samples (N, seq_len, embedding_dim)
        """
        logger.info("Visualizing generated samples...")
        
        n_samples = min(10, len(samples))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(n_samples):
            ax = axes[i]
            sample = samples[i]
            
            im = ax.imshow(sample.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Event Index')
            ax.set_ylabel('Embedding Dim')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'generated_samples_grid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample grid saved to {output_path}")
    
    def visualize_embedding_space(self, samples, real_samples=None, real_event_types=None):
        """
        Visualize embedding space using t-SNE and UMAP, colored by event types.
        Shows clustering of events based on their type, similar to t-SNE/UMAP analysis.
        
        Compares:
        - Generated samples: Embeddings produced by diffusion model (from noise -> denoised embeddings)
        - Real samples: Original embeddings from load_and_embed_matches (encoded real events)
        
        Both are in the same 32D embedding space from the autoencoder, so they can be
        directly compared. Real events are colored by their event type (extracted from
        events_dict). Generated events are shown as black 'x' markers since they don't
        have ground-truth event types.
        
        Args:
            samples: Generated samples from diffusion model (N, seq_len, embedding_dim)
            real_samples: Optional real samples for comparison (not used if events_dict is available)
            real_event_types: Optional event types for real samples (auto-extracted if not provided)
        """
        logger.info("Visualizing embedding space with event type clustering...")
        
        # If we have events_dict and embeddings_dict, extract event types from real data
        if self.events_dict is not None and self.embeddings_dict is not None and real_event_types is None:
            logger.info("Extracting event types from real events...")
            real_event_types = []
            real_embeddings_list = []
            
            # Sample a subset of events for visualization (max 5000)
            max_samples = 5000
            for match_id, events in self.events_dict.items():
                if match_id in self.embeddings_dict:
                    embeddings = self.embeddings_dict[match_id]
                    # Extract event types (index 0 of event vector)
                    event_types = events[:, 0]
                    real_event_types.extend(event_types)
                    real_embeddings_list.append(embeddings)
                    
                    if len(real_event_types) >= max_samples:
                        break
            
            if real_embeddings_list:
                real_embeddings = np.vstack(real_embeddings_list)
                real_event_types = np.array(real_event_types[:max_samples])
                real_embeddings = real_embeddings[:max_samples]
                
                logger.info(f"Using {len(real_event_types)} real events for comparison")
        
        # Determine if we have event type information
        has_event_types = (real_event_types is not None and len(real_event_types) > 0)
        
        if not has_event_types:
            logger.warning("No event type information available. Using basic visualization.")
            # Fall back to simple visualization
            return self._visualize_embedding_space_simple(samples, real_samples)
        
        # Prepare data for dimensionality reduction
        # For generated samples, we'll use individual events from sequences
        logger.info("Preparing generated samples...")
        # Flatten to individual events: (N, seq_len, emb_dim) -> (N*seq_len, emb_dim)
        gen_events = samples.reshape(-1, samples.shape[-1])
        
        # Sample a subset if too large
        max_gen_samples = 5000
        if len(gen_events) > max_gen_samples:
            indices = np.random.choice(len(gen_events), max_gen_samples, replace=False)
            gen_events = gen_events[indices]
        
        # For generated samples, we don't have true event types, so we'll show them separately
        # or try to decode them if we have an autoencoder
        
        # Combine real and generated for joint embedding
        logger.info("Combining real and generated embeddings...")
        all_embeddings = np.vstack([real_embeddings, gen_events])
        
        # Create labels: real event types and a special label for generated
        all_labels = np.concatenate([
            real_event_types,
            np.full(len(gen_events), -1)  # -1 for generated samples
        ])
        
        # Apply PCA for initial dimensionality reduction (if needed)
        logger.info("Applying PCA for dimensionality reduction...")
        if all_embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            all_embeddings_reduced = pca.fit_transform(all_embeddings)
        else:
            all_embeddings_reduced = all_embeddings
        
        # Apply t-SNE
        logger.info("Computing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) // 4))
        tsne_result = tsne.fit_transform(all_embeddings_reduced)
        
        # Split results back
        real_tsne = tsne_result[:len(real_event_types)]
        gen_tsne = tsne_result[len(real_event_types):]
        
        # Apply UMAP if available
        umap_result = None
        if UMAP_AVAILABLE:
            logger.info("Computing UMAP...")
            try:
                umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings) // 10))
                umap_result = umap_reducer.fit_transform(all_embeddings_reduced)
                real_umap = umap_result[:len(real_event_types)]
                gen_umap = umap_result[len(real_event_types):]
            except Exception as e:
                logger.error(f"UMAP failed: {e}")
                umap_result = None
        
        # Calculate silhouette scores for real events only
        try:
            silhouette_tsne = silhouette_score(real_tsne, real_event_types)
            logger.info(f"Silhouette score (t-SNE, real events): {silhouette_tsne:.3f}")
        except Exception:
            silhouette_tsne = None
        
        silhouette_umap = None
        if umap_result is not None:
            try:
                silhouette_umap = silhouette_score(real_umap, real_event_types)
                logger.info(f"Silhouette score (UMAP, real events): {silhouette_umap:.3f}")
            except Exception:
                silhouette_umap = None
        
        # Create visualization
        n_plots = 2 if umap_result is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]
        
        # Get unique event types for coloring
        unique_event_types = np.unique(real_event_types)
        n_event_types = len(unique_event_types)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('tab20' if n_event_types <= 20 else 'hsv')
        colors = [cmap(i / n_event_types) for i in range(n_event_types)]
        event_type_to_color = {et: colors[i] for i, et in enumerate(unique_event_types)}
        
        # Plot t-SNE
        ax = axes[0]
        
        # Plot real events by type
        for i, event_type in enumerate(unique_event_types):
            mask = real_event_types == event_type
            ax.scatter(
                real_tsne[mask, 0], 
                real_tsne[mask, 1],
                c=[event_type_to_color[event_type]],
                alpha=0.6,
                s=20,
                label=f'Real: {event_type:.2f}',
                edgecolors='none'
            )
        
        # Plot generated events
        ax.scatter(
            gen_tsne[:, 0],
            gen_tsne[:, 1],
            c='black',
            alpha=0.3,
            s=15,
            marker='x',
            label='Generated',
            linewidths=1
        )
        
        title = 't-SNE: Event Type Clustering'
        if silhouette_tsne is not None:
            title += f' (Silhouette: {silhouette_tsne:.3f})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        # Add legend with smaller font
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        
        # Plot UMAP if available
        if umap_result is not None:
            ax = axes[1]
            
            # Plot real events by type
            for i, event_type in enumerate(unique_event_types):
                mask = real_event_types == event_type
                ax.scatter(
                    real_umap[mask, 0],
                    real_umap[mask, 1],
                    c=[event_type_to_color[event_type]],
                    alpha=0.6,
                    s=20,
                    label=f'Real: {event_type:.2f}',
                    edgecolors='none'
                )
            
            # Plot generated events
            ax.scatter(
                gen_umap[:, 0],
                gen_umap[:, 1],
                c='black',
                alpha=0.3,
                s=15,
                marker='x',
                label='Generated',
                linewidths=1
            )
            
            title = 'UMAP: Event Type Clustering'
            if silhouette_umap is not None:
                title += f' (Silhouette: {silhouette_umap:.3f})'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            
            # Add legend with smaller font
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'embedding_space.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Embedding space visualization saved to {output_path}")
        logger.info(f"Real events: {len(real_event_types)}, Generated events: {len(gen_events)}")
        logger.info(f"Number of unique event types: {n_event_types}")
    
    def _visualize_embedding_space_simple(self, samples, real_samples=None):
        """
        Simple fallback visualization without event type information.
        
        Args:
            samples: Generated samples
            real_samples: Optional real samples for comparison
        """
        logger.info("Using simple embedding space visualization (no event type info)...")
        
        # Flatten sequences
        samples_flat = samples.reshape(samples.shape[0], -1)
        
        # Apply PCA first for dimensionality reduction
        pca = PCA(n_components=50)
        samples_pca = pca.fit_transform(samples_flat)
        
        # Apply t-SNE
        logger.info("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        if real_samples is not None:
            real_flat = real_samples.reshape(real_samples.shape[0], -1)
            real_pca = pca.transform(real_flat)
            
            # Combine for joint embedding
            combined = np.vstack([samples_pca, real_pca])
            embedded = tsne.fit_transform(combined)
            
            samples_embedded = embedded[:len(samples_pca)]
            real_embedded = embedded[len(samples_pca):]
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # t-SNE plot
            ax1.scatter(real_embedded[:, 0], real_embedded[:, 1], 
                       c='blue', alpha=0.5, label='Real', s=20)
            ax1.scatter(samples_embedded[:, 0], samples_embedded[:, 1], 
                       c='red', alpha=0.5, label='Generated', s=20)
            ax1.set_title('t-SNE Visualization')
            ax1.legend()
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            
            # PCA plot
            ax2.scatter(real_pca[:, 0], real_pca[:, 1], 
                       c='blue', alpha=0.5, label='Real', s=20)
            ax2.scatter(samples_pca[:, 0], samples_pca[:, 1], 
                       c='red', alpha=0.5, label='Generated', s=20)
            ax2.set_title('PCA Visualization')
            ax2.legend()
            ax2.set_xlabel('PC 1')
            ax2.set_ylabel('PC 2')
            
        else:
            # Only generated samples
            embedded = tsne.fit_transform(samples_pca)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # t-SNE plot
            ax1.scatter(embedded[:, 0], embedded[:, 1], 
                       c=range(len(embedded)), cmap='viridis', alpha=0.6, s=20)
            ax1.set_title('t-SNE Visualization (Generated)')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            
            # PCA plot
            ax2.scatter(samples_pca[:, 0], samples_pca[:, 1], 
                       c=range(len(samples_pca)), cmap='viridis', alpha=0.6, s=20)
            ax2.set_title('PCA Visualization (Generated)')
            ax2.set_xlabel('PC 1')
            ax2.set_ylabel('PC 2')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'embedding_space.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Embedding space visualization saved to {output_path}")
    
    def visualize_on_pitch(self, events, output_name='pitch_visualization.png'):
        """
        Visualize event sequences on a football pitch.
        
        Args:
            events: List of events with coordinates (x, y)
            output_name: Output filename
        """
        logger.info("Visualizing events on pitch...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw pitch
        self.draw_pitch(ax)
        
        # Plot events (assuming events have x, y coordinates)
        # This is a placeholder - adjust based on actual event structure
        if hasattr(events, 'shape') and len(events.shape) == 2:
            # If events are just coordinates
            x = events[:, 0] * 105  # Scale to pitch dimensions
            y = events[:, 1] * 68
            
            # Plot trajectory
            ax.plot(x, y, 'o-', color='red', markersize=8, linewidth=2, alpha=0.7)
            
            # Mark start and end
            ax.plot(x[0], y[0], 'go', markersize=12, label='Start')
            ax.plot(x[-1], y[-1], 'r*', markersize=15, label='Shot')
            
        ax.legend()
        ax.set_title('Event Sequence on Pitch')
        
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pitch visualization saved to {output_path}")
    
    def draw_pitch(self, ax, pitch_length=105, pitch_width=68):
        """Draw a football pitch on the given axes."""
        # Pitch outline
        ax.plot([0, 0], [0, pitch_width], color='white', linewidth=2)
        ax.plot([0, pitch_length], [pitch_width, pitch_width], color='white', linewidth=2)
        ax.plot([pitch_length, pitch_length], [pitch_width, 0], color='white', linewidth=2)
        ax.plot([pitch_length, 0], [0, 0], color='white', linewidth=2)
        
        # Middle line
        ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color='white', linewidth=2)
        
        # Center circle
        circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15, color='white', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Penalty areas
        penalty_length = 16.5
        penalty_width = 40.3
        
        # Left penalty area
        rect = patches.Rectangle((0, (pitch_width-penalty_width)/2), penalty_length, penalty_width, 
                                 linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        
        # Right penalty area
        rect = patches.Rectangle((pitch_length-penalty_length, (pitch_width-penalty_width)/2), 
                                 penalty_length, penalty_width, 
                                 linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        
        # Goal areas
        goal_length = 5.5
        goal_width = 18.3
        
        # Left goal area
        rect = patches.Rectangle((0, (pitch_width-goal_width)/2), goal_length, goal_width, 
                                 linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        
        # Right goal area
        rect = patches.Rectangle((pitch_length-goal_length, (pitch_width-goal_width)/2), 
                                 goal_length, goal_width, 
                                 linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_xlim(-5, pitch_length + 5)
        ax.set_ylim(-5, pitch_width + 5)
        ax.set_aspect('equal')
        ax.set_facecolor('#1e7a1e')  # Grass green
        ax.axis('off')
    
    def visualize_metrics(self, metrics_path):
        """
        Visualize evaluation metrics.
        
        Args:
            metrics_path: Path to metrics JSON file
        """
        import json
        
        logger.info("Visualizing metrics...")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create bar chart of metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Statistical metrics
        if 'statistical' in metrics:
            ax = axes[0, 0]
            stat_metrics = metrics['statistical']
            names = list(stat_metrics.keys())
            values = list(stat_metrics.values())
            ax.bar(names, values)
            ax.set_title('Statistical Metrics')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        # Diversity metrics
        if 'diversity' in metrics:
            ax = axes[0, 1]
            div_metrics = metrics['diversity']
            names = list(div_metrics.keys())
            values = list(div_metrics.values())
            ax.bar(names, values, color='orange')
            ax.set_title('Diversity Metrics')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        # Combined score
        if 'combined_score' in metrics:
            ax = axes[1, 0]
            ax.bar(['Combined Score'], [metrics['combined_score']], color='green')
            ax.set_title('Overall Combined Score')
            ax.set_ylabel('Score')
        
        # Coverage
        if 'coverage' in metrics:
            ax = axes[1, 1]
            ax.bar(['Coverage'], [metrics['coverage']], color='purple')
            ax.set_title('Coverage')
            ax.set_ylabel('Score')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'metrics_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics visualization saved to {output_path}")
    
    def create_event_sequence_gifs(self, decoded_sequences, num_gifs=10):
        """
        Create animated GIFs showing events on the football field photo.

        Each GIF spans 60 seconds total, with one frame per event showing the
        event type label and a dot at its (x, y) position, colored by team.

        Args:
            decoded_sequences: np.ndarray of shape (N, seq_len, 128)
            num_gifs: How many sequences to visualize
        """
        from extract_event_sequences import build_event_mapping, get_event_name

        if decoded_sequences is None or len(decoded_sequences) == 0:
            logger.warning("No decoded sequences for GIF visualization")
            return

        if not os.path.exists(FIELD_IMAGE_PATH):
            logger.warning(f"Field image not found at {FIELD_IMAGE_PATH}, skipping GIFs")
            return

        field_img = plt.imread(FIELD_IMAGE_PATH)
        _, _, _, boundaries = build_event_mapping()

        n = min(num_gifs, len(decoded_sequences))
        indices = np.random.choice(len(decoded_sequences), n, replace=False)

        gif_dir = os.path.join(self.output_dir, 'sequence_gifs')
        os.makedirs(gif_dir, exist_ok=True)

        for seq_idx in indices:
            seq = decoded_sequences[seq_idx]  # (seq_len, 128)
            seq_len = len(seq)
            interval_ms = 60_000 / seq_len

            event_names = []
            xs, ys, teams = [], [], []
            for event_vec in seq:
                name, _, _ = get_event_name(event_vec[0], boundaries)
                event_names.append(name)
                xs.append(event_vec[2] * 120)
                ys.append(event_vec[3] * 80)
                teams.append(0 if event_vec[12] < 0.75 else 1)

            # Convert field coords to pixel coords
            pxs = [FIELD_BOUNDS['left'] + (x / 120) * (FIELD_BOUNDS['right'] - FIELD_BOUNDS['left']) for x in xs]
            pys = [FIELD_BOUNDS['top'] + (y / 80) * (FIELD_BOUNDS['bottom'] - FIELD_BOUNDS['top']) for y in ys]

            fig, ax = plt.subplots(figsize=(10, 6.7))
            ax.imshow(field_img)
            ax.axis('off')

            dot, = ax.plot([], [], 'o', markersize=14, markeredgecolor='white', markeredgewidth=1.5)
            label = ax.text(0, 0, '', fontsize=10, fontweight='bold',
                            ha='center', va='bottom', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', alpha=0.85, edgecolor='none'))
            counter = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                              fontsize=9, va='top', color='white',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

            def update(frame):
                color = TEAM_COLORS[teams[frame]]
                dot.set_data([pxs[frame]], [pys[frame]])
                dot.set_color(color)
                label.set_position((pxs[frame], pys[frame] - 8))
                label.set_text(event_names[frame])
                label.get_bbox_patch().set_facecolor(color)
                counter.set_text(f'Event {frame + 1}/{seq_len}')
                return dot, label, counter

            anim = FuncAnimation(fig, update, frames=seq_len,
                                 interval=interval_ms, blit=True)
            out_path = os.path.join(gif_dir, f'event_sequence_{seq_idx:04d}.gif')
            anim.save(out_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
            plt.close(fig)
            logger.info(f"Saved GIF: {out_path}")

        logger.info(f"Created {n} event sequence GIFs in {gif_dir}")

    def visualize_all(self):
        """Create all visualizations from saved data."""
        logger.info("="*60)
        logger.info("CREATING ALL VISUALIZATIONS")
        logger.info("="*60)
        
        # Load generated samples
        decoded_sequences = None
        samples_path = os.path.join(self.output_dir, 'generated_samples.pkl')
        if os.path.exists(samples_path):
            with open(samples_path, 'rb') as f:
                data = pickle.load(f)
            
            samples = data.get('samples', data.get('generated'))
            trajectory = data.get('trajectory', [])
            decoded_sequences = data.get('decoded_generated')
            
            if samples is not None:
                self.visualize_generated_samples(samples[:50])
                self.visualize_embedding_space(samples)
            
            if trajectory:
                self.visualize_diffusion_trajectory(trajectory)
        
        # Load and visualize metrics
        metrics_path = os.path.join(self.output_dir, 'evaluation_results.json')
        if os.path.exists(metrics_path):
            self.visualize_metrics(metrics_path)
        
        # Load cached training samples
        cache_dir = os.path.join(self.args.cache_dir, 'diffusion')
        if os.path.exists(cache_dir):
            cache_files = sorted(Path(cache_dir).glob('cached_samples_*.pkl'))
            if cache_files:
                latest_cache = cache_files[-1]
                with open(latest_cache, 'rb') as f:
                    cached = pickle.load(f)
                
                if 'final' in cached:
                    self.visualize_generated_samples(cached['final'][:10])
        
        # Event sequence GIFs from decoded samples
        num_gifs = getattr(self.args, 'num_gifs', 10)
        if decoded_sequences is not None:
            self.create_event_sequence_gifs(decoded_sequences, num_gifs)
        
        logger.info("="*60)
        logger.info("ALL VISUALIZATIONS COMPLETE")
        logger.info("="*60)
