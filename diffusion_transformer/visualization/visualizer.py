"""
Visualization module for diffusion transformer outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
from pathlib import Path


class Visualizer:
    """Visualizer for diffusion transformer results."""
    
    def __init__(self, args):
        """
        Initialize visualizer.
        
        Args:
            args: Argument namespace with configuration
        """
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def visualize_diffusion_trajectory(self, trajectory, decoded_events=None):
        """
        Visualize the diffusion process from noise to clean samples.
        
        Args:
            trajectory: List of dicts with 'timestep' and 'samples'
            decoded_events: Optional decoded events for visualization
        """
        if not trajectory:
            print("No trajectory data to visualize")
            return
        
        print("Visualizing diffusion trajectory...")
        
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
        
        print(f"Trajectory visualization saved to {output_path}")
        
        # Create animated trajectory
        self.create_trajectory_animation(trajectory)
    
    def create_trajectory_animation(self, trajectory):
        """Create animated visualization of diffusion trajectory."""
        print("Creating trajectory animation...")
        
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
        
        print(f"Animation saved to {output_path}")
    
    def visualize_generated_samples(self, samples):
        """
        Visualize generated samples as heatmaps.
        
        Args:
            samples: Generated samples (N, seq_len, embedding_dim)
        """
        print("Visualizing generated samples...")
        
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
        
        print(f"Sample grid saved to {output_path}")
    
    def visualize_embedding_space(self, samples, real_samples=None):
        """
        Visualize embedding space using t-SNE and PCA.
        
        Args:
            samples: Generated samples
            real_samples: Optional real samples for comparison
        """
        print("Visualizing embedding space...")
        
        # Flatten sequences
        samples_flat = samples.reshape(samples.shape[0], -1)
        
        # Apply PCA first for dimensionality reduction
        pca = PCA(n_components=50)
        samples_pca = pca.fit_transform(samples_flat)
        
        # Apply t-SNE
        print("Computing t-SNE (this may take a while)...")
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
        
        print(f"Embedding space visualization saved to {output_path}")
    
    def visualize_on_pitch(self, events, output_name='pitch_visualization.png'):
        """
        Visualize event sequences on a football pitch.
        
        Args:
            events: List of events with coordinates (x, y)
            output_name: Output filename
        """
        print("Visualizing events on pitch...")
        
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
        
        print(f"Pitch visualization saved to {output_path}")
    
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
        
        print("Visualizing metrics...")
        
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
        
        print(f"Metrics visualization saved to {output_path}")
    
    def visualize_all(self):
        """Create all visualizations from saved data."""
        print("\n=== Creating All Visualizations ===\n")
        
        # Load generated samples
        samples_path = os.path.join(self.output_dir, 'generated_samples.pkl')
        if os.path.exists(samples_path):
            with open(samples_path, 'rb') as f:
                data = pickle.load(f)
            
            samples = data.get('samples', data.get('generated'))
            trajectory = data.get('trajectory', [])
            
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
        
        print("\n=== All Visualizations Complete ===")
