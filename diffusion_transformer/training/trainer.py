"""
Training module for Diffusion Transformer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer for Diffusion Transformer."""
    
    def __init__(self, args, train_loader: DataLoader, val_loader: DataLoader):
        """
        Initialize trainer.
        
        Args:
            args: Argument namespace with configuration
        """
        self.args = args
        self.device = torch.device(args.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create model based on type
        print(f"Creating {args.model_type.upper()} model...")
        
        if args.model_type == "dit":
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
        
        elif args.model_type == "unet":
            from diffusion_transformer.models.unet import UNet1D
            
            # Parse channel multipliers
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
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info(f"Model on CUDA: {next(self.model.parameters()).is_cuda}")
        logger.info(f"Model parameters: {num_params:,}")
        
        # Create diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=args.num_timesteps,
            schedule_type=args.noise_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create learning rate scheduler with warmup
        total_steps = len(self.train_loader) * args.num_epochs
        warmup_steps = len(self.train_loader) * args.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if args.device == "cuda" else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epoch_times = []
        self.grad_norms = []
        
        # Load checkpoint if provided
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)
        
        # Cache for intermediate samples
        self.cached_samples = []
    
    def train(self):
        """Main training loop."""
        import time
        
        print(f"\nStarting training for {self.args.num_epochs} epochs...")
        print(f"Current best validation loss: {self.best_val_loss:.6f}")
        print(f"Validation will run every {self.args.eval_every} epochs")
        print(f"Checkpoints will be saved every {self.args.save_every} epochs")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Track learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            
            # Track epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Validate (if available)
            if (epoch + 1) % self.args.eval_every == 0:
                val_loss = self.validate()
                if val_loss is None:
                    print(
                        f"Epoch {epoch+1}/{self.args.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: N/A (empty val_loader), "
                        f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s"
                    )
                    print("  → Skipping best-model saving because validation set is empty.")
                else:
                    self.val_losses.append(val_loss)
                    print(f"Epoch {epoch+1}/{self.args.num_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        print(f"  → New best validation loss: {val_loss:.6f} (previous: {self.best_val_loss:.6f})")
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        print(f"  → Best model saved!")
                    else:
                        print(f"  → Validation loss did not improve (best: {self.best_val_loss:.6f})")
            else:
                print(f"Epoch {epoch+1}/{self.args.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                logger.debug(f"Saving regular checkpoint at epoch {epoch + 1}")
                self.save_checkpoint()
            
            # Cache intermediate samples
            if (epoch + 1) % self.args.cache_interval == 0:
                self.cache_samples(epoch)
        
        print("\nTraining completed!")
        self.save_checkpoint(is_best=False, filename=f"{self.args.model_type}_final_model.pt")
        
        # Final training visualization
        self.plot_training_progress()
        self.save_training_metrics()
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", disable=not self.args.verbose)

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x_0 = batch.to(self.device)  # Clean sequences

            # Sample random timesteps
            batch_size = x_0.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps,
                (batch_size,),
                device=self.device
            ).long()

            # Sample noise
            noise = torch.randn_like(x_0)
            
            # Forward diffusion (add noise)
            x_t = self.diffusion.q_sample(x_0, t, noise)
            
            # Predict noise
            if self.scaler is not None:
                with autocast('cuda'):
                    noise_pred = self.model(x_t, t)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noise_pred = self.model(x_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            
            # Track gradient norm (sample every 100 batches to avoid overhead)
            if batch_idx % 100 == 0:
                self.grad_norms.append(float(grad_norm))
            
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        # Handle empty validation loader gracefully
        if self.val_loader is None or len(self.val_loader) == 0:
            logger.warning("Validation skipped: val_loader is empty (0 batches).")
            return None

        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", disable=not self.args.verbose):
            x_0 = batch.to(self.device)
            
            batch_size = x_0.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps,
                (batch_size,),
                device=self.device
            ).long()
            
            noise = torch.randn_like(x_0)
            x_t = self.diffusion.q_sample(x_0, t, noise)
            
            noise_pred = self.model(x_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()

        # len(self.val_loader) is guaranteed > 0 here due to the early return above
        return total_loss / len(self.val_loader)
    
    @torch.no_grad()
    def cache_samples(self, epoch):
        """Cache intermediate diffusion samples for visualization."""
        self.model.eval()
        
        logger.info(f"Caching {self.args.num_cached_samples} intermediate samples...")

        # Generate samples at various stages of denoising
        shape = (self.args.num_cached_samples, self.args.sequence_length, self.args.embedding_dim)
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Select timesteps to cache
        cache_timesteps = [
            0,
            self.diffusion.num_timesteps // 4,
            self.diffusion.num_timesteps // 2,
            3 * self.diffusion.num_timesteps // 4,
            self.diffusion.num_timesteps - 1
        ]
        
        samples_at_timesteps = {}
        
        for t_val in cache_timesteps:
            t = torch.full((self.args.num_cached_samples,), t_val, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t)
            x_0_pred = self.diffusion.predict_start_from_noise(x, t, noise_pred)
            samples_at_timesteps[t_val] = x_0_pred.cpu().numpy()
        
        # Also generate fully denoised samples
        final_samples = self.diffusion.p_sample_loop(
            self.model,
            shape,
            progress=False
        )
        samples_at_timesteps['final'] = final_samples.cpu().numpy()
        
        # Save to cache
        cache_path = os.path.join(
            self.args.cache_dir,
            "diffusion",
            f"{self.args.model_type}_cached_samples_epoch_{epoch+1}.pkl"
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(samples_at_timesteps, f)

        logger.debug(f"Cached samples saved to {cache_path}")

    def save_checkpoint(self, is_best=False, filename=None):
        """Save model checkpoint."""
        try:
            if filename is None:
                filename = f"{self.args.model_type}_checkpoint_epoch_{self.current_epoch+1}.pt"
            
            # Ensure models directory exists
            os.makedirs(self.args.models_dir, exist_ok=True)
            logger.debug(f"Models directory: {self.args.models_dir} (exists: {os.path.exists(self.args.models_dir)})")
            
            checkpoint_path = os.path.join(self.args.models_dir, filename)
            logger.debug(f"Saving checkpoint to: {checkpoint_path}")
            
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'model_type': self.args.model_type,  # Save model type
                'args': vars(self.args),
                # Save training metrics
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'epoch_times': self.epoch_times,
                'grad_norms': self.grad_norms,
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, str(checkpoint_path))
            
            # Verify the file was actually saved
            if os.path.exists(checkpoint_path):
                file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                print(f"✓ Checkpoint saved to {checkpoint_path} ({file_size:.2f} MB)")
            else:
                logger.error(f"✗ Failed to save checkpoint - file does not exist: {checkpoint_path}")
                return
            
            if is_best:
                best_path = os.path.join(self.args.models_dir, f"{self.args.model_type}_best_model.pt")
                torch.save(checkpoint, best_path)
                
                # Verify best model was saved
                if os.path.exists(best_path):
                    file_size = os.path.getsize(best_path) / (1024 * 1024)  # MB
                    logger.info(f"✓ Best model checkpoint saved to {best_path} ({file_size:.2f} MB)")
                else:
                    logger.error(f"✗ Failed to save best model - file does not exist: {best_path}")
                    
        except Exception as e:
            logger.error(f"✗ Error saving checkpoint: {e}", exc_info=True)
            print(f"✗ ERROR: Failed to save checkpoint: {e}")
            print(f"   Models directory: {self.args.models_dir}")
            print(f"   Directory exists: {os.path.exists(self.args.models_dir)}")
            print(f"   Directory writable: {os.access(self.args.models_dir, os.W_OK) if os.path.exists(self.args.models_dir) else 'N/A'}")
            raise
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load training metrics if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'learning_rates' in checkpoint:
            self.learning_rates = checkpoint['learning_rates']
        if 'epoch_times' in checkpoint:
            self.epoch_times = checkpoint['epoch_times']
        if 'grad_norms' in checkpoint:
            self.grad_norms = checkpoint['grad_norms']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def plot_training_progress(self):
        """Plot and save training progress visualizations."""
        import matplotlib.pyplot as plt
        
        if len(self.train_losses) == 0:
            return
        
        # Create output directory for plots
        plot_dir = os.path.join(self.args.output_dir, 'training_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Determine grid size based on available metrics
        n_plots = 2  # At minimum: loss and learning rate
        if len(self.grad_norms) > 0:
            n_plots += 1
        if len(self.epoch_times) > 0:
            n_plots += 1
        
        fig = plt.figure(figsize=(16, 4 * ((n_plots + 1) // 2)))
        plot_idx = 1
        
        # 1. Training and Validation Loss
        ax1 = plt.subplot(2, 2, plot_idx)
        plot_idx += 1
        
        epochs = list(range(1, len(self.train_losses) + 1))
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        
        if len(self.val_losses) > 0:
            # Val losses are only recorded every eval_every epochs
            val_epochs = list(range(self.args.eval_every, 
                                   len(self.train_losses) + 1, 
                                   self.args.eval_every))[:len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, 'r-', label='Val Loss', 
                    linewidth=2, alpha=0.8, marker='o', markersize=5)
            
            # Mark best validation loss
            if self.best_val_loss < float('inf'):
                best_epoch_idx = np.argmin(self.val_losses)
                best_epoch = val_epochs[best_epoch_idx]
                ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                           label=f'Best Val (epoch {best_epoch})')
                ax1.scatter([best_epoch], [self.best_val_loss], 
                           color='g', s=100, zorder=5, marker='*')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale often better for loss
        
        # 2. Learning Rate Schedule
        ax2 = plt.subplot(2, 2, plot_idx)
        plot_idx += 1
        
        if len(self.learning_rates) > 0:
            ax2.plot(epochs, self.learning_rates, 'g-', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Mark warmup period
            if self.args.warmup_epochs > 0:
                ax2.axvline(x=self.args.warmup_epochs, color='orange', 
                           linestyle='--', alpha=0.5, label=f'Warmup end')
                ax2.legend(loc='best', fontsize=10)
        
        # 3. Gradient Norms (if tracked)
        if len(self.grad_norms) > 0:
            ax3 = plt.subplot(2, 2, plot_idx)
            plot_idx += 1
            
            # Smooth gradient norms (they can be noisy)
            window_size = min(50, len(self.grad_norms) // 10)
            if window_size > 1:
                grad_norms_smooth = np.convolve(self.grad_norms, 
                                                np.ones(window_size)/window_size, 
                                                mode='valid')
                ax3.plot(grad_norms_smooth, 'purple', linewidth=2, alpha=0.8, label='Smoothed')
            
            ax3.plot(self.grad_norms, 'purple', linewidth=0.5, alpha=0.3, label='Raw')
            ax3.axhline(y=self.args.grad_clip, color='r', linestyle='--', 
                       alpha=0.5, label=f'Clip threshold ({self.args.grad_clip})')
            
            ax3.set_xlabel('Step (sampled)', fontsize=12)
            ax3.set_ylabel('Gradient Norm', fontsize=12)
            ax3.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. Epoch Times (if tracked)
        if len(self.epoch_times) > 0:
            ax4 = plt.subplot(2, 2, plot_idx)
            plot_idx += 1
            
            ax4.plot(epochs, self.epoch_times, 'orange', linewidth=2, alpha=0.8, marker='o', markersize=4)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Time (seconds)', fontsize=12)
            ax4.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(self.epoch_times)
            ax4.axhline(y=avg_time, color='red', linestyle='--', 
                       alpha=0.5, label=f'Average: {avg_time:.1f}s')
            ax4.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plot_dir, f'{self.args.model_type}_training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress plot saved to {plot_path}")
        
        # Also create a simple loss-only plot for quick viewing
        self._plot_loss_curve(plot_dir)
    
    def _plot_loss_curve(self, plot_dir):
        """Create a simplified loss curve plot."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = list(range(1, len(self.train_losses) + 1))
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2.5, alpha=0.9)
        
        if len(self.val_losses) > 0:
            val_epochs = list(range(self.args.eval_every, 
                                   len(self.train_losses) + 1, 
                                   self.args.eval_every))[:len(self.val_losses)]
            ax.plot(val_epochs, self.val_losses, 'r-', label='Val Loss', 
                   linewidth=2.5, alpha=0.9, marker='o', markersize=6)
            
            # Mark best
            if self.best_val_loss < float('inf'):
                best_epoch_idx = np.argmin(self.val_losses)
                best_epoch = val_epochs[best_epoch_idx]
                ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                          label=f'Best: epoch {best_epoch}, loss {self.best_val_loss:.4f}')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=14, fontweight='bold')
        ax.set_title(f'{self.args.model_type.upper()} Training Loss Curve', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'{self.args.model_type}_loss_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss curve plot saved to {plot_path}")
    
    def save_training_metrics(self):
        """Save training metrics to JSON and CSV files."""
        import json
        import pandas as pd
        
        # Create metrics directory
        metrics_dir = os.path.join(self.args.output_dir, 'training_metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Prepare metrics dictionary
        metrics = {
            'model_type': self.args.model_type,
            'total_epochs': len(self.train_losses),
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.train_losses[-1]) if self.train_losses else None,
            'final_val_loss': float(self.val_losses[-1]) if self.val_losses else None,
            'total_training_time': float(sum(self.epoch_times)) if self.epoch_times else None,
            'avg_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else None,
            'avg_grad_norm': float(np.mean(self.grad_norms)) if self.grad_norms else None,
            'max_grad_norm': float(np.max(self.grad_norms)) if self.grad_norms else None,
        }
        
        # Save summary as JSON
        json_path = os.path.join(metrics_dir, f'{self.args.model_type}_training_summary.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training summary saved to {json_path}")
        
        # Save detailed metrics as CSV
        epochs = list(range(1, len(self.train_losses) + 1))
        
        df_data = {
            'epoch': epochs,
            'train_loss': self.train_losses,
        }
        
        if len(self.learning_rates) > 0:
            df_data['learning_rate'] = self.learning_rates
        
        if len(self.epoch_times) > 0:
            df_data['epoch_time_s'] = self.epoch_times
        
        # Add validation losses (sparse)
        if len(self.val_losses) > 0:
            val_loss_full = [None] * len(epochs)
            val_epochs = list(range(self.args.eval_every - 1,  # 0-indexed
                                   len(epochs), 
                                   self.args.eval_every))
            for i, val_epoch_idx in enumerate(val_epochs):
                if i < len(self.val_losses) and val_epoch_idx < len(val_loss_full):
                    val_loss_full[val_epoch_idx] = self.val_losses[i]
            df_data['val_loss'] = val_loss_full
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(metrics_dir, f'{self.args.model_type}_training_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Detailed metrics saved to {csv_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model Type: {metrics['model_type'].upper()}")
        print(f"Total Epochs: {metrics['total_epochs']}")
        print(f"Best Val Loss: {metrics['best_val_loss']:.6f}")
        print(f"Final Train Loss: {metrics['final_train_loss']:.6f}")
        if metrics['final_val_loss']:
            print(f"Final Val Loss: {metrics['final_val_loss']:.6f}")
        if metrics['total_training_time']:
            print(f"Total Training Time: {metrics['total_training_time']/3600:.2f} hours")
            print(f"Avg Time per Epoch: {metrics['avg_epoch_time']:.1f}s")
        if metrics['avg_grad_norm']:
            print(f"Avg Gradient Norm: {metrics['avg_grad_norm']:.4f}")
        print("="*60 + "\n")