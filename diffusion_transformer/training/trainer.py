"""
Training module for Diffusion Transformer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from diffusion_transformer.models.diffusion import DiffusionProcess
from diffusion_transformer.data.dataset import create_dataloaders


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
        print(f"Model parameters: {num_params:,}")
        
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
        self.scaler = GradScaler() if args.device == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if provided
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)
        
        # Cache for intermediate samples
        self.cached_samples = []
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.num_epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.args.eval_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{self.args.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            else:
                print(f"Epoch {epoch+1}/{self.args.num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint()
            
            # Cache intermediate samples
            if (epoch + 1) % self.args.cache_interval == 0:
                self.cache_samples(epoch)
        
        print("\nTraining completed!")
        self.save_checkpoint(is_best=False, filename=f"{self.args.model_type}_final_model.pt")
    
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
                with autocast():
                    noise_pred = self.model(x_t, t)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noise_pred = self.model(x_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
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
        
        return total_loss / len(self.val_loader)
    
    @torch.no_grad()
    def cache_samples(self, epoch):
        """Cache intermediate diffusion samples for visualization."""
        self.model.eval()
        
        print(f"Caching {self.args.num_cached_samples} intermediate samples...")
        
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

        print(f"Cached samples saved to {cache_path}")
    
    def save_checkpoint(self, is_best=False, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"{self.args.model_type}_checkpoint_epoch_{self.current_epoch+1}.pt"
        
        checkpoint_path = os.path.join(self.args.models_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_type': self.args.model_type,  # Save model type
            'args': vars(self.args)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.args.models_dir, f"{self.args.model_type}_best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
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
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.current_epoch}")
