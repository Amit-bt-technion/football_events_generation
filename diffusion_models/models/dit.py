"""
Diffusion Transformer (DiT) model architecture.
Transformer-based denoising model for latent diffusion.
"""

import torch
import torch.nn as nn
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: Timestep tensor of shape (batch_size,)
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep.
    Uses timestep embedding to modulate scale and shift.
    """
    
    def __init__(self, dim, conditioning_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(conditioning_dim, dim * 2)
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)
    
    def forward(self, x, conditioning):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            conditioning: Conditioning tensor (batch, conditioning_dim)
        """
        scale_shift = self.scale_shift(conditioning)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer normalization.
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, conditioning_dim=None):
        super().__init__()
        
        if conditioning_dim is None:
            conditioning_dim = dim
        
        self.norm1 = AdaptiveLayerNorm(dim, conditioning_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = AdaptiveLayerNorm(dim, conditioning_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, conditioning, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            conditioning: Conditioning tensor (batch, conditioning_dim)
            mask: Optional attention mask
        """
        # Self-attention with residual
        attn_input = self.norm1(x, conditioning)
        attn_output, _ = self.attn(
            attn_input, attn_input, attn_input,
            key_padding_mask=mask,
            need_weights=False
        )
        x = x + attn_output
        
        # MLP with residual
        mlp_input = self.norm2(x, conditioning)
        x = x + self.mlp(mlp_input)
        
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for event sequence generation.
    Takes noisy latent sequences and timestep, predicts noise.
    """
    
    def __init__(
        self,
        input_dim=32,
        model_dim=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=50,
        conditioning_dim=None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        if conditioning_dim is None:
            conditioning_dim = model_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, model_dim) * 0.02
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_dim),
            nn.Linear(model_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                conditioning_dim=conditioning_dim
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(model_dim)
        
        # Output projection (predicts noise)
        self.output_proj = nn.Linear(model_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, t, mask=None):
        """
        Forward pass.
        
        Args:
            x: Noisy input sequences (batch, seq_len, input_dim)
            t: Timesteps (batch,)
            mask: Optional mask for attention (batch, seq_len)
                 True for positions to mask out
        
        Returns:
            Predicted noise (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Get timestep conditioning
        t_emb = self.time_embed(t)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, mask)
        
        # Final normalization and projection
        x = self.final_norm(x)
        noise_pred = self.output_proj(x)
        
        return noise_pred
    
    def forward_with_cfg(self, x, t, mask=None, condition=None, cfg_scale=1.0):
        """
        Forward pass with classifier-free guidance.
        For future conditioning tasks.
        
        Args:
            x: Noisy input
            t: Timesteps
            mask: Attention mask
            condition: Conditioning information (can be None)
            cfg_scale: Guidance scale (1.0 = no guidance)
        """
        if condition is None or cfg_scale == 1.0:
            return self.forward(x, t, mask)
        
        # TODO: Implement CFG for conditional generation
        # This is a placeholder for future conditioning
        return self.forward(x, t, mask)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = DiffusionTransformer(
        input_dim=32,
        model_dim=512,
        num_layers=8,
        num_heads=8,
        max_seq_len=50
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("Model test passed!")
