"""
U-Net architecture for diffusion models (DDPM paper).
Uses 1D convolutions for sequence data.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """
    Residual block with 1D convolutions and time conditioning.
    Similar to DDPM but adapted for 1D sequences.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        """
        Args:
            x: (batch, channels, seq_len)
            t_emb: (batch, time_emb_dim)
        """
        h = self.block1(x)
        
        # Add time embedding
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None]  # Broadcast over sequence length
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock1D(nn.Module):
    """
    Self-attention block for 1D sequences.
    """
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        """
        batch, channels, seq_len = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        head_dim = channels // self.num_heads
        q = q.view(batch, self.num_heads, head_dim, seq_len).transpose(2, 3)
        k = k.view(batch, self.num_heads, head_dim, seq_len).transpose(2, 3)
        v = v.view(batch, self.num_heads, head_dim, seq_len).transpose(2, 3)
        
        # Attention
        scale = head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = h.transpose(2, 3).contiguous().view(batch, channels, seq_len)
        h = self.proj(h)
        
        return x + h


class Downsample1D(nn.Module):
    """Downsample by factor of 2 using strided convolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample by factor of 2 using transposed convolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    """
    U-Net for 1D sequences (DDPM-style).
    
    Architecture:
        - Encoder: Downsample with residual blocks
        - Bottleneck: Residual + attention blocks
        - Decoder: Upsample with residual blocks + skip connections
    """
    
    def __init__(
        self,
        input_dim=32,
        model_channels=128,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        max_seq_len=50
    ):
        """
        Args:
            input_dim: Input dimension (embedding dim)
            model_channels: Base channel count
            channel_multipliers: Channel multiplier for each resolution
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply attention at
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.model_channels = model_channels
        self.max_seq_len = max_seq_len
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection (from embedding space to channels)
        self.input_proj = nn.Conv1d(input_dim, model_channels, kernel_size=1)
        
        # Calculate resolutions
        self.num_resolutions = len(channel_multipliers)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        
        ch = model_channels
        input_ch = ch
        
        for level in range(self.num_resolutions):
            ch_mult = channel_multipliers[level]
            out_ch = model_channels * ch_mult
            
            # Residual blocks
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock1D(input_ch, out_ch, time_emb_dim, dropout))
                
                # Add attention at specified resolutions
                current_res = max_seq_len // (2 ** level)
                if current_res in attention_resolutions:
                    attns.append(AttentionBlock1D(out_ch))
                else:
                    attns.append(nn.Identity())
                
                input_ch = out_ch
            
            self.encoder.append(blocks)
            self.encoder_attentions.append(attns)
            
            # Downsample (except last level)
            if level != self.num_resolutions - 1:
                self.encoder.append(nn.ModuleList([Downsample1D(out_ch)]))
                self.encoder_attentions.append(nn.ModuleList([nn.Identity()]))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock1D(out_ch, out_ch, time_emb_dim, dropout),
            AttentionBlock1D(out_ch),
            ResidualBlock1D(out_ch, out_ch, time_emb_dim, dropout)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        
        for level in reversed(range(self.num_resolutions)):
            ch_mult = channel_multipliers[level]
            out_ch = model_channels * ch_mult
            
            # Residual blocks
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            
            for i in range(num_res_blocks + 1):  # +1 for skip connection
                # First block receives skip connection
                skip_ch = model_channels * channel_multipliers[level] if i == 0 else 0
                blocks.append(ResidualBlock1D(input_ch + skip_ch, out_ch, time_emb_dim, dropout))
                
                current_res = max_seq_len // (2 ** level)
                if current_res in attention_resolutions:
                    attns.append(AttentionBlock1D(out_ch))
                else:
                    attns.append(nn.Identity())
                
                input_ch = out_ch
            
            self.decoder.append(blocks)
            self.decoder_attentions.append(attns)
            
            # Upsample (except first level when going backwards)
            if level != 0:
                self.decoder.append(nn.ModuleList([Upsample1D(out_ch)]))
                self.decoder_attentions.append(nn.ModuleList([nn.Identity()]))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, input_dim, kernel_size=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, t, mask=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) - noisy sequences
            t: (batch,) - timesteps
            mask: Optional mask (not used in U-Net, kept for compatibility)
        
        Returns:
            (batch, seq_len, input_dim) - predicted noise
        """
        batch_size, seq_len, _ = x.shape
        
        # Transpose to (batch, channels, seq_len) for convolutions
        x = x.transpose(1, 2)
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for i, (blocks, attns) in enumerate(zip(self.encoder, self.encoder_attentions)):
            for block, attn in zip(blocks, attns):
                if isinstance(block, (Downsample1D,)):
                    h = block(h)
                else:
                    h = block(h, t_emb)
                    h = attn(h)
                    skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock1D):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Decoder with skip connections
        for i, (blocks, attns) in enumerate(zip(self.decoder, self.decoder_attentions)):
            for j, (block, attn) in enumerate(zip(blocks, attns)):
                if isinstance(block, (Upsample1D,)):
                    h = block(h)
                else:
                    # Add skip connection for first block at each level
                    if j == 0 and len(skip_connections) > 0:
                        skip = skip_connections.pop()
                        # Interpolate skip if needed
                        if skip.shape[-1] != h.shape[-1]:
                            skip = nn.functional.interpolate(
                                skip, size=h.shape[-1], mode='linear', align_corners=False
                            )
                        h = torch.cat([h, skip], dim=1)
                    
                    h = block(h, t_emb)
                    h = attn(h)
        
        # Output projection
        h = self.output_proj(h)
        
        # Transpose back to (batch, seq_len, channels)
        h = h.transpose(1, 2)
        
        return h


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the U-Net
    model = UNet1D(
        input_dim=32,
        model_channels=128,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(8, 16),
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
    print("U-Net test passed!")
