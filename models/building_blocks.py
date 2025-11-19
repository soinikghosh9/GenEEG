"""
Building Blocks for Neural Network Architectures

This module contains reusable building blocks used across different models:
- TemporalSelfAttention1D: Self-attention mechanism for 1D temporal sequences
- CrossAttention: Cross-attention for conditioning in diffusion models
- AdaGN: Adaptive Group Normalization for conditional generation
- ResNetBlock: Residual block with optional time/feature/class conditioning
- SpatialChannelAttention: Combined spatial and channel attention
- ChannelAttention: Channel attention mechanism
- EnhancedTemporalAttention: Enhanced temporal attention with better numerical stability
- Downsample1D/Upsample1D: Downsampling and upsampling layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention1D(nn.Module):
    """
    Self-attention mechanism for 1D temporal sequences with numerical stability.
    
    Uses multi-head attention with scaled dot-product attention.
    Uses GroupNorm instead of LayerNorm for better stability with 1D convolutions.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: 8)
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use GroupNorm instead of LayerNorm for better stability
        num_groups = min(32, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, n = x.shape
        # GroupNorm operates on (B, C, N) directly
        x_norm = self.norm(x)
        
        # Check for NaN after normalization
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            print("[WARNING] NaN/Inf in TemporalSelfAttention1D after norm. Using input.")
            x_norm = x
        
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2), qkv)
        
        # Use PyTorch's efficient implementation with better numerical stability
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, n)
        
        result = self.to_out(out) + x
        
        # Final NaN check
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("[WARNING] NaN/Inf in TemporalSelfAttention1D output. Using input.")
            return x
        
        return result


class CrossAttention(nn.Module):
    """
    Cross-attention module for conditioning in latent diffusion models.
    
    Enables the model to attend to external context (e.g., class labels, features)
    while processing the main input.
    
    Args:
        query_dim: Dimension of query vectors
        context_dim: Dimension of context vectors
        n_heads: Number of attention heads (default: 8)
        head_dim: Dimension per head (default: 64)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, query_dim, context_dim, n_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5
        self.heads = n_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context):
        """
        Args:
            x: Query tensor of shape (batch, seq_len, query_dim)
            context: Context tensor of shape (batch, context_seq_len, context_dim)
        
        Returns:
            Output tensor of shape (batch, seq_len, query_dim)
        """
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.view(t.shape[0], -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2), (q, k, v))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = scores.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization for conditional generation.
    
    Applies group normalization and then modulates with learned scale/shift
    parameters conditioned on external embeddings (e.g., class labels).
    
    Args:
        num_groups: Number of groups for GroupNorm
        num_channels: Number of channels
        cond_dim: Dimension of conditioning embedding
        eps: Epsilon for numerical stability (default: 1e-4)
    """
    def __init__(self, num_groups, num_channels, cond_dim, eps=1e-4):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_proj = nn.Linear(cond_dim, 2 * num_channels)
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=self.eps, affine=False)

    def forward(self, x, cond_embedding):
        """
        Args:
            x: Input tensor of shape (batch, channels, length)
            cond_embedding: Conditioning embedding of shape (batch, cond_dim)
        
        Returns:
            Normalized and modulated tensor
        """
        x_norm = self.norm(x)
        scale_shift = self.cond_proj(cond_embedding)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x_norm * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class ResNetBlock(nn.Module):
    """
    Residual block with optional time/feature/class conditioning and improved stability.
    
    Features:
    - Optional time embedding conditioning (for diffusion models)
    - Optional feature embedding conditioning (for VAE)
    - Optional class conditioning via AdaGN
    - Optional self-attention
    - Optional upsampling/downsampling
    - Improved normalization strategy
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embedding (None to disable)
        feature_cond_dim: Dimension of feature conditioning (None to disable)
        dropout: Dropout probability (default: 0.1)
        up: Enable upsampling (default: False)
        down: Enable downsampling (default: False)
        use_attention: Enable self-attention (default: False)
        num_groups: Number of groups for GroupNorm (default: 32)
        use_adagn: Use Adaptive GroupNorm for class conditioning (default: False)
        class_cond_dim: Dimension of class conditioning embedding (required if use_adagn=True)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None, feature_cond_dim=None,
                 dropout=0.1, up=False, down=False, use_attention=False, 
                 num_groups=32, use_adagn=False, class_cond_dim=None):
        super().__init__()
        self.up, self.down, self.use_adagn = up, down, use_adagn
        
        # Adjust num_groups to avoid division issues with small channel counts
        actual_num_groups = min(num_groups, in_channels, out_channels)
        # Ensure channels are divisible by num_groups
        while in_channels % actual_num_groups != 0 and actual_num_groups > 1:
            actual_num_groups -= 1
        
        # Choose normalization type based on use_adagn
        norm1_class = AdaGN if use_adagn else nn.GroupNorm
        norm2_class = AdaGN if use_adagn else nn.GroupNorm
        norm1_args = {'num_groups': actual_num_groups, 'num_channels': in_channels, 'eps': 1e-6}
        
        # Adjust groups for out_channels
        actual_num_groups_out = min(num_groups, out_channels)
        while out_channels % actual_num_groups_out != 0 and actual_num_groups_out > 1:
            actual_num_groups_out -= 1
        
        norm2_args = {'num_groups': actual_num_groups_out, 'num_channels': out_channels, 'eps': 1e-6}
        
        if use_adagn:
            norm1_args['cond_dim'] = class_cond_dim
            norm2_args['cond_dim'] = class_cond_dim

        self.norm1 = norm1_class(**norm1_args)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Conv1d(in_channels, out_channels, 3, padding=1))
        
        # Time embedding projection (for diffusion models)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels)) if time_emb_dim else None
        
        # Feature embedding projection (for VAE conditioning)
        self.feature_mlp = nn.Sequential(nn.SiLU(), nn.Linear(feature_cond_dim, out_channels)) if feature_cond_dim else None
        
        self.norm2 = norm2_class(**norm2_args)
        self.block2 = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, 3, padding=1))
        
        # Residual connection
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Optional self-attention
        self.attn = TemporalSelfAttention1D(out_channels) if use_attention else nn.Identity()
        
        # Optional upsampling/downsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if self.up else nn.Identity()
        self.downsample_conv = nn.Conv1d(in_channels, in_channels, 4, 2, 1) if self.down else nn.Identity()
        
        # Initialize weights with smaller values for better stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with conservative but reasonable values"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                # Kaiming init is better for residual networks than orthogonal
                # Preserves variance through ReLU/SiLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Linear layers use Xavier with moderate gain
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # CRITICAL: Initialize shortcut to be VERY close to identity
        # This allows gradients to flow initially through the skip path
        if isinstance(self.shortcut, nn.Conv1d):
            # Use orthogonal with small gain for shortcuts
            nn.init.orthogonal_(self.shortcut.weight, gain=0.1)
            if self.shortcut.bias is not None:
                nn.init.zeros_(self.shortcut.bias)
    
    def forward(self, x, t_emb=None, f_emb=None, c_emb=None):
        """
        Args:
            x: Input tensor of shape (batch, in_channels, length)
            t_emb: Time embedding (batch, time_emb_dim) - optional
            f_emb: Feature embedding (batch, feature_cond_dim) - optional
            c_emb: Class embedding (batch, class_cond_dim) - optional, used with AdaGN
        
        Returns:
            Output tensor of shape (batch, out_channels, length')
        """
        # Input validation and AGGRESSIVE clamping
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf in ResNetBlock input. Clamping.")
            x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Clamp input to reasonable range BEFORE any operations
        x = torch.clamp(x, -5.0, 5.0)
        
        x_res = self.downsample_conv(x)
        
        # Aggressive clamp after downsample
        x_res = torch.clamp(x_res, -5.0, 5.0)
        
        # Check after downsample
        if torch.isnan(x_res).any() or torch.isinf(x_res).any():
            print(f"[WARNING] NaN/Inf after downsample in ResNetBlock. Clamping.")
            x_res = torch.nan_to_num(x_res, nan=0.0, posinf=3.0, neginf=-3.0)
            x_res = torch.clamp(x_res, -5.0, 5.0)
        
        # First normalization + convolution
        if self.use_adagn:
            h = self.block1(self.norm1(x_res, c_emb))
        else:
            norm1_out = self.norm1(x_res)
            # Clamp normalized output
            norm1_out = torch.clamp(norm1_out, -5.0, 5.0)
            if torch.isnan(norm1_out).any() or torch.isinf(norm1_out).any():
                print(f"[WARNING] NaN/Inf after norm1 in ResNetBlock. Clamping.")
                norm1_out = torch.nan_to_num(norm1_out, nan=0.0, posinf=3.0, neginf=-3.0)
                norm1_out = torch.clamp(norm1_out, -5.0, 5.0)
            h = self.block1(norm1_out)
        
        # Aggressive clamp after block1
        h = torch.clamp(h, -5.0, 5.0)
        
        # Check after block1
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[WARNING] NaN/Inf after block1 in ResNetBlock. Clamping.")
            h = torch.nan_to_num(h, nan=0.0, posinf=3.0, neginf=-3.0)
            h = torch.clamp(h, -5.0, 5.0)

        # Add time embedding conditioning
        if t_emb is not None and self.time_mlp:
            t_emb_proj = self.time_mlp(t_emb).unsqueeze(-1)
            t_emb_proj = torch.clamp(t_emb_proj, -3.0, 3.0)  # Clamp embeddings
            if torch.isnan(t_emb_proj).any() or torch.isinf(t_emb_proj).any():
                print(f"[WARNING] NaN/Inf in time embedding projection. Clamping.")
                t_emb_proj = torch.nan_to_num(t_emb_proj, nan=0.0, posinf=3.0, neginf=-3.0)
                t_emb_proj = torch.clamp(t_emb_proj, -3.0, 3.0)
            h = h + t_emb_proj
            h = torch.clamp(h, -5.0, 5.0)  # Clamp after addition
            
        # Add feature embedding conditioning
        if f_emb is not None and self.feature_mlp:
            # Ensure contiguity for cuBLAS
            feature_embedding = self.feature_mlp(f_emb.contiguous()).unsqueeze(-1)
            feature_embedding = torch.clamp(feature_embedding, -3.0, 3.0)  # Clamp embeddings
            if torch.isnan(feature_embedding).any() or torch.isinf(feature_embedding).any():
                print(f"[WARNING] NaN/Inf in feature embedding projection. Clamping.")
                feature_embedding = torch.nan_to_num(feature_embedding, nan=0.0, posinf=3.0, neginf=-3.0)
                feature_embedding = torch.clamp(feature_embedding, -3.0, 3.0)
            h = h + feature_embedding
            h = torch.clamp(h, -5.0, 5.0)  # Clamp after addition
        
        # Check after embedding additions
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[WARNING] NaN/Inf after embedding additions in ResNetBlock. Clamping.")
            h = torch.nan_to_num(h, nan=0.0, posinf=3.0, neginf=-3.0)
            h = torch.clamp(h, -5.0, 5.0)
            
        # Second normalization + convolution
        if self.use_adagn:
            h = self.block2(self.norm2(h, c_emb))
        else:
            norm2_out = self.norm2(h)
            norm2_out = torch.clamp(norm2_out, -5.0, 5.0)  # Clamp normalized output
            if torch.isnan(norm2_out).any() or torch.isinf(norm2_out).any():
                print(f"[WARNING] NaN/Inf after norm2 in ResNetBlock. Clamping.")
                norm2_out = torch.nan_to_num(norm2_out, nan=0.0, posinf=3.0, neginf=-3.0)
                norm2_out = torch.clamp(norm2_out, -5.0, 5.0)
            h = self.block2(norm2_out)
        
        # Aggressive clamp after block2
        h = torch.clamp(h, -5.0, 5.0)
        
        # Check after block2
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[WARNING] NaN/Inf after block2 in ResNetBlock. Clamping.")
            h = torch.nan_to_num(h, nan=0.0, posinf=3.0, neginf=-3.0)
            h = torch.clamp(h, -5.0, 5.0)

        # CRITICAL: Clamp x_res BEFORE shortcut to prevent shortcut explosion
        x_res = torch.clamp(x_res, -5.0, 5.0)
        
        # Residual connection - this is where NaN appears!
        shortcut_out = self.shortcut(x_res)
        
        # AGGRESSIVE clamping right after shortcut
        shortcut_out = torch.clamp(shortcut_out, -5.0, 5.0)
        
        if torch.isnan(shortcut_out).any() or torch.isinf(shortcut_out).any():
            print(f"[WARNING] NaN/Inf in shortcut AFTER clamp! Setting to zero.")
            shortcut_out = torch.zeros_like(shortcut_out)
        
        # Clamp both before adding
        h = torch.clamp(h, -5.0, 5.0)
        shortcut_out = torch.clamp(shortcut_out, -5.0, 5.0)
        
        h_residual = h + shortcut_out
        
        # Clamp after residual addition
        h_residual = torch.clamp(h_residual, -5.0, 5.0)
        
        if torch.isnan(h_residual).any() or torch.isinf(h_residual).any():
            print(f"[WARNING] NaN/Inf after residual addition. Using h only (skip shortcut).")
            h_residual = h  # Skip the shortcut if it causes issues
        
        h = self.attn(h_residual)
        
        # Clamp after attention
        h = torch.clamp(h, -5.0, 5.0)
        
        # Check after attention
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[WARNING] NaN/Inf after attention. Using pre-attention value.")
            h = h_residual
        
        h = self.upsample(h)
        
        # Final aggressive clamp
        h = torch.clamp(h, -5.0, 5.0)
        
        # Final check
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[WARNING] NaN/Inf in ResNetBlock output. Zeroing out.")
            h = torch.zeros_like(h)
        
        return h


class SpatialChannelAttention(nn.Module):
    """
    Combined spatial and channel attention mechanism.
    
    Applies channel attention followed by spatial attention to enhance
    feature representations.
    
    Args:
        channels: Number of input channels
    """
    def __init__(self, channels):
        super().__init__()
        
        # Channel attention
        self.channel_attn = ChannelAttention(channels)
        
        # Spatial attention
        self.spatial_conv = nn.Conv1d(2, 1, 7, padding=3)
        
    def forward(self, x):
        # Channel attention first
        x = self.channel_attn(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_attn))
        
        return x * spatial_attn


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism.
    
    Computes channel-wise attention weights using both average and max pooling.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the MLP (default: 8)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        
    def forward(self, x):
        B, C, L = x.shape
        
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        
        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1)
        return x * attn


class EnhancedTemporalAttention(nn.Module):
    """
    Enhanced temporal attention with better numerical stability.
    
    Improved version of TemporalSelfAttention1D with explicit softmax
    and dropout for better training stability.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads (default: 8)
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(min(32, channels // 4), channels, eps=1e-4)
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        b, c, n = x.shape
        residual = x
        
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        
        q, k, v = map(
            lambda t: t.reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2),
            qkv
        )
        
        # Enhanced attention with better numerical stability
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, n)
        
        return self.to_out(out) + residual


class Downsample1D(nn.Module):
    """
    1D downsampling layer using strided convolution.
    
    Reduces temporal resolution by factor of 2.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    1D upsampling layer using nearest neighbor interpolation + convolution.
    
    Increases temporal resolution by factor of 2.
    
    Args:
        in_channels: Number of input/output channels
    """
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, x):
        return self.conv(self.upsample(x))


class FeatureReconstructionHead(nn.Module):
    """
    Feature reconstruction head for VAE latent space to neurophysiological features.
    
    Maps from latent space to 12D neurophysiological feature vector. Used during
    VAE training to ensure the latent space preserves feature information.
    
    Args:
        latent_dim: Dimension of VAE latent space (default: 256)
        feature_dim: Dimension of feature vector (default: 12)
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
    
    Example:
        >>> head = FeatureReconstructionHead(latent_dim=256, feature_dim=12)
        >>> z = torch.randn(16, 256)  # Batch of latent vectors
        >>> features = head(z)  # Shape: (16, 12)
    """
    def __init__(self, latent_dim=256, feature_dim=12, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z: Latent vectors, shape (batch_size, latent_dim)
        
        Returns:
            Predicted features, shape (batch_size, feature_dim)
        """
        return self.model(z)


def sinusoidal_time_embedding(t_batch, embed_dim, device=None):
    """
    Generate sinusoidal time embeddings for diffusion models.
    
    Creates positional embeddings using sine and cosine functions at
    different frequencies, similar to Transformer positional encodings.
    
    Args:
        t_batch: Batch of timesteps, shape (batch_size,)
        embed_dim: Dimension of embedding
        device: Device to create embeddings on (default: same as t_batch)
    
    Returns:
        Time embeddings of shape (batch_size, embed_dim)
    """
    if device is None:
        device = t_batch.device
    
    half_dim = embed_dim // 2
    emb_log = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb_exp = torch.exp(torch.arange(half_dim, device=device) * -emb_log)
    emb = t_batch.float().unsqueeze(1) * emb_exp.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
