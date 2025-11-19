"""
Variational Autoencoder (VAE) for EEG Signal Compression

This module implements a VAE specifically designed for EEG signal compression
with the following enhancements:
- Skip connections for better information flow
- Improved numerical stability
- Feature conditioning support
- Enhanced temporal attention mechanisms
- Stable reparameterization trick

The VAE learns to compress EEG signals into a lower-dimensional latent space
while preserving important temporal and spectral characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import (
    ResNetBlock,
    TemporalSelfAttention1D,
    Downsample1D,
    Upsample1D
)


class DecoupledVAE(nn.Module):
    """
    Enhanced VAE with improved stability and reconstruction quality.
    
    Features:
    - Skip connections for better information flow
    - Improved numerical stability
    - Better feature conditioning support
    - Enhanced temporal attention mechanisms
    
    Args:
        in_channels: Number of input EEG channels (default: 16)
        base_channels: Base number of feature channels (default: 128)
        latent_channels: Number of latent channels (default: 64)
        ch_mults: Channel multipliers for each stage (default: (1, 2, 4, 8))
        blocks_per_stage: Number of ResNet blocks per stage (default: 2)
        use_feature_cond: Enable feature conditioning in decoder (default: True)
        feature_cond_dim: Dimension of feature conditioning (default: 12)
    """
    def __init__(self, 
                 in_channels=16, 
                 base_channels=128, 
                 latent_channels=64, 
                 ch_mults=(1, 2, 4, 8), 
                 blocks_per_stage=2,
                 use_feature_cond=True,
                 feature_cond_dim=12):
        super().__init__()
        
        # Store architecture parameters
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.use_feature_cond = use_feature_cond
        self.feature_cond_dim = feature_cond_dim
        
        # Encoder definition with skip connections
        self.encoder = nn.ModuleList()
        self.encoder_skip_convs = nn.ModuleList()  # Skip connection layers
        ch = base_channels
        
        # Initial conv
        self.encoder.append(nn.Conv1d(in_channels, ch, 3, padding=1))
        
        # Encoder blocks with skip connections
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            for j in range(blocks_per_stage):
                self.encoder.append(ResNetBlock(ch, out_ch))
                # Add skip connection projection for ResNetBlock outputs
                # Skip features will be used by decoder, so project to a consistent dimension
                # Use identity (no projection) if channels don't change, otherwise use 1x1 conv
                if j % 2 == 0:  # Add skip every other block for efficiency
                    # Project from out_ch (ResNetBlock output) to out_ch (keep same)
                    self.encoder_skip_convs.append(nn.Conv1d(out_ch, out_ch, kernel_size=1))
                else:
                    self.encoder_skip_convs.append(None)
                ch = out_ch
            if i != len(ch_mults) - 1:
                self.encoder.append(Downsample1D(ch))
                self.encoder_skip_convs.append(None)  # No skip for downsampling
        
        # Enhanced bottleneck with better attention
        self.enc_bottleneck_res1 = ResNetBlock(ch, ch)
        self.enc_bottleneck_attn = TemporalSelfAttention1D(ch)
        self.enc_bottleneck_dropout = nn.Dropout(0.1)  # Reduced from 0.15
        self.enc_bottleneck_res2 = ResNetBlock(ch, ch)
        
        # Improved latent projection with better initialization - Use GroupNorm for stability
        self.enc_pre_latent_norm = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch, eps=1e-6)
        self.to_latent = nn.Conv1d(ch, 2 * latent_channels, 1)
        
        # Initialize latent projection with conservative weights
        nn.init.xavier_normal_(self.to_latent.weight, gain=0.02)  # Reduced from 0.1
        nn.init.zeros_(self.to_latent.bias)

        # Enhanced Decoder with skip connections
        self.decoder = nn.ModuleList()
        self.decoder_skip_convs = nn.ModuleList()  # Decoder skip connections
        self.from_latent = nn.Conv1d(latent_channels, ch, 1)
        
        feature_dim_for_decoder = feature_cond_dim if use_feature_cond else None
        
        # Enhanced bottleneck
        self.dec_bottleneck_res1 = ResNetBlock(ch, ch, feature_cond_dim=feature_dim_for_decoder)
        self.dec_bottleneck_attn = TemporalSelfAttention1D(ch)
        self.dec_bottleneck_res2 = ResNetBlock(ch, ch, feature_cond_dim=feature_dim_for_decoder)
        
        # Decoder blocks with skip connections
        # The decoder processes in REVERSE order, so we build skip_convs accordingly
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_channels * mult
            for j in range(blocks_per_stage):
                # Decoder will receive skip features in reverse order
                # So decoder_skip_convs[0] corresponds to the LAST encoder block
                self.decoder_skip_convs.append(nn.Conv1d(ch + out_ch, out_ch, kernel_size=1))
                self.decoder.append(ResNetBlock(ch, out_ch, feature_cond_dim=feature_dim_for_decoder))
                ch = out_ch
                
            if i != 0: 
                self.decoder.append(Upsample1D(ch))
                self.decoder_skip_convs.append(None)  # No skip for upsampling

        # Enhanced final layers - Use GroupNorm instead of LayerNorm for stability
        self.dec_final_norm = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch, eps=1e-6)
        self.dec_final_act1 = nn.SiLU()
        self.dec_final_conv1 = nn.Conv1d(ch, ch//2, 3, padding=1)
        self.dec_final_norm2 = nn.GroupNorm(num_groups=min(32, ch//2), num_channels=ch//2, eps=1e-6)
        self.dec_final_act2 = nn.SiLU()
        self.dec_final_conv2 = nn.Conv1d(ch//2, in_channels, 1)
        
        # Initialize final layers with very small weights to prevent initial explosion
        nn.init.xavier_normal_(self.dec_final_conv1.weight, gain=0.02)
        nn.init.zeros_(self.dec_final_conv1.bias)
        nn.init.xavier_normal_(self.dec_final_conv2.weight, gain=0.01)
        nn.init.zeros_(self.dec_final_conv2.bias)
        
        # CRITICAL: Initialize encoder's initial conv layer properly
        # This is the first layer that receives raw input, needs careful init
        if isinstance(self.encoder[0], nn.Conv1d):
            # Use Kaiming init for first layer (good for ReLU-family activations)
            # mode='fan_in' ensures variance is preserved in forward pass
            nn.init.kaiming_normal_(self.encoder[0].weight, mode='fan_in', nonlinearity='linear')
            if self.encoder[0].bias is not None:
                nn.init.zeros_(self.encoder[0].bias)
        
        # Initialize encoder/decoder skip projections
        for skip_conv in self.encoder_skip_convs:
            if skip_conv is not None and isinstance(skip_conv, nn.Conv1d):
                nn.init.orthogonal_(skip_conv.weight, gain=1.0)  # Identity-like
                if skip_conv.bias is not None:
                    nn.init.zeros_(skip_conv.bias)
        
        for skip_conv in self.decoder_skip_convs:
            if skip_conv is not None and isinstance(skip_conv, nn.Conv1d):
                nn.init.orthogonal_(skip_conv.weight, gain=1.0)
                if skip_conv.bias is not None:
                    nn.init.zeros_(skip_conv.bias)
        
        # Initialize from_latent layer
        nn.init.kaiming_normal_(self.from_latent.weight, mode='fan_in', nonlinearity='linear')
        if self.from_latent.bias is not None:
            nn.init.zeros_(self.from_latent.bias)

    def encode(self, x):
        """
        Encode input to latent distribution parameters with skip connections.
        
        Skip features are collected at each ResNetBlock in the encoder.
        They will be used in REVERSE order by the decoder.
        
        Args:
            x: Input tensor of shape (batch, in_channels, length)
        
        Returns:
            mu: Mean of latent distribution (batch, latent_channels, latent_length)
            logvar: Log variance of latent distribution (batch, latent_channels, latent_length)
            skip_features: List of skip connection features for decoder (forward order from encoder)
        """
        h = x
        skip_features = []
        
        # Collect skip features from encoder blocks
        for i, block in enumerate(self.encoder):
            h = block(h)
            
            # Store skip features at ResNetBlock outputs
            if isinstance(block, ResNetBlock) and i < len(self.encoder_skip_convs):
                if self.encoder_skip_convs[i] is not None:
                    # Project skip feature to appropriate channels
                    skip_feat = self.encoder_skip_convs[i](h)
                    skip_features.append(skip_feat)
                else:
                    skip_features.append(None)
            elif isinstance(block, Downsample1D):
                # No skip for downsampling layers
                skip_features.append(None)
        
        # Enhanced bottleneck
        h = self.enc_bottleneck_res1(h)
        h = self.enc_bottleneck_attn(h)
        h = self.enc_bottleneck_dropout(h)
        h = self.enc_bottleneck_res2(h)
        
        # Stable latent projection - GroupNorm operates on (B, C, L) directly
        h_norm = self.enc_pre_latent_norm(h)
        
        # Check for NaN before latent projection
        if torch.isnan(h_norm).any() or torch.isinf(h_norm).any():
            print("[WARNING] NaN/Inf in encoder bottleneck. Clamping.")
            h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=5.0, neginf=-5.0)
        
        latent_params = self.to_latent(h_norm)
        mu, logvar = latent_params.chunk(2, dim=1)
        
        # Moderate clamping for numerical stability
        mu = torch.clamp(mu, -5.0, 5.0)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        
        return mu, logvar, skip_features

    def decode(self, z, feature_cond=None, skip_features=None):
        """
        Decode latent code to reconstructed signal with PROPER skip connections.
        
        Skip features from encoder are used in REVERSE order:
        - Last encoder block → First decoder block
        - First encoder block → Last decoder block
        
        Uses dimension alignment strategy:
        1. Adaptive pooling for spatial dimension matching
        2. Channel concatenation [decoder_h, skip_feat]
        3. 1x1 conv projection to target channels
        
        Args:
            z: Latent code (batch, latent_channels, latent_length)
            feature_cond: Optional feature conditioning (batch, feature_cond_dim)
            skip_features: Optional skip connection features from encoder (forward order)
        
        Returns:
            Reconstructed signal (batch, in_channels, length)
        """
        h = self.from_latent(z)
        
        # Check after from_latent
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("[WARNING] NaN/Inf after from_latent. Clamping.")
            h = torch.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Enhanced bottleneck
        h = self.dec_bottleneck_res1(h, f_emb=feature_cond)
        h = self.dec_bottleneck_attn(h)
        h = self.dec_bottleneck_res2(h, f_emb=feature_cond)
        
        # Check for NaN after bottleneck
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("[WARNING] NaN/Inf in decoder bottleneck. Clamping.")
            h = torch.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Decoder processes blocks in reverse order relative to encoder
        # So we reverse skip_features to match decoder block order
        if skip_features is not None:
            skip_features_reversed = list(reversed(skip_features))
        else:
            skip_features_reversed = None
        
        decoder_idx = 0  # Index into decoder blocks
        for i, block in enumerate(self.decoder):
            if isinstance(block, ResNetBlock):
                # Try to use skip connection if available
                use_skip = False
                if (skip_features_reversed is not None and 
                    decoder_idx < len(skip_features_reversed) and
                    decoder_idx < len(self.decoder_skip_convs) and
                    self.decoder_skip_convs[decoder_idx] is not None):
                    
                    skip_feat = skip_features_reversed[decoder_idx]
                    
                    # Validate skip feature
                    if skip_feat is not None:
                        if torch.isnan(skip_feat).any() or torch.isinf(skip_feat).any():
                            print(f"[WARNING] NaN/Inf in skip feature at decoder block {decoder_idx}. Skipping connection.")
                        else:
                            # DIMENSION ALIGNMENT STRATEGY
                            # Step 1: Align spatial dimensions using adaptive pooling (more stable than interpolate)
                            if h.shape[-1] != skip_feat.shape[-1]:
                                skip_feat = F.adaptive_avg_pool1d(skip_feat, h.shape[-1])
                            
                            # Step 2: Verify spatial dimensions now match
                            if h.shape[-1] == skip_feat.shape[-1]:
                                # Step 3: Concatenate along channel dimension: [decoder_h, skip_feat]
                                h_with_skip = torch.cat([h, skip_feat], dim=1)
                                
                                # Step 4: Project concatenated features to target channels using 1x1 conv
                                # This is the KEY step - reduces [h_channels + skip_channels] → [target_channels]
                                h_projected = self.decoder_skip_convs[decoder_idx](h_with_skip)
                                
                                # Step 5: Validate projection output
                                if torch.isnan(h_projected).any() or torch.isinf(h_projected).any():
                                    print(f"[WARNING] NaN/Inf after skip projection at decoder block {decoder_idx}. Skipping connection.")
                                else:
                                    # Successfully applied skip connection
                                    h = h_projected
                                    use_skip = True
                            else:
                                print(f"[WARNING] Spatial mismatch at decoder block {decoder_idx}: h={h.shape[-1]}, skip={skip_feat.shape[-1]}. Skipping.")
                
                # Apply ResNetBlock (with or without skip connection applied above)
                h = block(h, f_emb=feature_cond)
                decoder_idx += 1
            else:
                # Upsample or other non-ResNet layers - no skip connection
                h = block(h)
            
            # Check for NaN after each decoder block
            if torch.isnan(h).any() or torch.isinf(h).any():
                print(f"[WARNING] NaN/Inf in decoder block {i}. Clamping.")
                h = torch.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)
                    
        # Enhanced final reconstruction - GroupNorm operates on (B, C, L) directly
        h_norm = self.dec_final_norm(h)
        
        # Check for NaN after final norm
        if torch.isnan(h_norm).any() or torch.isinf(h_norm).any():
            print("[WARNING] NaN/Inf after final norm. Clamping.")
            h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Apply final convolutions step by step with checks
        recon = self.dec_final_act1(h_norm)
        recon = self.dec_final_conv1(recon)
        
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print("[WARNING] NaN/Inf after dec_final_conv1. Clamping.")
            recon = torch.nan_to_num(recon, nan=0.0, posinf=5.0, neginf=-5.0)
        
        recon = self.dec_final_norm2(recon)
        recon = self.dec_final_act2(recon)
        recon = self.dec_final_conv2(recon)
        
        # Final NaN check and replacement
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print("[WARNING] NaN/Inf in final reconstruction. Replacing with zeros.")
            recon = torch.nan_to_num(recon, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # CRITICAL FIX: Scale output to match expected EEG magnitude
        # The decoder output should have std ~0.2-0.3 for normalized EEG
        # Without this, the output collapses to near-zero magnitude
        recon = recon * 2.0  # FIXED: Empirical scaling factor to restore proper magnitude
        
        # Moderate output clamping
        recon = torch.clamp(recon, -5.0, 5.0)
        
        return recon

    def reparameterize(self, mu, logvar):
        """
        Enhanced reparameterization with maximum numerical stability.
        
        Uses the reparameterization trick to sample from the latent distribution
        while maintaining gradient flow: z = mu + std * epsilon
        
        Uses softplus instead of exp for better numerical stability.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent code
        """
        # Conservative clamping
        logvar_clamped = torch.clamp(logvar, -5.0, 2.0)
        
        # Use softplus for better stability: std = sqrt(softplus(logvar) + eps)
        # This avoids potential exp overflow and is smoother
        std = torch.sqrt(F.softplus(logvar_clamped) + 1e-8)
        
        # Clamp std to prevent explosion
        std = torch.clamp(std, min=1e-8, max=1.5)
        
        # Sample noise
        noise = torch.randn_like(std)
        z = mu + noise * std
        
        # Output clamping
        z = torch.clamp(z, -10.0, 10.0)
        
        return z

    def forward(self, x, feature_cond=None):
        """
        Full forward pass through VAE.
        
        Args:
            x: Input signal (batch, in_channels, length)
            feature_cond: Optional feature conditioning (batch, feature_cond_dim)
        
        Returns:
            recon_x: Reconstructed signal (batch, in_channels, length)
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent code
            is_stable: Boolean flag indicating numerical stability
        """
        # Check input for NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN/Inf in VAE input. Clamping.")
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            x = torch.clamp(x, -5.0, 5.0)
        
        mu, logvar, skip_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, feature_cond=feature_cond, skip_features=skip_features)
        
        # Final safety check - replace any NaN with zeros
        if torch.isnan(recon_x).any():
            print("[ERROR] NaN in final reconstruction. Replacing with zeros.")
            recon_x = torch.nan_to_num(recon_x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Enhanced stability checking
        is_stable = (torch.isfinite(recon_x).all() and 
                    torch.isfinite(mu).all() and 
                    torch.isfinite(logvar).all() and 
                    torch.isfinite(z).all() and
                    not torch.isnan(recon_x).any() and
                    not torch.isnan(mu).any() and
                    not torch.isnan(logvar).any())
        
        # Return enhanced output tuple
        return recon_x, mu, logvar, z, is_stable

    def encode_to_mu_logvar(self, x):
        """
        Backward compatibility method for encoding without skip features.
        
        Args:
            x: Input signal (batch, in_channels, length)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar, _ = self.encode(x)
        return mu, logvar
    
    def compute_latent_statistics(self, data_loader, device='cuda', max_batches=50):
        """
        Compute latent space statistics to assess learning quality.
        
        This helps monitor whether the VAE is learning meaningful class-separated
        representations that will effectively guide the LDM.
        
        Args:
            data_loader: DataLoader with (data, labels) batches
            device: Device to run computation on
            max_batches: Maximum number of batches to process
        
        Returns:
            Dictionary with latent space statistics:
            - intra_class_distances: Mean distance within each class
            - inter_class_distances: Mean distance between class centroids
            - class_centroids: Mean latent code for each class
            - class_stds: Standard deviation within each class
        """
        self.eval()
        latent_codes_by_class = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                # Extract data and labels
                if len(batch) == 4:
                    data, _, _, labels = batch
                elif len(batch) == 3:
                    data, labels, _ = batch
                elif len(batch) == 2:
                    data, labels = batch
                else:
                    continue
                
                data = data.to(device)
                labels = labels.cpu().numpy()
                
                # Encode to latent space
                mu, logvar, _ = self.encode(data)
                z = self.reparameterize(mu, logvar)
                
                # Store latent codes by class
                z_np = z.cpu().numpy()
                for i, label in enumerate(labels):
                    if label not in latent_codes_by_class:
                        latent_codes_by_class[label] = []
                    # Flatten spatial dimension for distance computation
                    latent_codes_by_class[label].append(z_np[i].reshape(-1))
        
        # Compute statistics
        import numpy as np
        
        class_centroids = {}
        class_stds = {}
        intra_class_distances = {}
        
        for label, codes in latent_codes_by_class.items():
            codes_array = np.array(codes)  # (num_samples, latent_dim)
            centroid = codes_array.mean(axis=0)
            std = codes_array.std(axis=0).mean()
            
            # Compute mean intra-class distance
            distances = np.linalg.norm(codes_array - centroid, axis=1)
            mean_dist = distances.mean()
            
            class_centroids[label] = centroid
            class_stds[label] = std
            intra_class_distances[label] = mean_dist
        
        # Compute inter-class distances
        inter_class_distances = {}
        class_labels = list(class_centroids.keys())
        for i, label1 in enumerate(class_labels):
            for label2 in class_labels[i+1:]:
                dist = np.linalg.norm(class_centroids[label1] - class_centroids[label2])
                inter_class_distances[f"{label1}_vs_{label2}"] = dist
        
        self.train()
        
        return {
            'intra_class_distances': intra_class_distances,
            'inter_class_distances': inter_class_distances,
            'class_centroids': class_centroids,
            'class_stds': class_stds,
            'separation_ratio': (
                np.mean(list(inter_class_distances.values())) / 
                np.mean(list(intra_class_distances.values()))
                if intra_class_distances and inter_class_distances else 0.0
            )
        }


def get_cosine_schedule(timesteps, s=0.008):
    """
    Create a cosine beta schedule optimized for EEG signals.
    
    Modified to better preserve low frequency content compared to linear schedules.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small (default: 0.008)
    
    Returns:
        Beta schedule tensor of shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # Reduce s parameter to make schedule more gradual and preserve low frequencies
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clamp to preserve more low frequency content
    return torch.clip(betas, 0.0001, 0.015)  # Reduced max from 0.02 to 0.015
