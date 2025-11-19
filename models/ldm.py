"""
Latent Diffusion Model (LDM) for EEG Generation

This module implements a conditional latent diffusion model for EEG signal generation.
The model operates in the latent space learned by the VAE and uses a U-Net architecture
with the following conditioning mechanisms:
- Class labels (seizure type) via AdaGN
- Neurophysiological features via cross-attention
- Time steps via sinusoidal embeddings

The LDM enables controlled generation of realistic EEG signals for data augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import (
    ResNetBlock,
    CrossAttention,
    sinusoidal_time_embedding
)


class LatentDiffusionUNetEEG(nn.Module):
    """
    U-Net based latent diffusion model for conditional EEG generation.
    
    Architecture:
    - Encoder: Downsampling path with ResNet blocks and cross-attention
    - Bottleneck: Middle blocks with self-attention and cross-attention
    - Decoder: Upsampling path with skip connections
    
    Conditioning:
    - Time steps: Sinusoidal embeddings + MLP projection
    - Class labels: Embedding + AdaGN modulation
    - Features: MLP + cross-attention
    
    Args:
        latent_channels: Number of channels in latent space (default: 64)
        base_unet_channels: Base number of U-Net channels (default: 256)
        time_emb_dim: Dimension of time embeddings (default: 256)
        num_classes: Number of EEG classes (default: 3)
        channel_mults: Channel multipliers for each stage (default: (1, 2, 4, 8))
        num_res_blocks: Number of ResNet blocks per stage (default: 2)
        use_cross_attention: Enable cross-attention for feature conditioning (default: True)
        use_adagn: Enable Adaptive GroupNorm for class conditioning (default: True)
        context_dim: Dimension of context embeddings (default: 256)
        feature_cond_dim: Dimension of neurophysiological features (default: 12)
        use_feature_cond: Enable feature conditioning (default: True)
    """
    def __init__(self, 
                 latent_channels=64,
                 base_unet_channels=256,
                 time_emb_dim=256,
                 num_classes=3,
                 channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2,
                 use_cross_attention=True,
                 use_adagn=True,
                 context_dim=256,
                 feature_cond_dim=12,
                 use_feature_cond=True):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_classes = num_classes
        self.time_embed_dim = time_emb_dim
        self.use_cross_attention = use_cross_attention
        self.use_adagn = use_adagn
        self.context_dim = context_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Context embedding: class labels
        self.class_embed = nn.Embedding(num_classes + 1, context_dim)  # +1 for unconditional
        
        # Context embedding: neurophysiological features (optional)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_cond_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        ) if use_feature_cond else None

        # Context projector: combines class + features
        context_input_dim = context_dim * 2 if self.feature_mlp is not None else context_dim
        self.context_projector = nn.Sequential(
            nn.Linear(context_input_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )

        # U-Net backbone
        self.initial_conv = nn.Conv1d(latent_channels, base_unet_channels, 3, padding=1)
        channels, ch = [base_unet_channels], base_unet_channels
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()

        # Encoder (downsampling path)
        for i, mult in enumerate(channel_mults):
            out_ch = base_unet_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResNetBlock(
                    ch, out_ch, 
                    time_emb_dim=time_emb_dim, 
                    use_adagn=use_adagn, 
                    class_cond_dim=context_dim
                ))
                if use_cross_attention:
                    self.downs.append(CrossAttention(out_ch, context_dim))
                ch = out_ch
                channels.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(ResNetBlock(
                    ch, ch, 
                    time_emb_dim=time_emb_dim, 
                    down=True, 
                    use_adagn=use_adagn, 
                    class_cond_dim=context_dim
                ))
                channels.append(ch)

        # Middle blocks (bottleneck)
        self.mid_block1 = ResNetBlock(
            ch, ch, 
            time_emb_dim=time_emb_dim, 
            use_attention=True, 
            use_adagn=use_adagn, 
            class_cond_dim=context_dim
        )
        if use_cross_attention:
            self.mid_attn = CrossAttention(ch, context_dim)
        self.mid_block2 = ResNetBlock(
            ch, ch, 
            time_emb_dim=time_emb_dim, 
            use_attention=True, 
            use_adagn=use_adagn, 
            class_cond_dim=context_dim
        )

        # Decoder (upsampling path)
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_unet_channels * mult
            for _ in range(num_res_blocks + 1):  # +1 for skip connection
                in_ch_skip = channels.pop()
                self.ups.append(ResNetBlock(
                    ch + in_ch_skip, out_ch, 
                    time_emb_dim=time_emb_dim, 
                    use_adagn=use_adagn, 
                    class_cond_dim=context_dim
                ))
                if use_cross_attention:
                    self.ups.append(CrossAttention(out_ch, context_dim))
                ch = out_ch
            if i != 0:
                self.ups.append(ResNetBlock(
                    ch, ch, 
                    up=True, 
                    time_emb_dim=time_emb_dim, 
                    use_adagn=use_adagn, 
                    class_cond_dim=context_dim
                ))

        # Final output layers
        self.final_norm = nn.GroupNorm(32, ch)
        self.final_conv = nn.Conv1d(ch, latent_channels, 1)

    def forward(self, z_t, time_steps, class_labels=None, feature_cond=None):
        """
        Forward pass through the U-Net.
        
        Args:
            z_t: Noisy latent code (batch, latent_channels, latent_length)
            time_steps: Diffusion time steps (batch,)
            class_labels: Class labels for conditioning (batch,) - optional
            feature_cond: Neurophysiological features (batch, feature_cond_dim) - optional
        
        Returns:
            Predicted noise or denoised sample (batch, latent_channels, latent_length)
        """
        # Time embedding
        t_emb = self.time_mlp(sinusoidal_time_embedding(time_steps, self.time_embed_dim))

        # Class embedding (unconditional if None)
        if class_labels is None:
            class_labels = torch.full((z_t.shape[0],), self.num_classes,
                                    device=z_t.device, dtype=torch.long)
        c_embed = self.class_embed(class_labels)  # (batch, context_dim)

        # Feature embedding (if enabled)
        if self.feature_mlp is not None:
            if feature_cond is not None:
                f_embed = self.feature_mlp(feature_cond)  # (batch, context_dim)
            else:
                # Zero embedding if features not provided
                f_embed = torch.zeros(z_t.shape[0], self.context_dim,
                                    device=z_t.device, dtype=c_embed.dtype)
            combined_embed = torch.cat([c_embed, f_embed], dim=1)  # (batch, context_dim*2)
        else:
            combined_embed = c_embed  # (batch, context_dim)

        # Project combined context
        final_context_embed = self.context_projector(combined_embed)  # (batch, context_dim)

        # U-Net forward pass
        h = self.initial_conv(z_t)
        skips = [h]
        
        # Prepare cross-attention context
        cross_attn_context = final_context_embed.unsqueeze(1) if self.use_cross_attention else None

        # Encoder - collect skip features ONLY from ResNetBlocks
        for block in self.downs:
            if isinstance(block, CrossAttention):
                if cross_attn_context is not None:
                    # Cross-attention: (batch, length, channels)
                    h_attn = block(h.permute(0, 2, 1), cross_attn_context).permute(0, 2, 1)
                    h = h_attn + h  # Residual connection
                # CrossAttention doesn't change shape, no skip needed
            else:
                # ResNetBlock - save output for skip connection
                h = block(h, t_emb=t_emb, c_emb=final_context_embed)
                skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb=t_emb, c_emb=final_context_embed)
        if self.use_cross_attention and hasattr(self, 'mid_attn') and cross_attn_context is not None:
            h = self.mid_attn(h.permute(0, 2, 1), cross_attn_context).permute(0, 2, 1) + h
        h = self.mid_block2(h, t_emb=t_emb, c_emb=final_context_embed)

        # Decoder - use skip connections properly
        for block in self.ups:
            if isinstance(block, CrossAttention):
                if cross_attn_context is not None:
                    h_attn = block(h.permute(0, 2, 1), cross_attn_context).permute(0, 2, 1)
                    h = h_attn + h  # Residual connection
            elif hasattr(block, 'up') and block.up:
                # Upsampling block - no skip connection
                h = block(h, t_emb=t_emb, c_emb=final_context_embed)
            else:
                # Regular ResNetBlock - concatenate skip connection
                if len(skips) > 0:
                    skip_feat = skips.pop()
                    # Ensure spatial dimensions match before concatenation
                    if h.shape[-1] != skip_feat.shape[-1]:
                        # Use adaptive pooling for dimension alignment
                        skip_feat = F.adaptive_avg_pool1d(skip_feat, h.shape[-1])
                    h = torch.cat([h, skip_feat], dim=1)
                h = block(h, t_emb=t_emb, c_emb=final_context_embed)

        # Final output
        out = self.final_conv(F.silu(self.final_norm(h)))
        
        # Validate output
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[WARNING] NaN/Inf detected in LDM output. Clamping...")
            out = torch.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return out
    
    @torch.no_grad()
    def sample(self, 
               batch_size, 
               latent_shape,
               num_steps=50,
               class_labels=None,
               feature_cond=None,
               cfg_scale=1.0,
               scheduler=None,
               device='cuda'):
        """
        Generate samples using DDPM sampling with optional classifier-free guidance.
        
        Args:
            batch_size: Number of samples to generate
            latent_shape: Shape of latent code (channels, length), e.g., (64, 20)
            num_steps: Number of denoising steps (default: 50)
            class_labels: Class labels for conditional generation (batch_size,)
            feature_cond: Neurophysiological features (batch_size, feature_dim)
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger)
            scheduler: DDPMScheduler instance
            device: Device to generate on
        
        Returns:
            Generated latent codes (batch_size, latent_channels, latent_length)
        """
        if scheduler is None:
            scheduler = DDPMScheduler(num_timesteps=1000).to(device)
        
        # Start from pure noise
        z_t = torch.randn(batch_size, *latent_shape, device=device)
        
        # Sampling timesteps (linearly spaced from T to 0)
        timesteps = torch.linspace(scheduler.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        # Classifier-free guidance: run unconditional in parallel if cfg_scale > 1.0
        use_cfg = cfg_scale > 1.0
        
        for i, t in enumerate(timesteps):
            # Expand time step for batch
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            if use_cfg:
                # Run conditional and unconditional in parallel
                # Duplicate input
                z_t_combined = torch.cat([z_t, z_t], dim=0)
                t_combined = torch.cat([t_batch, t_batch], dim=0)
                
                # Conditional uses real labels, unconditional uses None (→ class=num_classes)
                labels_combined = torch.cat([
                    class_labels if class_labels is not None else torch.zeros(batch_size, dtype=torch.long, device=device),
                    torch.full((batch_size,), self.num_classes, dtype=torch.long, device=device)
                ], dim=0)
                
                # Features: use for conditional, zeros for unconditional
                if self.feature_mlp is not None:
                    if feature_cond is not None:
                        feat_combined = torch.cat([
                            feature_cond,
                            torch.zeros_like(feature_cond)
                        ], dim=0)
                    else:
                        feat_combined = None
                else:
                    feat_combined = None
                
                # Predict noise
                noise_pred_combined = self(z_t_combined, t_combined, labels_combined, feat_combined)
                
                # Split predictions
                noise_pred_cond, noise_pred_uncond = noise_pred_combined.chunk(2, dim=0)
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard conditional sampling
                noise_pred = self(z_t, t_batch, class_labels, feature_cond)
            
            # Denoise one step
            if i < len(timesteps) - 1:
                # Standard DDPM step
                alpha_t = scheduler.alphas_cumprod[t]
                alpha_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]]
                
                # Predict x_0
                pred_x0 = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                
                # Clamp predicted x_0 for stability
                pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
                
                # Compute direction to x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
                
                # Compute x_{t-1}
                z_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
                
                # Add noise (except for last step)
                if i < len(timesteps) - 2:
                    noise = torch.randn_like(z_t)
                    sigma = torch.sqrt(scheduler.posterior_variance[t])
                    z_t = z_t + sigma * noise
            else:
                # Final step: predict x_0 directly
                alpha_t = scheduler.alphas_cumprod[t]
                z_t = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                z_t = torch.clamp(z_t, -5.0, 5.0)
        
        return z_t
    
    def get_loss(self, z_0, class_labels=None, feature_cond=None, scheduler=None, device='cuda'):
        """
        Compute diffusion loss for training.
        
        Args:
            z_0: Clean latent codes (batch, latent_channels, latent_length)
            class_labels: Class labels (batch,)
            feature_cond: Neurophysiological features (batch, feature_dim)
            scheduler: DDPMScheduler instance
            device: Device
        
        Returns:
            loss: MSE loss between predicted and true noise
        """
        if scheduler is None:
            scheduler = DDPMScheduler(num_timesteps=1000).to(device)
        
        batch_size = z_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(z_0)
        
        # Add noise to latent (forward diffusion)
        z_t = scheduler.add_noise(z_0, t, noise)
        
        # Predict noise
        noise_pred = self(z_t, t, class_labels, feature_cond)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Model) noise scheduler.
    
    Implements the forward diffusion process (adding noise) and reverse
    process (denoising) with cosine or linear beta schedules.
    
    Args:
        num_timesteps: Number of diffusion steps (default: 1000)
        beta_schedule: Type of schedule ('cosine' or 'linear', default: 'cosine')
        beta_start: Starting beta value for linear schedule (default: 0.0001)
        beta_end: Ending beta value for linear schedule (default: 0.02)
    """
    def __init__(self, num_timesteps=1000, beta_schedule='cosine', 
                 beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Generate beta schedule
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_0: Original data (batch, channels, length)
            t: Time steps (batch,)
            noise: Gaussian noise (batch, channels, length) - optional
        
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
    
    def step(self, model_output, t, x_t):
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        
        Args:
            model_output: Model prediction (noise or x_0)
            t: Current time step (batch,)
            x_t: Current noisy data (batch, channels, length)
        
        Returns:
            Denoised sample x_{t-1}
        """
        alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0
        pred_original_sample = (x_t - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # Compute coefficients for x_{t-1}
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample
    
    def to(self, device):
        """Move scheduler tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
