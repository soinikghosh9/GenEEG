"""
Helper Utilities for GenEEG Training

This module contains helper functions, optimizers, schedulers, and utilities
used throughout the training pipeline.

Classes:
    - WarmupCosineAnnealingLR: LR scheduler with linear warmup + cosine annealing
    - Lookahead: Lookahead optimizer wrapper for better convergence
    - BatchAugmentation: Simple data augmentation module
    
Functions:
    - get_cosine_schedule: Cosine beta schedule for diffusion
    - robust_latent_scaling_from_mu: Compute LDM scaling factor from VAE latents
    - get_vae_encoder_features: Extract VAE latent features from EEG data
    - nt_xent_loss: NT-Xent contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
import numpy as np
import math
from typing import Optional, Tuple


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing learning rate scheduler.
    
    Safe when max_epochs == warmup_epochs (avoids division by zero).
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs (linear ramp from 0 to base_lr)
        max_epochs: Total number of training epochs
        min_lr: Minimum learning rate for cosine phase (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=100)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        assert max_epochs > 0, "max_epochs must be > 0"
        # Clamp warmup so we always have at least 1 cosine step
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(max(0, min(warmup_epochs, self.max_epochs - 1)))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        base_lrs = self.base_lrs

        # Warmup phase (linear from 0 → base_lr)
        if self.last_epoch < self.warmup_epochs:
            warmup_ratio = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [lr * warmup_ratio for lr in base_lrs]

        # Cosine annealing phase
        elapsed = (self.last_epoch + 1) - self.warmup_epochs
        total_cos_epochs = max(1, self.max_epochs - self.warmup_epochs)  # Critical guard
        progress = min(elapsed, total_cos_epochs)

        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress / total_cos_epochs))
        return [self.min_lr + (lr - self.min_lr) * cos_factor for lr in base_lrs]


# =============================================================================
# Optimizers
# =============================================================================

class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimizer wrapper for improved convergence.
    
    Implements the Lookahead algorithm which maintains two sets of weights:
    - Fast weights (updated every step)
    - Slow weights (updated every k steps)
    
    Args:
        optimizer: Base optimizer (e.g., Adam, AdamW)
        k: Number of fast steps before slow update (default: 6)
        alpha: Slow weights step size (default: 0.5)
    
    Reference:
        Zhang et al. "Lookahead Optimizer: k steps forward, 1 step back"
        NeurIPS 2019
    
    Example:
        >>> base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> optimizer = Lookahead(base_optimizer, k=6, alpha=0.5)
    """
    
    def __init__(self, optimizer, k: int = 6, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        # Initialize counters
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        """Update slow weights."""
        for fast in group['params']:
            if fast.grad is None:
                continue
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.zeros_like(fast.data)
                param_state['slow_param'].copy_(fast.data)
            slow = param_state['slow_param']
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self.update(group)
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
        return loss

    def state_dict(self):
        """Return state dict for checkpointing."""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            'state': self.state,
            'param_groups': self.param_groups,
        }
        return {'fast_state': fast_state_dict, 'slow_state': slow_state}

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        slow_state_dict = state_dict['slow_state']
        self.optimizer.load_state_dict(state_dict['fast_state'])
        self.param_groups = slow_state_dict['param_groups']
        self.state = slow_state_dict['state']

    def __getattr__(self, name):
        """Forward missing attribute requests to the underlying optimizer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.optimizer, name)


# =============================================================================
# Data Augmentation
# =============================================================================

class BatchAugmentation(nn.Module):
    """
    Simple batch augmentation for EEG signals.
    
    Applies:
    - Gaussian noise
    - Time cutout (random segment masking)
    
    Args:
        noise_level: Standard deviation of Gaussian noise (default: 0.005 - reduced for stability)
        cutout_ratio: Fraction of sequence to mask (default: 0.05 - reduced for stability)
    """
    
    def __init__(self, noise_level: float = 0.005, cutout_ratio: float = 0.05):
        super().__init__()
        self.noise_level = noise_level
        self.cutout_ratio = cutout_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, L)
        
        Returns:
            Augmented tensor of shape (B, C, L)
        """
        # Apply Gaussian noise
        x = x + torch.randn_like(x) * self.noise_level
        
        # Apply time cutout
        batch_size, channels, length = x.shape
        cutout_length = int(length * self.cutout_ratio)
        
        for i in range(batch_size):
            if cutout_length > 0 and cutout_length < length:
                cutout_start = torch.randint(0, length - cutout_length, (1,)).item()
                x[i, :, cutout_start:cutout_start + cutout_length] = 0
        
        return x


# =============================================================================
# Diffusion Utilities
# =============================================================================

def get_cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create a cosine beta schedule optimized for EEG signals.
    
    Modified to better preserve low frequency content compared to linear schedule.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset for numerical stability (default: 0.008)
    
    Returns:
        Beta values for each timestep, shape (timesteps,)
    
    Reference:
        Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
        ICML 2021
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # Cosine schedule with reduced s parameter for gradual transitions
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Clamp to preserve more low frequency content
    # Reduced max from 0.02 to 0.015 for EEG signals
    return torch.clip(betas, 0.0001, 0.015)


# =============================================================================
# VAE Utilities
# =============================================================================

def robust_latent_scaling_from_mu(
    vae_model,
    loader,
    device,
    max_samples: int = 200_000
) -> float:
    """
    Compute robust scaling factor for LDM based on VAE latent μ distribution.
    
    Uses quantiles for robustness and subsamples large datasets to avoid
    'quantile() input tensor is too large' RuntimeError.
    
    Args:
        vae_model: Trained VAE model
        loader: DataLoader with EEG samples
        device: Device to run computation on
        max_samples: Maximum number of latent values to use (default: 200,000)
    
    Returns:
        Scaling factor (float) computed as IQR/1.349 ≈ std if Gaussian
    
    Note:
        This scaling factor is used to normalize VAE latents before LDM training:
        z_normalized = mu / scaling_factor
    """
    vae_model.eval()
    all_mus = []

    with torch.no_grad():
        for batch in loader:
            # Handle different loader formats
            if isinstance(batch, (list, tuple)):
                xb = batch[0]  # First element is always the data
            else:
                xb = batch
            
            xb = xb.to(device, non_blocking=True)
            
            # Handle enhanced VAE encode method
            encode_result = vae_model.encode(xb)
            if isinstance(encode_result, (tuple, list)):
                if len(encode_result) == 3:
                    # ImprovedDecoupledVAE returns (mu, logvar, skip_features)
                    mu, logvar, skip_features = encode_result
                elif len(encode_result) == 2:
                    # Standard VAE returns (mu, logvar)
                    mu, logvar = encode_result
                else:
                    # Fallback: take first element
                    mu = encode_result[0]
            else:
                # Single tensor return
                mu = encode_result
            
            # Flatten across all but batch dimension
            all_mus.append(mu.detach().cpu().reshape(-1))

    if not all_mus:
        print("[WARN] No latents found, returning scale=1.0")
        return 1.0

    M = torch.cat(all_mus, dim=0)

    # Subsample if too large to avoid memory issues
    if M.numel() > max_samples:
        idx = torch.randperm(M.numel())[:max_samples]
        M = M[idx]

    # Move to numpy for robust quantile computation
    M_np = M.cpu().numpy()
    q25, q75 = np.percentile(M_np, [25, 75])
    iqr = max(q75 - q25, 1e-6)

    # Scale estimate ~ IQR/1.349 (approx std if Gaussian)
    scale = float(iqr / 1.349)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(M_np.std())

    print(f"[robust_latent_scaling_from_mu] scale={scale:.4f} (q25={q25:.4f}, q75={q75:.4f})")
    return scale


def get_vae_encoder_features(
    eeg_data_np: np.ndarray,
    vae_model,
    device,
    batch_size: int = 32,
    flatten_latent_seq: bool = False
) -> np.ndarray:
    """
    Encode EEG data into VAE latent features (mu).
    
    Args:
        eeg_data_np: NumPy array of shape (N, C, L) with EEG segments
        vae_model: Trained VAE model
        device: Device to run inference on
        batch_size: Batch size for encoding (default: 32)
        flatten_latent_seq: If True, flatten latent_channels * latent_seq_len
                           If False, average over latent_seq_len (default: False)
    
    Returns:
        NumPy array of features (N, D_feat) where:
        - D_feat = latent_channels * latent_seq_len if flatten_latent_seq=True
        - D_feat = latent_channels if flatten_latent_seq=False
    
    Example:
        >>> eeg_data = np.random.randn(100, 18, 3840)  # 100 samples
        >>> vae = DecoupledVAE().eval()
        >>> features = get_vae_encoder_features(eeg_data, vae, device='cuda')
        >>> print(features.shape)  # (100, 64) for VAE_LATENT_CHANNELS=64
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    vae_model.eval()
    all_features_list = []
    
    # Create simple dataset
    eeg_tensor = torch.from_numpy(eeg_data_np).float()
    temp_dataset = TensorDataset(eeg_tensor)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (segments_batch,) in temp_loader:
            segments_batch = segments_batch.to(device)
            
            # Encode to latent space
            encode_result = vae_model.encode(segments_batch)
            if isinstance(encode_result, (tuple, list)):
                mu = encode_result[0]  # Always take mu (first element)
            else:
                mu = encode_result
            
            # Process features
            if flatten_latent_seq:
                # Flatten: (B, C_latent, L_latent) -> (B, C_latent * L_latent)
                features_batch = mu.reshape(mu.size(0), -1).cpu().numpy()
            else:
                # Average: (B, C_latent, L_latent) -> (B, C_latent)
                features_batch = torch.mean(mu, dim=2).cpu().numpy()
            
            all_features_list.append(features_batch)
    
    if not all_features_list:
        return np.array([])
    
    return np.concatenate(all_features_list, axis=0)


# =============================================================================
# Contrastive Loss
# =============================================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.
    
    Used for contrastive learning in VAE training (teacher-student framework).
    
    Args:
        z1: First set of representations, shape (B, D) or (B, D, L)
        z2: Second set of representations, shape (B, D) or (B, D, L)
        temperature: Temperature parameter for scaling (default: 0.1)
    
    Returns:
        Scalar loss value
    
    Reference:
        Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
        ICML 2020 (SimCLR)
    """
    # Flatten to 2D if needed
    if z1.ndim == 3:
        z1 = z1.mean(dim=2)  # Average over sequence dimension
    if z2.ndim == 3:
        z2 = z2.mean(dim=2)  # Average over sequence dimension
    
    # Normalize representations
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate representations
    z = torch.cat([z1, z2], dim=0)  # Shape: (2B, D)
    
    # Calculate similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
    # Create labels and masks
    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size).to(z1.device)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    
    # Select negative samples (exclude self-similarity)
    logits = sim_matrix[~mask].view(2 * batch_size, -1) / temperature
    
    # Get positive pairs using indexing instead of diag()
    positives = torch.cat([
        sim_matrix[torch.arange(batch_size), torch.arange(batch_size) + batch_size],
        sim_matrix[torch.arange(batch_size) + batch_size, torch.arange(batch_size)]
    ]) / temperature
    
    # Combine into logits tensor for CrossEntropyLoss
    all_logits = torch.cat([positives.unsqueeze(1), logits], dim=1)
    
    # Target for all positive pairs is index 0
    loss_targets = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)
    
    return F.cross_entropy(all_logits, loss_targets)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Testing helper utilities...")
    
    # Test WarmupCosineAnnealingLR
    print("\n1. Testing WarmupCosineAnnealingLR...")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=100)
    
    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    assert lrs[0] < lrs[4], "LR should increase during warmup"
    assert lrs[50] > lrs[99], "LR should decrease during cosine phase"
    print(f"   LR at epoch 0: {lrs[0]:.6f}")
    print(f"   LR at epoch 5: {lrs[5]:.6f}")
    print(f"   LR at epoch 50: {lrs[50]:.6f}")
    print(f"   LR at epoch 99: {lrs[99]:.6f}")
    
    # Test Lookahead
    print("\n2. Testing Lookahead optimizer...")
    model = nn.Linear(10, 10)
    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = Lookahead(base_opt, k=6, alpha=0.5)
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    print("   Lookahead step completed successfully")
    
    # Test get_cosine_schedule
    print("\n3. Testing get_cosine_schedule...")
    betas = get_cosine_schedule(1000)
    assert betas.shape == (1000,), "Beta schedule shape mismatch"
    assert (betas >= 0.0001).all() and (betas <= 0.015).all(), "Beta values out of range"
    print(f"   Beta schedule: min={betas.min():.6f}, max={betas.max():.6f}")
    
    # Test NT-Xent loss
    print("\n4. Testing nt_xent_loss...")
    z1 = torch.randn(16, 128)
    z2 = torch.randn(16, 128)
    loss = nt_xent_loss(z1, z2)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"   NT-Xent loss: {loss.item():.4f}")
    
    print("\n[SUCCESS] All helper utility tests passed!")
