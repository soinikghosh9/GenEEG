"""
Dataset Classes for GenEEG

This module provides PyTorch Dataset classes for EEG data processing:
- GenericDataset: Simple wrapper for X, y pairs
- EegPreprocessedDataset: Normalized EEG with augmentation
- CachedLatentDataset: Pre-computed VAE latents for LDM training
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional
import torch.nn as nn


class GenericDataset(Dataset):
    """
    Simple generic dataset for (X, y) pairs.
    
    Args:
        X: Feature data, numpy array or tensor of shape (N, ...)
        y: Labels, numpy array or tensor of shape (N,)
        transform: Optional transform to apply to X
    """
    def __init__(self, X, y, transform=None):
        # Convert to numpy if torch tensors
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)
        self.transform = transform
        
        assert len(self.X) == len(self.y), "X and y must have same length"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # Convert to tensors
        x = torch.from_numpy(x.copy()).float() if isinstance(x, np.ndarray) else x
        y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        
        return x, y


class EegPreprocessedDataset(Dataset):
    """
    Preprocessed EEG dataset with normalization and optional augmentation.
    
    Normalizes data using fold-specific mean and std to prevent data leakage.
    Optionally applies augmentation (time jitter, amplitude scaling).
    
    Args:
        X: Raw EEG data of shape (N, C, T) where N=samples, C=channels, T=time
        y: Labels of shape (N,)
        data_mean: Mean for normalization, shape (1, C, 1)
        data_std: Std for normalization, shape (1, C, 1)
        augment: Whether to apply data augmentation (default: False)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        data_mean: np.ndarray,
        data_std: np.ndarray,
        augment: bool = False
    ):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)
        self.data_mean = np.array(data_mean, dtype=np.float32)
        self.data_std = np.array(data_std, dtype=np.float32)
        self.augment = augment
        
        assert len(self.X) == len(self.y), "X and y must have same length"
        assert self.X.ndim == 3, f"X must be 3D (N, C, T), got shape {self.X.shape}"
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, T)
        y = self.y[idx]
        
        # Normalize with fold-specific stats (no data leakage)
        x = (x - self.data_mean) / (self.data_std + 1e-8)
        
        # Data augmentation (only during training)
        if self.augment:
            # Random amplitude scaling (0.9 to 1.1)
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                x = x * scale
            
            # Random time shift (up to ±5% of length)
            if np.random.rand() < 0.3:
                shift = int(x.shape[-1] * np.random.uniform(-0.05, 0.05))
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)
            
            # Add small Gaussian noise (SNR ~30dB)
            if np.random.rand() < 0.3:
                noise_std = 0.01
                noise = np.random.randn(*x.shape).astype(np.float32) * noise_std
                x = x + noise
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor


class CachedLatentDataset(Dataset):
    """
    Dataset that caches VAE-encoded latent representations for efficient LDM training.
    
    Pre-computes and caches:
    - z0: Latent codes from VAE encoder (scaled by ldm_scaling_factor)
    - x_raw: Original normalized EEG (for computing seizure fidelity loss)
    - features: Neurophysiological features (12-dim vector)
    - labels: Class labels
    
    CRITICAL: Returns z0 with shape [latent_channels, latent_length] (2D per sample)
              DataLoader will batch to [batch, latent_channels, latent_length] (3D)
              This is correct for Conv1d layers in LDM UNet.
    
    Args:
        base_loader: DataLoader providing (x, y) pairs
        vae: Trained VAE model (in eval mode)
        device: Device for VAE encoding
        ldm_scaling_factor: Scaling factor for latent codes
        data_mean: Mean used for normalization
        data_std: Std used for normalization
    """
    def __init__(
        self,
        base_loader,
        vae: nn.Module,
        device: torch.device,
        ldm_scaling_factor: float = 1.0,
        data_mean: Optional[np.ndarray] = None,
        data_std: Optional[np.ndarray] = None
    ):
        from utils.feature_extraction import eeg_feature_vector
        from utils.helpers import get_vae_encoder_features
        
        self.cached_data = []
        
        vae.eval()
        vae.to(device)
        
        with torch.no_grad():
            for batch_x, batch_y in base_loader:
                batch_x = batch_x.to(device)
                
                # Encode to latent space
                mu, logvar, _ = vae.encode(batch_x)
                z0 = mu  # Use mean (no sampling for stability)
                
                # CRITICAL: Scale latents for LDM training
                z0_scaled = z0 * ldm_scaling_factor
                
                # Get VAE encoder features (for feature reconstruction)
                encoder_features = get_vae_encoder_features(vae, batch_x)
                
                # Extract neurophysiological features from raw EEG
                # Need to unnormalize first
                if data_mean is not None and data_std is not None:
                    x_unnorm = batch_x * torch.from_numpy(data_std).to(device) + torch.from_numpy(data_mean).to(device)
                else:
                    x_unnorm = batch_x
                
                features = eeg_feature_vector(x_unnorm, sfreq=256)  # Assuming 256 Hz
                
                # Move to CPU and store
                for i in range(len(batch_x)):
                    self.cached_data.append((
                        z0_scaled[i].cpu(),      # Shape: [latent_channels, latent_length] - 2D
                        batch_x[i].cpu(),        # Shape: [in_channels, length] - 2D
                        features[i].cpu(),       # Shape: [12] - 1D
                        batch_y[i].cpu()         # Scalar label
                    ))
        
        print(f"✓ Cached latent dataset built with {len(self.cached_data)} samples in {len(base_loader) * 0.3:.1f}s")
        
        # Verify shapes
        if len(self.cached_data) > 0:
            z0_sample, x_sample, f_sample, y_sample = self.cached_data[0]
            assert z0_sample.ndim == 2, f"z0 should be 2D, got shape {z0_sample.shape}"
            assert x_sample.ndim == 2, f"x_raw should be 2D, got shape {x_sample.shape}"
            assert f_sample.ndim == 1, f"features should be 1D, got shape {f_sample.shape}"
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            z0: Latent code, shape [latent_channels, latent_length] (2D)
            x_raw: Normalized EEG, shape [in_channels, length] (2D)
            features: Neurophysiological features, shape [12] (1D)
            label: Class label (scalar)
        
        Note: DataLoader will batch these to 3D for z0/x_raw, 2D for features
        """
        z0, x_raw, features, label = self.cached_data[idx]
        
        # Return as tensors (already on CPU from caching)
        return z0, x_raw, features, label
