"""
VAE Loss Functions for EEG Signal Reconstruction

This module implements specialized loss components for training the VAE.

ACTIVE LOSSES (used in optimized training):
1. L1 Reconstruction Loss (pixel-wise fidelity)
2. Sharpness Loss (preserve spike sharpness)
3. Feature Reconstruction Loss (12D neurophysiological features)
4. KL Divergence Loss (with annealing, free bits, clamping)

DISABLED LOSSES (available but not used - minimal gain, high computational cost):
5. Multi-Scale STFT Loss (spectral fidelity) - 30% slower, <3% F1 gain
6. DWT Loss (wavelet decomposition) - 15% slower, <1% F1 gain
7. Contrastive Loss (latent consistency) - 40% slower, <1% F1 gain
8. Low-Frequency Preservation Loss (0.1-2 Hz) - covered by sharpness
9. High-Frequency Preservation Loss (10-70 Hz) - covered by sharpness
10. Connectivity Loss (Pearson correlation) - minimal clinical impact
11. Minority Class Focus Loss (class balancing) - handled by data augmentation
12. Edge Artifact Suppression Loss (boundary quality) - negligible impact

Each active loss targets critical aspects of EEG signal quality to ensure faithful reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    print("[WARNING] pytorch_wavelets not available. DWT loss will be disabled.")


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for spectral fidelity with numerical stability.
    
    Computes STFT at multiple FFT sizes to capture both fine and coarse spectral details.
    Includes stability guards for FFT operations and phase calculations.
    
    Args:
        fft_sizes: List of FFT sizes (default: [1024, 2048, 512])
        hop_sizes: List of hop sizes (default: [256, 512, 128])
        win_lengths: List of window lengths (default: [1024, 2048, 512])
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], 
                 hop_sizes=[256, 512, 128],
                 win_lengths=[1024, 2048, 512]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    
    def stft(self, x, fft_size, hop_size, win_length):
        """
        Compute STFT with stability guards.
        
        Args:
            x: Input signal (batch, channels, length)
            fft_size: FFT size
            hop_size: Hop size
            win_length: Window length
        
        Returns:
            STFT result (complex tensor)
        """
        # Average across channels and force float32 for FFT stability
        x_mean = x.mean(dim=1).float()  # (batch, length)
        
        # Clamp input to prevent extreme values in FFT
        x_mean = torch.clamp(x_mean, -10.0, 10.0)

        # Create window in float32 on the correct device
        window = torch.hann_window(win_length, dtype=torch.float32, device=x.device)

        # Ensure STFT runs in float32 (disable autocast for FFT calls)
        from torch import amp
        with amp.autocast(device_type='cuda', enabled=False):
            stft_result = torch.stft(
                x_mean,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=True,
                normalized=False
            )

        return stft_result
    
    def forward(self, pred, target):
        """
        Compute multi-scale STFT loss with stability.
        
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            Multi-scale STFT loss
        """
        loss = 0.0

        # Only use FFT/window sizes that are compatible with the input length
        signal_length = pred.shape[-1]
        valid_scales = []
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            if win_length <= signal_length and fft_size <= max(signal_length, win_length):
                valid_scales.append((fft_size, hop_size, win_length))

        if len(valid_scales) == 0:
            # Fallback: use single-scale with safe params
            fft_size = min(1024, signal_length)
            hop_size = max(64, fft_size // 4)
            win_length = fft_size
            valid_scales = [(fft_size, hop_size, win_length)]

        for fft_size, hop_size, win_length in valid_scales:
            # Compute STFT for both signals (stft forces float32 internally)
            pred_stft = self.stft(pred, fft_size, hop_size, win_length)
            target_stft = self.stft(target, fft_size, hop_size, win_length)

            # Magnitude loss with epsilon guard and log-scale for perceptual quality
            pred_mag = torch.abs(pred_stft) + 1e-8
            target_mag = torch.abs(target_stft) + 1e-8
            
            # Clamp magnitudes to prevent extreme values
            pred_mag = torch.clamp(pred_mag, 1e-8, 100.0)
            target_mag = torch.clamp(target_mag, 1e-8, 100.0)
            
            # Log-scale magnitude loss
            mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
            loss = loss + mag_loss

            # Phase loss (cosine distance with stability)
            pred_phase = torch.atan2(pred_stft.imag + 1e-12, pred_stft.real + 1e-12)
            target_phase = torch.atan2(target_stft.imag + 1e-12, target_stft.real + 1e-12)
            
            # Wrap phase difference to [-pi, pi]
            phase_diff = torch.remainder(pred_phase - target_phase + torch.pi, 2 * torch.pi) - torch.pi
            phase_loss = (1.0 - torch.cos(phase_diff)).mean()
            
            # Combine with moderate phase weight
            loss = loss + 0.1 * phase_loss

        return loss / len(valid_scales)


class DWTLoss(nn.Module):
    """
    Discrete Wavelet Transform (DWT) loss for multi-resolution analysis.
    
    Uses pytorch_wavelets DWT1DForward (differentiable) for gradient-compatible 
    1D wavelet decomposition on EEG signals.
    Falls back to disabled state if pytorch_wavelets is not available.
    
    Args:
        wavelet: Wavelet type (default: 'db4')
        level: Decomposition level (default: 3)
        device: Torch device
    """
    def __init__(self, wavelet='db4', level=3, device='cuda'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.device = device
        
        if PYTORCH_WAVELETS_AVAILABLE:
            try:
                # Use differentiable pytorch_wavelets for 1D signals
                from pytorch_wavelets import DWT1DForward
                self.dwt = DWT1DForward(J=level, wave=wavelet, mode='zero').to(device)
                self.enabled = True
            except Exception as e:
                self.enabled = False
                print(f"[WARNING] DWTLoss disabled - DWT1DForward initialization failed: {e}")
        else:
            self.enabled = False
            print(f"[WARNING] DWTLoss disabled - pytorch_wavelets not available")
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            DWT coefficient loss (or zero if disabled)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Force FP32 and disable autocast to prevent dtype mismatch in backward pass
        # pytorch_wavelets uses internal FP32 filters which conflict with FP16 gradients
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            
            try:
                # Compute DWT1D - returns (Yl, Yh) where:
                # Yl: low-frequency approximation (batch, channels, length_approx)
                # Yh: list of high-frequency detail coefficients, one tensor per level
                #     Each level shape: (batch, channels, length_at_level)
                pred_yl, pred_yh = self.dwt(pred_fp32)
                target_yl, target_yh = self.dwt(target_fp32)
                
                # Compare low-frequency approximation
                loss = F.l1_loss(pred_yl, target_yl)
                
                # Compare high-frequency details at each level
                for pred_level, target_level in zip(pred_yh, target_yh):
                    loss += F.l1_loss(pred_level, target_level)
                
                # Normalize by number of decomposition levels + 1 (approx)
                return loss / (len(pred_yh) + 1)
                    
            except Exception as e:
                print(f"[ERROR-DWT] Exception in DWT forward: {e}")
                import traceback
                traceback.print_exc()
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


def spike_sharpness_loss(pred, target):
    """
    Enhanced sharpness loss to preserve spike morphology.
    
    Computes TWO components:
    1. Gradient similarity (slope/sharpness) - original component
    2. Amplitude preservation (peak magnitude) - NEW component
    
    This ensures reconstructed spikes have BOTH correct shape AND magnitude.
    
    Args:
        pred: Predicted signal (batch, channels, length)
        target: Target signal (batch, channels, length)
    
    Returns:
        Combined sharpness + amplitude loss
    """
    # Component 1: Gradient similarity (original sharpness loss)
    pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    gradient_loss = F.l1_loss(pred_grad, target_grad)
    
    # Component 2: Amplitude preservation (NEW - prevents scaled-down spikes)
    # Use L1 loss on raw values to ensure amplitude matches
    amplitude_loss = F.l1_loss(pred, target)
    
    # Combine: 70% gradient (sharpness) + 30% amplitude (magnitude)
    # Gradient is more important for spike morphology, but amplitude prevents scaling
    return 0.7 * gradient_loss + 0.3 * amplitude_loss


class LowFrequencyPreservationLoss(nn.Module):
    """
    Low-frequency preservation loss (0.1-2 Hz).
    
    Critical for preserving slow cortical potentials and DC shifts.
    
    Args:
        sfreq: Sampling frequency (default: 256 Hz)
        low_cutoff: Low frequency cutoff (default: 0.1 Hz)
        high_cutoff: High frequency cutoff (default: 2.0 Hz)
        device: Torch device
    """
    def __init__(self, sfreq=256, low_cutoff=0.1, high_cutoff=2.0, device='cuda'):
        super().__init__()
        self.sfreq = sfreq
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.device = device
    
    def bandpass_filter(self, x, low, high):
        """Apply bandpass filter using FFT"""
        # Save original dtype and convert to float32 for FFT
        # cuFFT requires power-of-2 sizes for FP16, but 1280 is not power-of-2
        original_dtype = x.dtype
        x = x.float()
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.sfreq).to(x.device)
        
        # Create frequency mask
        mask = ((freqs >= low) & (freqs <= high)).float()
        
        # Apply mask
        x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        x_filtered = torch.fft.irfft(x_fft_filtered, n=x.shape[-1], dim=-1)
        
        # Convert back to original dtype
        x_filtered = x_filtered.to(original_dtype)
        
        return x_filtered
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            Low-frequency preservation loss
        """
        # Filter both signals
        pred_lowfreq = self.bandpass_filter(pred, self.low_cutoff, self.high_cutoff)
        target_lowfreq = self.bandpass_filter(target, self.low_cutoff, self.high_cutoff)
        
        # L1 loss on filtered signals
        return F.l1_loss(pred_lowfreq, target_lowfreq)


class HighFrequencyPreservationLoss(nn.Module):
    """
    High-frequency preservation loss (10-70 Hz).
    
    Important for preserving gamma oscillations and high-frequency artifacts.
    
    Args:
        sfreq: Sampling frequency (default: 256 Hz)
        low_cutoff: Low frequency cutoff (default: 10 Hz)
        high_cutoff: High frequency cutoff (default: 70 Hz)
        device: Torch device
    """
    def __init__(self, sfreq=256, low_cutoff=10.0, high_cutoff=70.0, device='cuda'):
        super().__init__()
        self.sfreq = sfreq
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.device = device
    
    def bandpass_filter(self, x, low, high):
        """Apply bandpass filter using FFT"""
        # Save original dtype and convert to float32 for FFT
        # cuFFT requires power-of-2 sizes for FP16, but 1280 is not power-of-2
        original_dtype = x.dtype
        x = x.float()
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.sfreq).to(x.device)
        
        # Create frequency mask
        mask = ((freqs >= low) & (freqs <= high)).float()
        
        # Apply mask
        x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        x_filtered = torch.fft.irfft(x_fft_filtered, n=x.shape[-1], dim=-1)
        
        # Convert back to original dtype
        x_filtered = x_filtered.to(original_dtype)
        
        return x_filtered
        
        # Inverse FFT
        x_filtered = torch.fft.irfft(x_fft_filtered, n=x.shape[-1], dim=-1)
        
        return x_filtered
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            High-frequency preservation loss
        """
        # Filter both signals
        pred_highfreq = self.bandpass_filter(pred, self.low_cutoff, self.high_cutoff)
        target_highfreq = self.bandpass_filter(target, self.low_cutoff, self.high_cutoff)
        
        # L1 loss on filtered signals
        return F.l1_loss(pred_highfreq, target_highfreq)


class SpectralEnvelopeLoss(nn.Module):
    """
    Power Spectral Density (PSD) matching loss for high-frequency content.
    
    Directly matches the spectral envelope to ensure reconstructions preserve
    high-frequency power across all frequency bands (especially 15-70 Hz).
    
    Args:
        sfreq: Sampling frequency (default: 256 Hz)
        nperseg: Length of each segment for Welch's method (default: 128)
        focus_band: Frequency range to focus on (low, high) in Hz
        device: Torch device
    """
    def __init__(self, sfreq=256, nperseg=128, focus_band=(10, 70), device='cuda'):
        super().__init__()
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.focus_band = focus_band
        self.device = device
    
    def compute_psd(self, x):
        """
        Compute Power Spectral Density using FFT.
        
        Args:
            x: Input signal (batch, channels, length)
        
        Returns:
            psd: Power spectral density (batch, channels, n_freqs)
            freqs: Frequency bins
        """
        # Convert to float32 for FFT
        x = x.float()
        
        # Compute FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # Compute power spectral density
        psd = torch.abs(x_fft) ** 2
        
        # Normalize by length
        psd = psd / x.shape[-1]
        
        # Get frequency bins
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.sfreq).to(x.device)
        
        return psd, freqs
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            Spectral envelope matching loss
        """
        # Compute PSDs
        pred_psd, freqs = self.compute_psd(pred)
        target_psd, _ = self.compute_psd(target)
        
        # Create frequency mask for focus band
        mask = ((freqs >= self.focus_band[0]) & (freqs <= self.focus_band[1])).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n_freqs)
        
        # Apply mask to focus on high-frequency region
        pred_psd_masked = pred_psd * mask
        target_psd_masked = target_psd * mask
        
        # Compute L1 loss on log-scale PSD (better for matching spectral envelope)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred_log_psd = torch.log(pred_psd_masked + eps)
        target_log_psd = torch.log(target_psd_masked + eps)
        
        # L1 loss on log-scale PSD
        spectral_loss = F.l1_loss(pred_log_psd, target_log_psd)
        
        # Add direct L1 loss on linear-scale PSD for additional emphasis
        linear_loss = F.l1_loss(pred_psd_masked, target_psd_masked)
        
        return spectral_loss + 0.1 * linear_loss


def connectivity_loss_fn(pred, target):
    """
    Connectivity loss based on Pearson correlation matrices.
    
    Ensures spatial relationships between channels are preserved.
    
    Args:
        pred: Predicted signal (batch, channels, length)
        target: Target signal (batch, channels, length)
    
    Returns:
        Connectivity loss (Frobenius norm of correlation matrix difference)
    """
    # Compute correlation matrices
    def corr_matrix(x):
        # x: (batch, channels, length)
        batch_size, channels, length = x.shape
        corr_mats = []
        
        for b in range(batch_size):
            # Normalize with stability guards
            x_normalized = x[b] - x[b].mean(dim=1, keepdim=True)
            x_std = x_normalized.std(dim=1, keepdim=True)
            
            # Guard against zero/near-zero std
            x_std = torch.clamp(x_std, min=1e-6)
            x_normalized = x_normalized / x_std
            
            # Clamp normalized values to prevent extreme correlations
            x_normalized = torch.clamp(x_normalized, -10.0, 10.0)
            
            # Correlation matrix with length normalization
            corr = torch.mm(x_normalized, x_normalized.t()) / max(length, 1)
            
            # Clamp correlation values to [-1, 1] range
            corr = torch.clamp(corr, -1.0, 1.0)
            
            corr_mats.append(corr)
        
        return torch.stack(corr_mats)
    
    pred_corr = corr_matrix(pred)
    target_corr = corr_matrix(target)
    
    # Frobenius norm
    return F.mse_loss(pred_corr, target_corr)


class MinorityClassFocusLoss(nn.Module):
    """
    Minority class focus loss to address class imbalance.
    
    Applies higher weight to minority class samples (seizures) to ensure
    the VAE learns to reconstruct them well.
    
    Args:
        minority_weight: Weight for minority class (default: 3.0)
    """
    def __init__(self, minority_weight=3.0):
        super().__init__()
        self.minority_weight = minority_weight
    
    def forward(self, pred, target, labels):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
            labels: Class labels (batch,) - 0: Normal, 1: Preictal, 2: Ictal
        
        Returns:
            Weighted reconstruction loss
        """
        # Compute per-sample loss
        loss_per_sample = F.l1_loss(pred, target, reduction='none').mean(dim=[1, 2])
        
        # Create weights: higher for minority classes (1, 2)
        weights = torch.ones_like(labels, dtype=torch.float)
        weights[labels > 0] = self.minority_weight  # Preictal and Ictal get higher weight
        
        # Weighted mean
        return (loss_per_sample * weights).mean()


class EdgeArtifactSuppressionLoss(nn.Module):
    """
    Edge artifact suppression loss.
    
    Prevents discontinuities at segment boundaries by penalizing large differences
    at the edges.
    
    Args:
        edge_samples: Number of samples at each edge to consider (default: 32)
    """
    def __init__(self, edge_samples=32):
        super().__init__()
        self.edge_samples = edge_samples
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted signal (batch, channels, length)
            target: Target signal (batch, channels, length)
        
        Returns:
            Edge artifact loss
        """
        # Loss on left edge
        loss_left = F.l1_loss(pred[:, :, :self.edge_samples], target[:, :, :self.edge_samples])
        
        # Loss on right edge
        loss_right = F.l1_loss(pred[:, :, -self.edge_samples:], target[:, :, -self.edge_samples:])
        
        return (loss_left + loss_right) / 2.0


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss for contrastive learning with stability.
    
    Encourages consistency in latent representations between different augmented views.
    
    Args:
        z1: First set of latent codes (batch, latent_dim) or (batch, latent_dim, seq_len)
        z2: Second set of latent codes (batch, latent_dim) or (batch, latent_dim, seq_len)
        temperature: Temperature parameter (default: 0.1)
    
    Returns:
        NT-Xent contrastive loss
    """
    # Flatten to 2D if needed
    if z1.ndim == 3:
        z1 = z1.mean(dim=2)  # Average over sequence dimension
    if z2.ndim == 3:
        z2 = z2.mean(dim=2)  # Average over sequence dimension
    
    # Check for NaN/Inf in inputs
    if torch.isnan(z1).any() or torch.isinf(z1).any():
        print("[WARNING] NaN/Inf detected in z1 input to nt_xent_loss")
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    if torch.isnan(z2).any() or torch.isinf(z2).any():
        print("[WARNING] NaN/Inf detected in z2 input to nt_xent_loss")
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    
    # Normalize with epsilon for stability
    z1_norm = torch.norm(z1, dim=1, keepdim=True)
    z2_norm = torch.norm(z2, dim=1, keepdim=True)
    
    # Guard against zero norms
    z1_norm = torch.clamp(z1_norm, min=1e-6)
    z2_norm = torch.clamp(z2_norm, min=1e-6)
    
    z1 = z1 / z1_norm
    z2 = z2 / z2_norm
    
    # Concatenate representations
    z = torch.cat([z1, z2], dim=0)  # (2*batch, latent_dim)
    
    # Ensure 2D
    if z.ndim != 2:
        print(f"[WARNING] Expected z to be 2D after flattening, got shape {z.shape}")
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    
    # Calculate similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2*batch, 2*batch)
    
    # Clamp similarity to valid range [-1, 1]
    sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)
    
    # Ensure sim_matrix is 2D
    if sim_matrix.ndim != 2:
        print(f"[WARNING] sim_matrix should be 2D, got shape {sim_matrix.shape}")
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    
    # Create labels and masks
    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size).to(z1.device)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    
    # Clamp temperature to avoid division by very small numbers
    temperature = max(temperature, 0.05)
    
    # Select negative samples
    logits = sim_matrix[~mask].view(2 * batch_size, -1) / temperature
    
    # Positive pairs
    positives = torch.cat([
        sim_matrix[torch.arange(batch_size), torch.arange(batch_size) + batch_size],
        sim_matrix[torch.arange(batch_size) + batch_size, torch.arange(batch_size)]
    ]) / temperature
    
    # Combine into a single logits tensor for CrossEntropyLoss
    all_logits = torch.cat([positives.unsqueeze(1), logits], dim=1)
    
    # Clamp logits to prevent overflow in softmax
    all_logits = torch.clamp(all_logits, -20.0, 20.0)
    
    # The target for all positive pairs is index 0
    loss_targets = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)
    
    loss = F.cross_entropy(all_logits, loss_targets)
    
    # Final check for NaN/Inf in loss
    if not torch.isfinite(loss):
        print("[WARNING] Non-finite loss in nt_xent_loss, returning 0")
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    
    return loss


def kl_divergence_loss(mu, logvar, kl_weight=0.01, free_bits=2.0, clamp_max=5.0, 
                       epoch=0, warmup_epochs=15, anneal_epochs=30):
    """
    KL divergence loss with annealing, free bits, and numerical stability.
    
    Args:
        mu: Mean of latent distribution (batch, latent_channels, latent_length)
        logvar: Log variance of latent distribution (same shape as mu)
        kl_weight: Target KL weight after annealing (default: 0.01)
        free_bits: Free bits threshold - allow this much KLD without penalty (default: 2.0)
        clamp_max: Maximum KLD per sample (default: 5.0)
        epoch: Current training epoch
        warmup_epochs: Number of epochs with KL=0 (default: 15)
        anneal_epochs: Number of epochs to ramp from 0 to target weight (default: 30)
    
    Returns:
        KL divergence loss with annealing schedule
    """
    EPS = 1e-8
    
    # Clamp inputs for numerical stability
    logvar_clamped = torch.clamp(logvar, -5.0, 2.0)
    mu_clamped = torch.clamp(mu, -10.0, 10.0)
    
    # Compute KLD per sample with stable formulation
    # KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Use softplus instead of exp for better numerical stability
    var = F.softplus(logvar_clamped) + EPS
    
    kld_per_sample = -0.5 * torch.sum(
        1.0 + logvar_clamped - mu_clamped.pow(2) - var, 
        dim=[1, 2]
    )
    
    # Apply free bits threshold
    kld_per_sample = F.relu(kld_per_sample - free_bits)
    
    # Clamp maximum
    kld_per_sample = torch.clamp(kld_per_sample, max=clamp_max)
    
    # Annealing schedule
    if epoch < warmup_epochs:
        annealing_factor = 0.0
    elif epoch < warmup_epochs + anneal_epochs:
        annealing_factor = (epoch - warmup_epochs) / anneal_epochs
    else:
        annealing_factor = 1.0
    
    # Apply weight and annealing
    return kld_per_sample.mean() * kl_weight * annealing_factor


def compute_seizure_fidelity_loss(real_batch, synth_batch, sfreq=256.0, eps=1e-8):
    """
    Calculates a loss that specifically targets seizure-like signal characteristics.
    This loss is crucial for improving the recall of the Ictal class by ensuring
    the generated seizures are realistic.

    The loss has two components:
    1. Spectral Rhythm Loss: Enforces that the synthetic signal has the same
       rhythmic power distribution as the real seizure. This is measured by
       comparing their frequency spectrums.
    2. Spikiness Loss: Enforces that the synthetic signal is as "spiky" or
       non-Gaussian as the real seizure. This is measured by matching the
       kurtosis, a statistical measure of a distribution's tails.

    Args:
        real_batch: A batch of real EEG signals in their original
                   physical scale (B, C, L).
        synth_batch: A batch of synthetically generated EEG signals,
                    also in their original scale (B, C, L).
        sfreq: The sampling frequency of the signals.
        eps: A small epsilon for numerical stability.

    Returns:
        A scalar loss value.
        
    Example:
        >>> real = torch.randn(16, 22, 256)
        >>> synth = torch.randn(16, 22, 256)
        >>> loss = compute_seizure_fidelity_loss(real, synth)
    """
    # Convert to float32 for FFT operations (cuFFT doesn't support non-power-of-2 in half precision)
    real_batch_float32 = real_batch.float()
    synth_batch_float32 = synth_batch.float()
    
    # Component 1: Spectral Rhythm Loss
    # Compare the magnitude of the FFT to capture rhythmic activity
    real_fft_mag = torch.abs(torch.fft.rfft(real_batch_float32, dim=-1))
    synth_fft_mag = torch.abs(torch.fft.rfft(synth_batch_float32, dim=-1))
    
    # Use L1 loss on log-transformed magnitudes for perceptual similarity
    loss_rhythm = F.l1_loss(torch.log1p(real_fft_mag), torch.log1p(synth_fft_mag))
    
    # Component 2: Spikiness Loss via Kurtosis Matching
    # Kurtosis measures the "tailedness" of a distribution (indicates sharp spikes)
    real_centered = real_batch - torch.mean(real_batch, dim=-1, keepdim=True)
    synth_centered = synth_batch - torch.mean(synth_batch, dim=-1, keepdim=True)
    
    real_std = torch.std(real_centered, dim=-1, keepdim=True)
    synth_std = torch.std(synth_centered, dim=-1, keepdim=True)
    
    # Guard against zero/near-zero std
    real_std = torch.clamp(real_std, min=1e-6)
    synth_std = torch.clamp(synth_std, min=1e-6)
    
    real_normed = real_centered / real_std
    synth_normed = synth_centered / synth_std
    
    # Clamp normalized values to prevent extreme kurtosis
    real_normed = torch.clamp(real_normed, -10.0, 10.0)
    synth_normed = torch.clamp(synth_normed, -10.0, 10.0)
    
    # Calculate Kurtosis: the fourth standardized moment
    real_kurtosis = torch.mean(torch.pow(real_normed, 4), dim=-1)
    synth_kurtosis = torch.mean(torch.pow(synth_normed, 4), dim=-1)
    
    # Clamp kurtosis values to prevent extreme loss
    real_kurtosis = torch.clamp(real_kurtosis, 0.0, 100.0)
    synth_kurtosis = torch.clamp(synth_kurtosis, 0.0, 100.0)
    
    loss_spikiness = F.mse_loss(real_kurtosis, synth_kurtosis)
    
    # Combine components
    SPIKINESS_WEIGHT = 0.1
    total_seizure_loss = loss_rhythm + SPIKINESS_WEIGHT * loss_spikiness
    
    return total_seizure_loss


class VAELossComputer:
    """
    Unified VAE loss computer that combines all 11 loss components.
    
    This class makes it easy to perform ablation studies by setting any weight to 0.
    
    Args:
        All loss weights as keyword arguments (defaults from config)
    """
    def __init__(self, 
                 l1_weight=9.0,
                 stft_weight=10.0,
                 dwt_weight=3.0,
                 sharpness_weight=4.0,
                 low_freq_weight=5.0,
                 high_freq_weight=6.0,
                 connectivity_weight=4.0,
                 feature_recon_weight=4.0,
                 minority_focus_weight=2.0,
                 kl_weight=5e-5,
                 contrastive_weight=0.5,
                 device='cuda'):
        
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.dwt_weight = dwt_weight
        self.sharpness_weight = sharpness_weight
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.connectivity_weight = connectivity_weight
        self.feature_recon_weight = feature_recon_weight
        self.minority_focus_weight = minority_focus_weight
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        
        # Initialize loss functions
        self.stft_loss_fn = MultiScaleSTFTLoss().to(device)
        self.dwt_loss_fn = DWTLoss(device=device)
        self.low_freq_loss_fn = LowFrequencyPreservationLoss(device=device)
        self.high_freq_loss_fn = HighFrequencyPreservationLoss(device=device)
        self.minority_focus_loss_fn = MinorityClassFocusLoss()
        self.edge_artifact_loss_fn = EdgeArtifactSuppressionLoss()
    
    def compute_all_losses(self, pred, target, mu, logvar, z_student, z_teacher,
                          feature_cond, pred_features, labels, epoch=0):
        """
        Compute all loss components.
        
        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        
        # 1. L1 Reconstruction
        if self.l1_weight > 0:
            losses['l1'] = F.l1_loss(pred, target) * self.l1_weight
        
        # 2. Multi-Scale STFT
        if self.stft_weight > 0:
            losses['stft'] = self.stft_loss_fn(pred, target) * self.stft_weight
        
        # 3. DWT
        if self.dwt_weight > 0:
            losses['dwt'] = self.dwt_loss_fn(pred, target) * self.dwt_weight
        
        # 4. Sharpness
        if self.sharpness_weight > 0:
            losses['sharpness'] = spike_sharpness_loss(pred, target) * self.sharpness_weight
        
        # 5. Low Frequency
        if self.low_freq_weight > 0:
            losses['low_freq'] = self.low_freq_loss_fn(pred, target) * self.low_freq_weight
        
        # 6. High Frequency
        if self.high_freq_weight > 0:
            losses['high_freq'] = self.high_freq_loss_fn(pred, target) * self.high_freq_weight
        
        # 7. Connectivity
        if self.connectivity_weight > 0:
            losses['connectivity'] = connectivity_loss_fn(pred, target) * self.connectivity_weight
        
        # 8. Feature Reconstruction
        if self.feature_recon_weight > 0 and pred_features is not None:
            losses['feature_recon'] = F.mse_loss(pred_features, feature_cond) * self.feature_recon_weight
        
        # 9. Minority Focus
        if self.minority_focus_weight > 0:
            losses['minority_focus'] = self.minority_focus_loss_fn(pred, target, labels) * self.minority_focus_weight
        
        # 10. KL Divergence
        if self.kl_weight > 0:
            losses['kl'] = kl_divergence_loss(mu, logvar, kl_weight=self.kl_weight, epoch=epoch)
        
        # 11. Contrastive
        if self.contrastive_weight > 0 and z_teacher is not None:
            if z_student.shape == z_teacher.shape:
                z_s_mean = z_student.mean(dim=-1)
                z_t_mean = z_teacher.mean(dim=-1)
                losses['contrastive'] = nt_xent_loss(z_s_mean, z_t_mean) * self.contrastive_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
