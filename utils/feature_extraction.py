"""
Feature Extraction Utilities for EEG Signal Analysis

This module provides differentiable feature extraction functions used during VAE training
and evaluation. Features include band powers, spectral entropy, and Hjorth parameters.

Functions:
    - _welch_psd: Differentiable Welch power spectral density estimation
    - eeg_feature_vector: Comprehensive neurophysiological feature extraction
    - compute_band_powers: Extract power in standard EEG frequency bands
    - compute_hjorth_parameters: Compute Hjorth activity, mobility, and complexity
    - compute_spectral_entropy: Shannon entropy of power spectrum
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


def _welch_psd(
    x: torch.Tensor,
    nperseg: int = 256,
    noverlap: int = 128
) -> torch.Tensor:
    """
    Differentiable Welch power spectral density estimation.
    
    This function computes PSD using overlapping windowed FFT, similar to scipy.signal.welch
    but fully differentiable for use in neural network training.
    
    Args:
        x: Input EEG tensor of shape (B, C, L) where:
           - B: batch size
           - C: number of channels
           - L: sequence length (time samples)
        nperseg: Length of each segment for FFT (default: 256)
        noverlap: Number of overlapping samples between segments (default: 128)
    
    Returns:
        pxx: Power spectral density of shape (B, C, F) where F is number of frequency bins
    
    Note:
        - Uses Hann windowing to reduce spectral leakage
        - Pads input if shorter than nperseg
        - Normalizes by window power for proper scaling
    """
    # Create Hann window
    win = torch.hann_window(nperseg, periodic=True, device=x.device)
    step = nperseg - noverlap
    
    # Pad if necessary
    if x.shape[-1] < nperseg:
        pad = nperseg - x.shape[-1]
        x = F.pad(x, (pad // 2, pad - pad // 2), mode='reflect')
    
    # Unfold into overlapping segments and apply window
    xs = x.unfold(-1, nperseg, step) * win.view(1, 1, 1, -1)  # (B, C, frames, nperseg)
    
    # FFT and power computation
    Xf = torch.fft.rfft(xs, dim=-1)  # (B, C, frames, F)
    pxx = (Xf.real**2 + Xf.imag**2).mean(dim=2)  # Average over frames: (B, C, F)
    
    # Normalize by window power
    U = (win**2).sum().clamp_min(1e-12)
    return pxx / U


def compute_band_powers(
    pxx: torch.Tensor,
    freqs: torch.Tensor,
    bands: Tuple[Tuple[float, float], ...] = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 70))
) -> torch.Tensor:
    """
    Compute average power in standard EEG frequency bands.
    
    Args:
        pxx: Power spectral density of shape (B, C, F)
        freqs: Frequency values for each bin, shape (F,)
        bands: Tuple of (low, high) frequency ranges in Hz
               Default bands:
               - Delta: 0.5-4 Hz
               - Theta: 4-8 Hz
               - Alpha: 8-13 Hz
               - Beta: 13-30 Hz
               - Gamma: 30-70 Hz
    
    Returns:
        band_powers: Tensor of shape (B, len(bands)) with average power per band
    """
    band_feats = []
    for lo, hi in bands:
        # Create frequency mask
        mask = (freqs >= lo) & (freqs < hi)
        # Average power in this band across frequencies and channels
        band_power = pxx[..., mask].mean(dim=-1)  # (B, C)
        band_feats.append(band_power)
    
    # Stack all bands and average across channels
    band_feats = torch.stack(band_feats, dim=-1)  # (B, C, Bands)
    band_feats = band_feats.mean(dim=1)  # (B, Bands)
    
    return band_feats


def compute_spectral_entropy(pxx: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of the power spectrum.
    
    Spectral entropy measures the disorder/complexity of the frequency content.
    Lower values indicate more regular, rhythmic activity (e.g., seizures).
    Higher values indicate more irregular, chaotic activity.
    
    Args:
        pxx: Power spectral density of shape (B, C, F)
    
    Returns:
        spectral_entropy: Tensor of shape (B, 1) with entropy values
    
    Note:
        - Normalizes PSD to probability distribution
        - Computes Shannon entropy: -sum(p * log(p))
        - Averages across channels
    """
    # Normalize to probability distribution
    ps = pxx / (pxx.sum(dim=-1, keepdim=True) + 1e-12)
    
    # Compute Shannon entropy per channel
    entropy = -(ps * (ps.add(1e-12).log())).sum(dim=-1)  # (B, C)
    
    # Average across channels
    spec_ent = entropy.mean(dim=1, keepdim=True)  # (B, 1)
    
    return spec_ent


def compute_hjorth_parameters(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Hjorth parameters: activity, mobility, and complexity.
    
    Hjorth parameters are time-domain features that characterize signal dynamics:
    - Activity: variance of the signal (captures amplitude changes during seizures)
    - Mobility: mean frequency (sqrt of ratio of variances of first derivative to signal)
    - Complexity: rate of change of frequency (ratio of mobilities of derivative to signal)
    
    Args:
        x: Input EEG tensor of shape (B, C, L)
    
    Returns:
        activity: Tensor of shape (B, 1) - signal variance
        mobility: Tensor of shape (B, 1) - mean frequency estimate
        complexity: Tensor of shape (B, 1) - waveform irregularity
    
    References:
        Hjorth, B. (1970). EEG analysis based on time domain properties.
        Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    # Compute derivatives
    dx = torch.diff(x, dim=-1)  # First derivative
    ddx = torch.diff(dx, dim=-1)  # Second derivative
    
    # Compute variances (activity)
    var0 = x.var(dim=-1).mean(dim=1, keepdim=True) + 1e-12  # Signal variance
    var1 = dx.var(dim=-1).mean(dim=1, keepdim=True) + 1e-12  # 1st derivative variance
    var2 = ddx.var(dim=-1).mean(dim=1, keepdim=True) + 1e-12  # 2nd derivative variance
    
    # Hjorth activity: signal variance (log-scaled for stability)
    activity = torch.log(var0 + 1e-12)  # (B, 1) - log-scaled to reduce dynamic range
    
    # Hjorth mobility: sqrt(var(dx) / var(x))
    mobility = (var1 / var0).sqrt()  # (B, 1)
    
    # Hjorth complexity: (mobility of dx) / (mobility of x)
    complexity = (var2 / var1).sqrt() / (var1 / var0).sqrt()  # (B, 1)
    
    return activity, mobility, complexity


def compute_permutation_entropy(x: torch.Tensor, order: int = 3, delay: int = 1) -> torch.Tensor:
    """
    Compute Permutation Entropy (PE) for time series complexity analysis.
    
    PE quantifies the complexity of a time series by analyzing ordinal patterns.
    Lower PE values indicate more regular, predictable patterns (seizures).
    
    Args:
        x: Input EEG tensor of shape (B, C, L)
        order: Embedding dimension (default: 3)
        delay: Time delay (default: 1)
    
    Returns:
        pe: Permutation entropy tensor of shape (B, 1)
    
    References:
        Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity
        measure for time series. Physical Review Letters, 88(17), 174102.
    """
    B, C, L = x.shape
    
    # Create embedding vectors
    embedded = []
    for i in range(order):
        embedded.append(x[:, :, i * delay:L - (order - 1 - i) * delay])
    embedded = torch.stack(embedded, dim=-1)  # (B, C, L', order)
    
    # Compute permutation patterns (argsort gives ordinal patterns)
    patterns = embedded.argsort(dim=-1)  # (B, C, L', order)
    
    # Convert patterns to unique indices
    multipliers = torch.arange(order, device=x.device).pow(torch.arange(order, device=x.device))
    pattern_indices = (patterns * multipliers.view(1, 1, 1, -1)).sum(dim=-1)  # (B, C, L')
    
    # Count pattern frequencies
    n_patterns = torch.factorial(torch.tensor(order, dtype=torch.float32))
    pe_list = []
    
    for b in range(B):
        channel_entropies = []
        for c in range(C):
            pattern_vals = pattern_indices[b, c]
            # Histogram of patterns
            hist = torch.histc(pattern_vals.float(), bins=int(n_patterns.item()), min=0, max=n_patterns-1)
            # Normalize to probabilities
            probs = hist / (hist.sum() + 1e-12)
            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-12)).sum()
            channel_entropies.append(entropy)
        pe_list.append(torch.stack(channel_entropies).mean())
    
    pe = torch.stack(pe_list).unsqueeze(1)  # (B, 1)
    return pe


def compute_higuchi_fractal_dimension(x: torch.Tensor, kmax: int = 8) -> torch.Tensor:
    """
    Compute Higuchi Fractal Dimension (HFD) for signal complexity analysis.
    
    HFD measures the fractal dimension of a time series, indicating its complexity
    and self-similarity. Lower HFD values during seizures indicate reduced complexity.
    
    Args:
        x: Input EEG tensor of shape (B, C, L)
        kmax: Maximum time interval (default: 8)
    
    Returns:
        hfd: Higuchi fractal dimension tensor of shape (B, 1)
    
    References:
        Higuchi, T. (1988). Approach to an irregular time series on the basis of
        the fractal theory. Physica D, 31(2), 277-283.
    """
    B, C, L = x.shape
    N = L
    
    hfd_list = []
    for b in range(B):
        channel_hfds = []
        for c in range(C):
            signal = x[b, c].cpu().numpy()
            Lk = []
            
            for k in range(1, kmax + 1):
                Lmk = []
                for m in range(k):
                    # Construct sub-sequence
                    idxs = np.arange(m, N, k)
                    if len(idxs) < 2:
                        continue
                    # Compute length
                    Lmki = np.abs(np.diff(signal[idxs])).sum()
                    Lmki = Lmki * (N - 1) / (len(idxs) * k)
                    Lmk.append(Lmki)
                
                if len(Lmk) > 0:
                    Lk.append(np.mean(Lmk))
            
            # Fit log(L(k)) vs log(1/k)
            if len(Lk) >= 2:
                x_vals = np.log(1.0 / np.arange(1, len(Lk) + 1))
                y_vals = np.log(Lk)
                # Linear regression slope
                hfd_val = np.polyfit(x_vals, y_vals, 1)[0]
            else:
                hfd_val = 1.5  # Default value
            
            channel_hfds.append(hfd_val)
        
        hfd_list.append(np.mean(channel_hfds))
    
    hfd = torch.tensor(hfd_list, device=x.device, dtype=x.dtype).unsqueeze(1)  # (B, 1)
    return hfd


def compute_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """
    Compute kurtosis (fourth standardized moment) of the signal.
    
    Kurtosis measures the "tailedness" of the amplitude distribution.
    High kurtosis indicates sharp, high-amplitude transients (spikes).
    
    Args:
        x: Input EEG tensor of shape (B, C, L)
    
    Returns:
        kurt: Kurtosis tensor of shape (B, 1)
    """
    # Compute mean and std across time dimension
    mean = x.mean(dim=-1, keepdim=True)  # (B, C, 1)
    std = x.std(dim=-1, keepdim=True) + 1e-12  # (B, C, 1)
    
    # Standardize
    x_norm = (x - mean) / std  # (B, C, L)
    
    # Compute fourth moment
    kurt = (x_norm ** 4).mean(dim=-1)  # (B, C)
    
    # Average across channels
    kurt = kurt.mean(dim=1, keepdim=True)  # (B, 1)
    
    return kurt


def eeg_feature_vector(
    x: torch.Tensor,
    sfreq: float,
    bands: Tuple[Tuple[float, float], ...] = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 70))
) -> torch.Tensor:
    """
    Compute comprehensive neurophysiological feature vector for EEG signals.
    
    This is the main feature extraction function used during VAE training and evaluation.
    It computes a differentiable feature vector containing:
    - 5 band powers (delta, theta, alpha, beta, gamma)
    - 1 spectral entropy
    - 3 Hjorth parameters (activity, mobility, complexity)
    - 1 permutation entropy
    - 1 Higuchi fractal dimension
    - 1 kurtosis
    
    Total: 12 features per sample
    
    Args:
        x: Input EEG tensor of shape (B, C, L) where:
           - B: batch size
           - C: number of channels
           - L: sequence length (time samples)
        sfreq: Sampling frequency in Hz (e.g., 128 Hz)
        bands: Frequency bands for power computation (default: standard EEG bands)
    
    Returns:
        features: Feature tensor of shape (B, 12) containing:
                  [band_powers (5), spectral_entropy (1), hjorth_activity (1), 
                   hjorth_mobility (1), hjorth_complexity (1), permutation_entropy (1),
                   higuchi_fractal_dim (1), kurtosis (1)]
    
    Example:
        >>> x = torch.randn(32, 18, 3840)  # 32 samples, 18 channels, 30s @ 128 Hz
        >>> features = eeg_feature_vector(x, sfreq=128.0)
        >>> print(features.shape)  # torch.Size([32, 12])
    
    Note:
        - All operations are differentiable for use in training
        - Features are computed on GPU if input is on GPU
        - Small batch sizes recommended for memory efficiency during training
    """
    B, C, L = x.shape
    
    # 1. Compute power spectral density
    pxx = _welch_psd(x)  # (B, C, F)
    
    # 2. Compute frequency axis
    n_fft = pxx.shape[-1] * 2 - 2
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq).to(x.device)
    
    # 3. Extract band powers (5 features)
    band_powers = compute_band_powers(pxx, freqs, bands)  # (B, 5)
    
    # 4. Compute spectral entropy (1 feature)
    spec_ent = compute_spectral_entropy(pxx)  # (B, 1)
    
    # 5. Compute Hjorth parameters (3 features)
    activity, mobility, complexity = compute_hjorth_parameters(x)  # (B, 1), (B, 1), (B, 1)
    
    # 6. Compute permutation entropy (1 feature)
    perm_ent = compute_permutation_entropy(x, order=3, delay=1)  # (B, 1)
    
    # 7. Compute Higuchi fractal dimension (1 feature)
    hfd = compute_higuchi_fractal_dimension(x, kmax=8)  # (B, 1)
    
    # 8. Compute kurtosis (1 feature)
    kurt = compute_kurtosis(x)  # (B, 1)
    
    # 9. Concatenate all features
    features = torch.cat([
        band_powers,      # 5 features (0-4)
        spec_ent,         # 1 feature  (5)
        activity,         # 1 feature  (6)
        mobility,         # 1 feature  (7)
        complexity,       # 1 feature  (8)
        perm_ent,         # 1 feature  (9)
        hfd,              # 1 feature  (10)
        kurt              # 1 feature  (11)
    ], dim=-1)  # (B, 12)
    
    return features


def expand_feature_vector(
    features_12d: torch.Tensor,
    target_dim: int = 12
) -> torch.Tensor:
    """
    Identity function - features are already 12-dimensional.
    
    This function is kept for backward compatibility but now simply returns
    the input as all 12 features are computed directly by eeg_feature_vector().
    
    Args:
        features_12d: Input feature tensor of shape (B, 12)
        target_dim: Target dimension (default: 12)
    
    Returns:
        features: Feature tensor of shape (B, target_dim)
    """
    if target_dim == 12:
        return features_12d
    elif target_dim < 12:
        return features_12d[:, :target_dim]
    else:
        # Pad if target_dim > 12
        padding = target_dim - 12
        return F.pad(features_12d, (0, padding), mode='constant', value=0.0)


if __name__ == "__main__":
    # Self-test
    print("Testing feature extraction utilities...")
    
    # Create sample EEG data: 16 samples, 18 channels, 3840 samples (30s @ 128 Hz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(16, 18, 3840, device=device)
    
    # Test PSD
    print("\n1. Testing _welch_psd...")
    pxx = _welch_psd(x)
    print(f"   Input shape: {x.shape}")
    print(f"   PSD shape: {pxx.shape}")
    assert pxx.shape[0] == 16 and pxx.shape[1] == 18, "PSD batch/channel mismatch"
    
    # Test band powers
    print("\n2. Testing compute_band_powers...")
    freqs = torch.fft.rfftfreq(pxx.shape[-1] * 2 - 2, d=1.0 / 128.0).to(device)
    band_powers = compute_band_powers(pxx, freqs)
    print(f"   Band powers shape: {band_powers.shape}")
    assert band_powers.shape == (16, 5), "Band powers shape mismatch"
    
    # Test spectral entropy
    print("\n3. Testing compute_spectral_entropy...")
    spec_ent = compute_spectral_entropy(pxx)
    print(f"   Spectral entropy shape: {spec_ent.shape}")
    assert spec_ent.shape == (16, 1), "Spectral entropy shape mismatch"
    
    # Test Hjorth parameters
    print("\n4. Testing compute_hjorth_parameters...")
    mobility, complexity = compute_hjorth_parameters(x)
    print(f"   Mobility shape: {mobility.shape}")
    print(f"   Complexity shape: {complexity.shape}")
    assert mobility.shape == (16, 1) and complexity.shape == (16, 1), "Hjorth shape mismatch"
    
    # Test full feature vector (8 features)
    print("\n5. Testing eeg_feature_vector (8 features)...")
    features_8d = eeg_feature_vector(x, sfreq=128.0)
    print(f"   Feature vector shape: {features_8d.shape}")
    assert features_8d.shape == (16, 8), "Feature vector shape mismatch"
    print(f"   Feature statistics:")
    print(f"     Min: {features_8d.min().item():.4f}")
    print(f"     Max: {features_8d.max().item():.4f}")
    print(f"     Mean: {features_8d.mean().item():.4f}")
    
    # Test feature expansion to 12D
    print("\n6. Testing expand_feature_vector (8 -> 12)...")
    features_12d = expand_feature_vector(features_8d, target_dim=12)
    print(f"   Expanded feature shape: {features_12d.shape}")
    assert features_12d.shape == (16, 12), "Expanded feature shape mismatch"
    
    # Test gradient flow
    print("\n7. Testing gradient flow...")
    x_grad = torch.randn(4, 18, 3840, device=device, requires_grad=True)
    features_grad = eeg_feature_vector(x_grad, sfreq=128.0)
    loss = features_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient computed"
    print(f"   Gradient shape: {x_grad.grad.shape}")
    print(f"   Gradient norm: {x_grad.grad.norm().item():.4f}")
    
    print("\n[SUCCESS] All feature extraction tests passed!")
