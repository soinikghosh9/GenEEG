"""
Synthetic EEG Generation Utilities

Provides functions for generating synthetic EEG data using trained LDM.
Implements DDIM sampling for faster generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm

# Import feature extraction function
from data.preprocessing import calculate_eeg_features
from configs.dataset_config import TARGET_SFREQ
from utils.quality_filter import enhanced_filter_synthetic_batch


def extract_features_from_real_samples(
    vae_model,
    real_segments: np.ndarray,
    data_mean: np.ndarray,
    data_std: np.ndarray,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract neurophysiological features from real EEG segments.
    
    Uses the same feature extraction function as preprocessing to ensure consistency.
    Features are computed from RAW (original scale) EEG data.
    
    Args:
        vae_model: Trained VAE model (not used, kept for API compatibility)
        real_segments: Real EEG data (N, C, T) in ORIGINAL SCALE (not normalized)
        data_mean: Normalization mean (not used for feature extraction)
        data_std: Normalization std (not used for feature extraction)
        device: Device to use (not used for feature extraction)
    
    Returns:
        features: (N, 12) neurophysiological features
    """
    # Compute features from raw data (original scale)
    all_features = []
    for i in range(len(real_segments)):
        features = calculate_eeg_features(real_segments[i], sfreq=TARGET_SFREQ)
        all_features.append(features)
    
    return np.array(all_features)


def ddim_sampling(
    ldm_unet,
    latent_shape: Tuple[int, int, int],
    class_labels: torch.Tensor,
    feature_vectors: torch.Tensor,
    diffusion_timesteps: int = 1000,
    num_inference_steps: int = 50,
    eta: float = 0.5,  # OPTIMIZED: Increased from 0.3 to 0.5 for better diversity
    cfg_scale: float = 5.0,  # FIXED: Increased from 2.5 to 5.0 for stronger conditioning (matches original)
    device: str = 'cuda'
) -> torch.Tensor:
    """
    DDIM sampling for faster generation with controlled stochasticity and CFG.
    
    Args:
        ldm_unet: Trained LDM U-Net
        latent_shape: (batch_size, latent_channels, latent_time)
        class_labels: Class labels (batch_size,)
        feature_vectors: Neurophysiological features (batch_size, 12)
        diffusion_timesteps: Total diffusion steps (training)
        num_inference_steps: Actual sampling steps (< diffusion_timesteps)
        eta: Stochasticity (0=deterministic DDIM, 1=stochastic DDPM, 0.5=balanced diversity)
        cfg_scale: Classifier-free guidance scale (1.0=no guidance, 5.0=strong conditioning)
        device: Device
    
    Returns:
        latent: Generated latent codes (batch_size, latent_channels, latent_time)
    """
    ldm_unet.eval()
    
    batch_size = latent_shape[0]
    use_cfg = cfg_scale > 1.0
    
    # Start from pure noise
    latent = torch.randn(latent_shape, device=device)
    
    # DIAGNOSTIC: Check initial noise variance
    initial_noise_std = latent.std().item()
    print(f"    [DEBUG] Initial noise std: {initial_noise_std:.4f}, CFG scale: {cfg_scale}")
    
    # Create timestep schedule (subsample from full schedule)
    timesteps = torch.linspace(diffusion_timesteps - 1, 0, num_inference_steps).long().to(device)
    
    # Cosine schedule for alpha_bar
    def alpha_bar_fn(t):
        """Cosine schedule"""
        s = 0.008
        t_norm = t / diffusion_timesteps
        return torch.cos(((t_norm + s) / (1 + s)) * torch.pi * 0.5) ** 2
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling", leave=False)):
            # Current timestep
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get alpha values
            alpha_bar_t = alpha_bar_fn(t)
            
            if i < len(timesteps) - 1:
                alpha_bar_t_prev = alpha_bar_fn(timesteps[i + 1])
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)
            
            # DIAGNOSTIC: Check alpha values
            if i == 0:
                print(f"    [DEBUG] First timestep t={t.item():.0f}, alpha_bar_t={alpha_bar_t.item():.6f}, alpha_bar_t_prev={alpha_bar_t_prev.item():.6f}")
                print(f"    [DEBUG] Latent at start - mean: {latent.mean().item():.4f}, std: {latent.std().item():.4f}, range: [{latent.min().item():.2f}, {latent.max().item():.2f}]")
            
            # CRITICAL FIX: Apply Classifier-Free Guidance
            if use_cfg:
                # Run conditional and unconditional predictions in parallel
                latent_combined = torch.cat([latent, latent], dim=0)
                t_combined = torch.cat([t_batch, t_batch], dim=0)
                
                # Conditional: use real labels/features
                # Unconditional: use num_classes (reserved for uncond) and zero features
                labels_conditional = class_labels
                labels_unconditional = torch.full((batch_size,), ldm_unet.num_classes, 
                                                 device=device, dtype=torch.long)
                labels_combined = torch.cat([labels_conditional, labels_unconditional], dim=0)
                
                features_conditional = feature_vectors
                features_unconditional = torch.zeros_like(feature_vectors)
                features_combined = torch.cat([features_conditional, features_unconditional], dim=0)
                
                # Predict noise (both conditional and unconditional)
                noise_pred_combined = ldm_unet(latent_combined, t_combined, 
                                              labels_combined, features_combined)
                
                # Split predictions
                noise_pred_cond, noise_pred_uncond = noise_pred_combined.chunk(2, dim=0)
                
                # Apply CFG: interpolate between unconditional and conditional
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG: standard conditional prediction
                noise_pred = ldm_unet(latent, t_batch, class_labels, feature_vectors)
            
            # DIAGNOSTIC: Check if predictions vary
            if i == 0:
                pred_std = noise_pred.std().item()
                pred_mean = noise_pred.mean().item()
                print(f"    [DEBUG] noise_pred - mean: {pred_mean:.4f}, std: {pred_std:.4f}")
            
            # DDIM update rule with V-PREDICTION
            # V-prediction formula: v = sqrt(alpha_bar_t)*eps - sqrt(1-alpha_bar_t)*x0
            # Inverse to get x0: x0 = sqrt(alpha_bar_t)*z_t - sqrt(1-alpha_bar_t)*v_pred
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            # CORRECT V-PREDICTION FORMULA (not epsilon-prediction!)
            # Model outputs v_pred, we need to recover x0
            v_pred = noise_pred  # Rename for clarity - model outputs v, not epsilon
            pred_x0 = sqrt_alpha_bar_t * latent - sqrt_one_minus_alpha_bar_t * v_pred
            
            # DIAGNOSTIC: Check pred_x0 before clamping
            if i == 0:
                pred_x0_mean = pred_x0.mean().item()
                pred_x0_std = pred_x0.std().item()
                pred_x0_min = pred_x0.min().item()
                pred_x0_max = pred_x0.max().item()
                print(f"    [DEBUG] pred_x0 from v-prediction - mean: {pred_x0_mean:.2f}, std: {pred_x0_std:.2f}, range: [{pred_x0_min:.2f}, {pred_x0_max:.2f}]")
                
                # Check percentage that WOULD be clamped
                would_be_clamped = ((pred_x0 < -2.5) | (pred_x0 > 2.5)).sum().item()
                total = pred_x0.numel()
                print(f"    [DEBUG] Values outside [-2.5,2.5]: {would_be_clamped}/{total} ({would_be_clamped/total*100:.1f}%)")
            
            # OPTIMIZED: Tighter clamping for better VAE decoding
            # VAE latent space typically trained with values in [-2, 2] range
            pred_x0 = torch.clamp(pred_x0, -2.5, 2.5)
            
            # DIAGNOSTIC: Check pred_x0 after clamping
            if i == 0:
                pred_x0_after_mean = pred_x0.mean().item()
                pred_x0_after_std = pred_x0.std().item()
                print(f"    [DEBUG] pred_x0 AFTER clamp - mean: {pred_x0_after_mean:.2f}, std: {pred_x0_after_std:.2f}")
            
            # Compute direction for DDIM
            # For v-prediction, we need epsilon for the direction term
            # Recover epsilon from v and x0: eps = (v + sqrt(1-alpha_bar)*x0) / sqrt(alpha_bar)
            epsilon_recovered = (v_pred + sqrt_one_minus_alpha_bar_t * pred_x0) / (sqrt_alpha_bar_t + 1e-8)
            
            # DDIM direction term
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * epsilon_recovered
            
            # FIXED: Always add random noise for diversity (except last step)
            if i < len(timesteps) - 1:
                # CRITICAL FIX: Add temperature scaling for better diversity
                # Higher temperature = more variation between samples
                temperature = 1.2  # FIXED: Increased from 1.0 for better diversity
                noise = temperature * torch.randn_like(latent)
                noise_std = noise.std().item()
                if i == 0:
                    print(f"    [DEBUG] Adding noise with sigma_t={sigma_t.item():.4f}, noise_std={noise_std:.4f}, temperature={temperature}")
                latent = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                # Final step: no noise
                latent = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
    
    # DIAGNOSTIC: Check final latent variance
    final_latent_std = latent.std().item()
    print(f"    [DEBUG] Final latent std: {final_latent_std:.4f}")
    print(f"    [DEBUG] Latent range: [{latent.min().item():.2f}, {latent.max().item():.2f}]")

    
    return latent


def generate_synthetic_eeg_batch(
    ldm_unet,
    vae_model,
    class_label: int,
    feature_vectors: np.ndarray,
    n_samples: int,
    diffusion_timesteps: int = 1000,
    num_inference_steps: int = 50,
    device: str = 'cuda',
    ldm_scaling_factor: float = 1.0,
    data_mean: Optional[np.ndarray] = None,
    data_std: Optional[np.ndarray] = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate synthetic EEG samples for a specific class.
    
    Args:
        ldm_unet: Trained LDM U-Net
        vae_model: Trained VAE (for decoding)
        class_label: Target class (0=Normal, 1=Preictal, 2=Ictal)
        feature_vectors: Real features to condition on (N, 12)
        n_samples: Number of samples to generate
        diffusion_timesteps: Total diffusion steps
        num_inference_steps: Sampling steps (50 is good balance)
        device: Device
        ldm_scaling_factor: Latent scaling factor
        data_mean: For denormalization
        data_std: For denormalization
        batch_size: Generation batch size
    
    Returns:
        synthetic_eeg: Generated EEG (n_samples, C, T) - denormalized
    """
    ldm_unet.eval()
    vae_model.eval()
    
    all_synthetic = []
    
    # Generate in batches
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        actual_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # OPTIMIZED: Sample random features with variation for diversity
        # Sample base features from different real samples
        feature_indices = np.random.choice(len(feature_vectors), size=actual_batch_size, replace=True)
        batch_features = feature_vectors[feature_indices].copy()
        
        # CRITICAL: Add controlled Gaussian noise to features for intra-class variation
        # Increase noise from 5% to 10% of std for better diversity while maintaining realism
        feature_std = np.std(feature_vectors, axis=0) + 1e-8  # Avoid division by zero
        feature_noise = np.random.randn(*batch_features.shape) * (feature_std * 0.10)
        batch_features = batch_features + feature_noise
        
        # Prepare conditioning
        class_labels = torch.full((actual_batch_size,), class_label, device=device, dtype=torch.long)
        feature_tensor = torch.from_numpy(batch_features).float().to(device)
        
        # Get latent shape from VAE
        with torch.no_grad():
            # Create dummy input to get latent shape
            dummy_input = torch.randn(1, 16, 1280, device=device)  # (1, C, T)
            _, _, z_dummy, _, _ = vae_model(dummy_input)
            latent_channels = z_dummy.shape[1]
            latent_time = z_dummy.shape[2]
        
        latent_shape = (actual_batch_size, latent_channels, latent_time)
        
        # Generate latent codes using DDIM with optimized parameters
        latent_generated = ddim_sampling(
            ldm_unet=ldm_unet,
            latent_shape=latent_shape,
            class_labels=class_labels,
            feature_vectors=feature_tensor,
            diffusion_timesteps=diffusion_timesteps,
            num_inference_steps=num_inference_steps,
            eta=0.5,  # OPTIMIZED: Increased for better diversity (was 0.3)
            cfg_scale=5.0,  # FIXED: Increased from 2.5 to 5.0 for stronger conditioning
            device=device
        )
        
        # DIAGNOSTIC: Check latent codes BEFORE unscaling
        latent_before_unscale_std = latent_generated.std().item()
        latent_before_unscale_mean = latent_generated.mean().item()
        print(f"    [DEBUG] Latent BEFORE unscale - mean: {latent_before_unscale_mean:.4f}, std: {latent_before_unscale_std:.4f}, scaling_factor: {ldm_scaling_factor:.4f}")
        
        # Unscale latent
        latent_generated = latent_generated / ldm_scaling_factor
        
        # DIAGNOSTIC: Check latent codes AFTER unscaling
        latent_after_unscale_std = latent_generated.std().item()
        latent_after_unscale_mean = latent_generated.mean().item()
        print(f"    [DEBUG] Latent AFTER unscale - mean: {latent_after_unscale_mean:.4f}, std: {latent_after_unscale_std:.4f}")
        
        # Decode to EEG
        with torch.no_grad():
            synthetic_eeg = vae_model.decode(latent_generated)
        
        # DIAGNOSTIC: Check if decoded samples vary
        decoded_std = synthetic_eeg.std().item()
        decoded_mean = synthetic_eeg.mean().item()
        print(f"    [DEBUG] Decoded EEG - mean: {decoded_mean:.4f}, std: {decoded_std:.4f}")
        
        # CRITICAL DIAGNOSTIC: Check pairwise differences in THIS batch (on GPU before moving to CPU)
        if actual_batch_size > 1:
            sample_diffs = []
            for i in range(min(3, actual_batch_size - 1)):
                diff = torch.abs(synthetic_eeg[i] - synthetic_eeg[i+1]).mean().item()
                sample_diffs.append(diff)
            avg_diff_in_batch = np.mean(sample_diffs)
            print(f"    [DEBUG] Avg difference IN THIS BATCH: {avg_diff_in_batch:.6f}")
        
        # Move to CPU and denormalize
        synthetic_eeg = synthetic_eeg.cpu().numpy()
        
        if data_mean is not None and data_std is not None:
            synthetic_eeg = synthetic_eeg * data_std + data_mean
        
        all_synthetic.append(synthetic_eeg)
    
    # Concatenate all batches
    all_synthetic = np.concatenate(all_synthetic, axis=0)
    
    # DIAGNOSTIC: Check if samples are identical
    if len(all_synthetic) > 1:
        sample_diffs = []
        for i in range(min(5, len(all_synthetic) - 1)):
            diff = np.abs(all_synthetic[i] - all_synthetic[i+1]).mean()
            sample_diffs.append(diff)
        avg_diff = np.mean(sample_diffs)
        print(f"    [DEBUG] Avg difference between consecutive samples: {avg_diff:.6f}")
        if avg_diff < 1e-6:
            print(f"    [WARNING] Samples appear IDENTICAL! (diff < 1e-6)")
    
    return all_synthetic[:n_samples]  # Trim to exact count


def generate_balanced_synthetic_dataset(
    ldm_unet,
    vae_model,
    real_data: Dict[int, np.ndarray],
    real_data_raw: Dict[int, np.ndarray],
    n_samples_per_class: int,
    diffusion_timesteps: int = 1000,
    num_inference_steps: int = 50,
    device: str = 'cuda',
    ldm_scaling_factor: float = 1.0,
    data_mean: Optional[np.ndarray] = None,
    data_std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced synthetic dataset for all classes.
    
    Args:
        ldm_unet: Trained LDM U-Net
        vae_model: Trained VAE
        real_data: Dict mapping class_label -> normalized real_segments (N, C, T)
        real_data_raw: Dict mapping class_label -> RAW real_segments (N, C, T) for feature extraction
        n_samples_per_class: Number of synthetic samples per class
        diffusion_timesteps: Total diffusion steps
        num_inference_steps: Sampling steps
        device: Device
        ldm_scaling_factor: Latent scaling factor
        data_mean: For normalization/denormalization
        data_std: For normalization/denormalization
    
    Returns:
        X_synthetic: (n_samples_per_class * n_classes, C, T)
        y_synthetic: (n_samples_per_class * n_classes,)
    """
    X_synthetic_list = []
    y_synthetic_list = []
    
    for class_label in sorted(real_data.keys()):
        real_segments = real_data[class_label]
        real_segments_raw = real_data_raw.get(class_label, real_segments)  # Fallback to normalized if raw not available
        
        if len(real_segments) == 0:
            print(f"  WARNING: No real samples for class {class_label}, skipping generation")
            continue
        
        print(f"  Generating {n_samples_per_class} samples for class {class_label}...")
        
        # Extract features from RAW real samples (features must be computed from original scale)
        feature_vectors = extract_features_from_real_samples(
            vae_model=vae_model,
            real_segments=real_segments_raw,  # Use RAW data for feature extraction
            data_mean=data_mean,
            data_std=data_std,
            device=device
        )
        
        # Generate synthetic samples
        synthetic_batch = generate_synthetic_eeg_batch(
            ldm_unet=ldm_unet,
            vae_model=vae_model,
            class_label=class_label,
            feature_vectors=feature_vectors,
            n_samples=n_samples_per_class,
            diffusion_timesteps=diffusion_timesteps,
            num_inference_steps=num_inference_steps,
            device=device,
            ldm_scaling_factor=ldm_scaling_factor,
            data_mean=data_mean,
            data_std=data_std,
            batch_size=32
        )
        
        # FIXED: Apply quality filtering to synthetic samples
        # Use real samples from this class for quality reference
        if len(real_segments_raw) > 0:
            synthetic_batch = enhanced_filter_synthetic_batch(
                synth_raw_np=synthetic_batch,
                real_samples_for_comparison=real_segments_raw[:min(10, len(real_segments_raw))],
                sfreq=TARGET_SFREQ,
                feature_similarity_threshold=3.0,
                amplitude_ratio_threshold=4.0,
                spectral_similarity_threshold=0.6,
                physiological_constraint_threshold=2.5,
                min_samples_to_return=max(5, n_samples_per_class // 2)
            )
            print(f"  Quality filtering: {len(synthetic_batch)}/{n_samples_per_class} samples retained")
        
        X_synthetic_list.append(synthetic_batch)
        y_synthetic_list.append(np.full(len(synthetic_batch), class_label, dtype=np.int64))
    
    if len(X_synthetic_list) == 0:
        return np.array([]), np.array([])
    
    X_synthetic = np.concatenate(X_synthetic_list, axis=0)
    y_synthetic = np.concatenate(y_synthetic_list, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_synthetic))
    np.random.shuffle(indices)
    X_synthetic = X_synthetic[indices]
    y_synthetic = y_synthetic[indices]
    
    return X_synthetic, y_synthetic
