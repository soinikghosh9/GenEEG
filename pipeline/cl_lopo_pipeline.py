"""
Continual Learning Leave-One-Patient-Out (CL-LOPO) Validation Pipeline

This module implements the complete CL-LOPO hybrid validation framework that addresses
Reviewer #1's critical concern about data leakage in LOPO validation.

Key Features:
- **OUTER LOOP**: Iterates through each patient as held-out test set (N folds)
- **INNER LOOP**: Sequentially trains on remaining N-1 patients using EWC+ER
- **Fold-Specific Normalization**: Computed from training patients only (no leakage)
- **EWC + Experience Replay**: Prevents catastrophic forgetting
- **Comprehensive Evaluation**: Tests on completely unseen held-out patient

Architecture:
    FOR each patient P in all_patients:
        1. Hold out patient P as test set
        2. Compute normalization stats from OTHER patients only (no leakage!)
        3. Initialize fresh models (VAE, LDM, Classifier)
        4. FOR each training patient T in (all_patients - P):
            a. Load and balance T's data
            b. Mix with Experience Replay buffer (20%)
            c. Train VAE with EWC regularization
            d. Train LDM with seizure fidelity loss
            e. Update EWC Fisher Information
            f. Add samples to replay buffer
        5. Evaluate final models on held-out patient P
        6. Save results for this LOPO fold
    
    Aggregate results across all N folds

Usage:
    from pipeline import main_cl_lopo_validation
    
    # Run complete CL-LOPO validation (14 folds for Siena dataset)
    final_results = main_cl_lopo_validation(
        data_root_dir='/path/to/siena',
        output_dir='./cl_lopo_results',
        device='cuda'
    )

References:
    - Reviewer #1 Comment: "Data leakage in normalization step"
    - Kirkpatrick et al. (2017): Overcoming catastrophic forgetting with EWC
    - Shin et al. (2017): Experience Replay for continual learning

Author: GenEEG Team
Date: October 2025
"""

import os
# Optimized for Intel i7-14700 (20 cores) + RTX 4060 Ti (16GB VRAM) + 32GB RAM
os.environ.setdefault('NUMEXPR_MAX_THREADS', '20')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '12')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import sys
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import models
from models import DecoupledVAE, LatentDiffusionUNetEEG, CNNBiLSTM, EEGNet

# Import training utilities
from training import (
    train_vae,
    train_latent_diffusion_unet,
    train_pytorch_classifier,
    evaluate_pytorch_classifier,
    compute_fisher_information,
    ewc_loss_fn
)

# Import data utilities
from data import (
    SienaLoader,
    CHBMITLoader,
    EegPreprocessedDataset,
    CachedLatentDataset,
    GenericDataset,
    balance_dataset_by_undersampling,
    balance_dataset_with_smote,
    oversample_data,
    create_optimized_dataloader
)

# Import utilities
from utils import (
    get_cosine_schedule,
    robust_latent_scaling_from_mu,
    get_vae_encoder_features,
    expand_feature_vector
)

# Import evaluation
from evaluation import plot_confusion_matrix, plot_roc_auc_curves

# Import configs
from configs import (
    COMMON_EEG_CHANNELS,
    TARGET_SFREQ,
    SEGMENT_SAMPLES,
    SEED_VALUE,
    VAE_EPOCHS,
    VAE_LR,
    VAE_BATCH_SIZE,
    LDM_EPOCHS,
    LDM_LR,
    LDM_BATCH_SIZE,
    LDM_DIFFUSION_TIMESTEPS,
    FINETUNE_EPOCHS,
    FINETUNE_LR_FACTOR,
    EXPERIENCE_REPLAY_RATIO,
    VAE_FEATURE_COND_DIM,
    VAE_LATENT_CHANNELS,
    NUM_CLASSES,
    LDM_LOW_FREQ_PRESERVATION_WEIGHT,
    LDM_MOMENT_LOSS_WEIGHT
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


def standardize_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Standardize EEG data to have a specific number of channels.
    
    If data has MORE channels: Select first target_channels
    If data has FEWER channels: Pad with zeros
    
    Args:
        data: Input data of shape (batch, channels, time) or (channels, time)
        target_channels: Target number of channels
    
    Returns:
        Standardized data with target_channels channels
    """
    if data.ndim == 2:
        # Shape: (channels, time)
        current_channels = data.shape[0]
        if current_channels == target_channels:
            return data
        elif current_channels > target_channels:
            # Select first target_channels
            return data[:target_channels, :]
        else:
            # Pad with zeros
            pad_width = ((0, target_channels - current_channels), (0, 0))
            return np.pad(data, pad_width, mode='constant', constant_values=0)
    
    elif data.ndim == 3:
        # Shape: (batch, channels, time)
        current_channels = data.shape[1]
        if current_channels == target_channels:
            return data
        elif current_channels > target_channels:
            # Select first target_channels
            return data[:, :target_channels, :]
        else:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_channels - current_channels), (0, 0))
            return np.pad(data, pad_width, mode='constant', constant_values=0)
    
    else:
        raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")


def compute_fold_specific_normalization(
    patient_data_dict: Dict[str, List[Tuple[np.ndarray, int]]],
    training_patient_ids: List[str],
    max_samples_per_patient: int = 2000,  # Increased from 50 to get reliable statistics
    target_channels: int = COMMON_EEG_CHANNELS  # Target number of channels
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fold-specific normalization statistics from training patients only.
    
    This function addresses Reviewer #1's concern about data leakage by computing
    normalization statistics ONLY from the training patients, NOT from the test patient.
    
    Also handles channel standardization - if data has more/fewer channels than expected,
    it will be adjusted to match target_channels.
    
    Args:
        patient_data_dict: Dictionary mapping patient IDs to list of (segment, label) tuples
        training_patient_ids: List of patient IDs to use for computing stats (excludes test patient)
        max_samples_per_patient: Maximum samples per patient to use (for efficiency)
        target_channels: Expected number of channels for the model
    
    Returns:
        Tuple of (fold_mean_np, fold_std_np):
        - fold_mean_np: Mean of shape (1, target_channels, 1) computed from training patients only
        - fold_std_np: Std of shape (1, target_channels, 1) computed from training patients only
    
    Example:
        >>> # Correct: No data leakage
        >>> train_ids = ['PN00', 'PN01', 'PN03']  # PN02 is test patient
        >>> fold_mean, fold_std = compute_fold_specific_normalization(
        ...     all_patient_data, train_ids
        ... )
        >>> # fold_mean and fold_std computed from PN00, PN01, PN03 ONLY
    """
    print(f"\n[INFO] Computing fold-specific normalization from {len(training_patient_ids)} training patients...")
    print(f"  Training patients: {training_patient_ids}")
    
    all_train_segments_for_stats = []
    
    for train_pid in training_patient_ids:
        if train_pid not in patient_data_dict:
            print(f"  [WARN] Patient {train_pid} not found in data dict. Skipping.")
            continue
        
        # patient_data_dict[train_pid] is a tuple of (segments_array, labels_array)
        segments_array, labels_array = patient_data_dict[train_pid]
        train_patient_segments = segments_array  # This is already a numpy array
        
        # Standardize channels to target_channels (e.g., 16)
        train_patient_segments = standardize_channels(train_patient_segments, target_channels)
        
        # Sample subset if too many samples (for efficiency)
        if len(train_patient_segments) > max_samples_per_patient:
            indices = np.random.choice(
                len(train_patient_segments), 
                max_samples_per_patient, 
                replace=False
            )
            sampled_segments = train_patient_segments[indices]
        else:
            sampled_segments = train_patient_segments
        
        # Append the sampled segments
        all_train_segments_for_stats.append(sampled_segments)
    
    if len(all_train_segments_for_stats) == 0:
        print("  [ERROR] No training data available for normalization! Using identity scaling.")
        # Use target_channels as fallback
        fold_mean_np = np.zeros((1, target_channels, 1), dtype=np.float32)
        fold_std_np = np.ones((1, target_channels, 1), dtype=np.float32)
    else:
        # Concatenate all segments from different patients
        fold_stats_np = np.concatenate(all_train_segments_for_stats, axis=0).astype(np.float32)
        # Channels should now match target_channels due to standardization
        n_channels = fold_stats_np.shape[1]
        
        if n_channels != target_channels:
            print(f"  [WARN] Channel mismatch: data has {n_channels}, expected {target_channels}. Standardizing...")
            fold_stats_np = standardize_channels(fold_stats_np, target_channels)
            n_channels = target_channels
        
        # Check data characteristics
        print(f"  Normalization computed from {len(fold_stats_np)} training segments")
        print(f"  Standardized to {n_channels} channels (target: {target_channels})")
        data_min = float(fold_stats_np.min())
        data_max = float(fold_stats_np.max())
        data_std = float(fold_stats_np.std())
        data_range = data_max - data_min
        print(f"  Data value range: [{data_min:.6f}, {data_max:.6f}]")
        print(f"  Data std: {data_std:.6f}, range: {data_range:.6f}")

        # CRITICAL FIX: Data in range [-0.006, 0.005] is TOO SMALL for neural networks
        # Neural networks expect inputs with std around 0.5-2.0 for proper gradient flow
        # Raw EEG std ~0.00006 → Need to scale by ~10000× to reach std ~0.6
        if data_std < 0.001 or data_range < 0.1:
            print(f"  [INFO] Data range very small (likely microvolts). Scaling UP by 10000× for neural network.")
            fold_mean_np = np.zeros((1, n_channels, 1), dtype=np.float32)
            # std = 0.0001 means divide by 0.0001 = multiply by 10000
            fold_std_np = np.ones((1, n_channels, 1), dtype=np.float32) * 0.0001
        else:
            # Data has reasonable variance - use robust median + IQR normalization
            fold_median_np = np.median(fold_stats_np, axis=(0, 2), keepdims=True).astype(np.float32)
            q75 = np.percentile(fold_stats_np, 75, axis=(0, 2), keepdims=True).astype(np.float32)
            q25 = np.percentile(fold_stats_np, 25, axis=(0, 2), keepdims=True).astype(np.float32)
            iqr = (q75 - q25).astype(np.float32)

            # Convert IQR to approximate std (IQR / 1.349) and enforce reasonable minimums
            iqr_std = (iqr / 1.349).astype(np.float32)
            # If IQR-based std is effectively zero (flat data), fall back to plain std
            plain_std = fold_stats_np.std(axis=(0, 2), keepdims=True).astype(np.float32)
            robust_std = np.where(iqr_std < 1e-6, plain_std, iqr_std)
            # Final safety floor (but not too small to cause explosions)
            robust_std[robust_std < 1e-3] = 1e-3

            fold_mean_np = fold_median_np
            fold_std_np = robust_std
            
            print(f"  Median range: [{fold_mean_np.min():.6f}, {fold_mean_np.max():.6f}]")
            print(f"  IQR-based std range: [{fold_std_np.min():.6f}, {fold_std_np.max():.6f}]")
    
    return fold_mean_np, fold_std_np


def sequential_patient_training(
    vae_model: nn.Module,
    feature_recon_head: nn.Module,
    ldm_unet: nn.Module,
    train_patient_ids: List[str],
    patient_data_dict: Dict[str, List[Tuple[np.ndarray, int]]],
    fold_mean_np: np.ndarray,
    fold_std_np: np.ndarray,
    device: torch.device,
    fold_output_dir: str,
    ldm_scaling_factor: float = 1.0,
    ewc_vae_params: Optional[Dict] = None,
    ewc_ldm_params: Optional[Dict] = None,
    experience_replay_buffer: Optional[List] = None
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict, Dict, List]:
    """
    Sequentially train VAE and LDM on multiple patients using EWC and Experience Replay.
    
    This function implements the INNER LOOP of CL-LOPO validation, training models
    sequentially on N-1 patients while preventing catastrophic forgetting.
    
    Args:
        vae_model: VAE model to train
        feature_recon_head: Feature reconstruction head
        ldm_unet: LDM UNet to train
        train_patient_ids: List of patient IDs to train on sequentially
        patient_data_dict: Dictionary of all patient data
        fold_mean_np: Fold-specific normalization mean (no leakage!)
        fold_std_np: Fold-specific normalization std (no leakage!)
        device: Device for training
        fold_output_dir: Directory to save checkpoints
        ldm_scaling_factor: Latent scaling factor for LDM
        ewc_vae_params: Existing EWC parameters for VAE (or None for first patient)
        ewc_ldm_params: Existing EWC parameters for LDM (or None for first patient)
        experience_replay_buffer: Existing replay buffer (or None/empty list)
    
    Returns:
        Tuple of (vae_model, feature_recon_head, ldm_unet, ewc_vae_params, ewc_ldm_params, replay_buffer)
    """
    if experience_replay_buffer is None:
        experience_replay_buffer = []
    
    # Diffusion schedule
    betas = get_cosine_schedule(LDM_DIFFUSION_TIMESTEPS).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    for cl_step_idx, train_patient_id in enumerate(train_patient_ids):
        print(f"\n  --- CL Step {cl_step_idx+1}/{len(train_patient_ids)}: Training on {train_patient_id} ---")
        
        is_first_patient = (cl_step_idx == 0)
        
        # Get current patient's data (already in array format)
        current_data = patient_data_dict.get(train_patient_id, None)
        if current_data is None:
            print(f"    [WARN] No data for patient {train_patient_id}. Skipping.")
            continue
        
        X_patient_full_raw, y_patient_full = current_data
        
        # Standardize channels to 16 (handle CHB-MIT's 23 channels, Siena's 16 channels, etc.)
        X_patient_full_raw = standardize_channels(X_patient_full_raw, COMMON_EEG_CHANNELS)
        print(f"    Standardized to {COMMON_EEG_CHANNELS} channels")
        
        # Balance the dataset
        print(f"    Balancing dataset...")
        undersampled_segments, undersampled_labels = balance_dataset_by_undersampling(
            list(X_patient_full_raw), list(y_patient_full)
        )
        balanced_segments, balanced_labels = balance_dataset_with_smote(
            undersampled_segments, undersampled_labels
        )
        
        X_train_task_raw = np.array(balanced_segments, dtype=np.float32)
        y_train_task = np.array(balanced_labels, dtype=np.int64)
        
        # Mix with Experience Replay
        X_train_final_raw = list(X_train_task_raw)
        y_train_final = list(y_train_task)
        
        if (not is_first_patient) and experience_replay_buffer and (EXPERIENCE_REPLAY_RATIO > 0):
            num_replay_samples = int(len(X_train_final_raw) * (EXPERIENCE_REPLAY_RATIO / (1 - EXPERIENCE_REPLAY_RATIO)))
            num_replay_samples = min(num_replay_samples, len(experience_replay_buffer))
            print(f"    Mixing in {num_replay_samples} samples from Experience Replay buffer.")
            replay_indices = np.random.choice(len(experience_replay_buffer), num_replay_samples, replace=False)
            X_replay, y_replay = zip(*[experience_replay_buffer[i] for i in replay_indices])
            X_train_final_raw.extend(X_replay)
            y_train_final.extend(y_replay)
        
        X_train_final_raw = np.array(X_train_final_raw)
        y_train_final = np.array(y_train_final)
        
        # Create dataset and loader
        train_dataset = EegPreprocessedDataset(
            X_train_final_raw, y_train_final, fold_mean_np, fold_std_np, augment=True
        )
        train_loader = create_optimized_dataloader(train_dataset, VAE_BATCH_SIZE, shuffle=True)
        
        # Determine training epochs and learning rates
        # Uses VAE_EPOCHS and LDM_EPOCHS from config (2 for testing, 100 for production)
        # Initial training: full epochs for proper learning
        # Fine-tuning: reduced epochs for faster adaptation
        epochs_vae = VAE_EPOCHS if is_first_patient else max(2, VAE_EPOCHS // 3)
        epochs_ldm = LDM_EPOCHS if is_first_patient else max(2, LDM_EPOCHS // 3)
        lr_vae = VAE_LR if is_first_patient else VAE_LR / FINETUNE_LR_FACTOR
        lr_ldm = LDM_LR if is_first_patient else LDM_LR / FINETUNE_LR_FACTOR
        
        print(f"    Training config: VAE={epochs_vae} epochs, LDM={epochs_ldm} epochs")
        
        # Create checkpoint directory for this CL step
        cl_step_dir = os.path.join(fold_output_dir, f"CL_step_{cl_step_idx+1}_{train_patient_id}")
        os.makedirs(cl_step_dir, exist_ok=True)
        
        # Train VAE with EWC
        print(f"    Training VAE ({'Initial' if is_first_patient else 'Finetuning'})...")
        vae_model, feature_recon_head, vae_history = train_vae(
            vae_model, 
            feature_recon_head, 
            train_loader, 
            epochs_vae, 
            lr_vae, 
            device,
            os.path.join(cl_step_dir, "vae_model.pt"), 
            ewc_params=ewc_vae_params
        )
        
        # Create latent cache
        latent_dataset = CachedLatentDataset(
            base_loader=train_loader, 
            vae=vae_model, 
            device=device,
            ldm_scaling_factor=ldm_scaling_factor,
            data_mean=fold_mean_np, 
            data_std=fold_std_np
        )
        latent_loader = DataLoader(
            latent_dataset, 
            batch_size=LDM_BATCH_SIZE, 
            shuffle=True, 
            num_workers=0,  # CRITICAL FIX: Windows multiprocessing stability
            pin_memory=True,  # RTX 4060 Ti optimization
            persistent_workers=False,  # Disabled for Windows
            prefetch_factor=None,  # Disabled when num_workers=0
            drop_last=True
        )
        
        # Train LDM
        print(f"    Training LDM...")
        ldm_unet, ldm_history = train_latent_diffusion_unet(
            unet_model=ldm_unet,
            vae_model_frozen=vae_model,
            train_loader=latent_loader,
            val_loader=None,
            epochs=epochs_ldm,
            lr=lr_ldm,
            diffusion_timesteps=LDM_DIFFUSION_TIMESTEPS,
            device=device,
            save_path=os.path.join(cl_step_dir, "ldm_model.pt"),
            target_sfreq=TARGET_SFREQ,
            ldm_scaling_factor=ldm_scaling_factor,
            low_freq_preservation_weight=LDM_LOW_FREQ_PRESERVATION_WEIGHT,
            moment_loss_weight=LDM_MOMENT_LOSS_WEIGHT,
            loss_target="v",
            min_snr_gamma=2.0, 
            p2_k=1.0, 
            p2_gamma=1.0,
            ema_decay=0.9995,
            ewc_params=ewc_ldm_params,
            lambda_ewc=5000.0
        )
        
        # CRITICAL: Clear GPU memory after intensive training
        print(f"    Cleaning up GPU memory after training...")
        del latent_loader, latent_dataset
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Update EWC parameters
        print(f"    Updating EWC parameters...")
        
        # CRITICAL: Aggressive memory cleanup before Fisher computation
        # After 50 VAE + 20 LDM epochs, GPU is nearly full
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # FIXED: Use smaller batch size for Fisher computation to avoid OOM
        fisher_batch_size = min(32, VAE_BATCH_SIZE // 2)  # Half of training batch size
        fisher_dataset = EegPreprocessedDataset(X_train_task_raw, y_train_task, fold_mean_np, fold_std_np)
        fisher_loader_vae = create_optimized_dataloader(fisher_dataset, batch_size=fisher_batch_size, shuffle=True)
        
        def vae_loss_for_fisher(model, batch):
            """Loss function for Fisher Information computation.
            
            Args:
                model: VAE model
                batch: Full batch from DataLoader (could be tuple or single tensor)
            
            Returns:
                Scalar loss tensor with requires_grad=True
            """
            # Extract x from batch
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # First element is the scaled EEG data
            else:
                x = batch
            
            x = x.to(device, non_blocking=True)
            
            # VAE forward pass
            recon, mu, logvar, z, _ = model(x)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kld = torch.mean(torch.clamp(kld, min=0.0, max=1e6))
            return recon_loss + kld
        
        # FIXED: Reduced num_batches from 200 to 50 for memory efficiency
        new_fisher_vae = compute_fisher_information(
            model=vae_model, 
            data_loader=fisher_loader_vae,
            loss_fn=vae_loss_for_fisher, 
            device=device, 
            num_batches=50  # Reduced from 200 to save GPU memory
        )
        
        # Clear memory after Fisher computation
        torch.cuda.empty_cache()
        
        # Merge with previous Fisher if present (EWC accumulation)
        if ewc_vae_params:
            for name, Fi in new_fisher_vae.items():
                if name in ewc_vae_params['fisher_info']:
                    new_fisher_vae[name] += ewc_vae_params['fisher_info'][name]
        
        ewc_vae_params = {
            'old_params': {n: p.detach().clone() for n, p in vae_model.named_parameters()},
            'fisher_info': new_fisher_vae
        }
        
        # CRITICAL FIX: Compute Fisher Information for LDM to enable continual learning
        # LDM was forgetting previous patients, causing increasing loss across folds
        print(f"    Computing Fisher Information for LDM...")
        
        # Prepare latent dataset for LDM Fisher computation
        fisher_latent_dataset = CachedLatentDataset(
            base_loader=fisher_loader_vae,
            vae=vae_model,
            device=device,
            ldm_scaling_factor=ldm_scaling_factor,
            data_mean=fold_mean_np,
            data_std=fold_std_np
        )
        fisher_loader_ldm = DataLoader(
            fisher_latent_dataset,
            batch_size=LDM_BATCH_SIZE // 2,  # Half batch size for memory safety
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None,
            drop_last=True
        )
        
        def ldm_loss_for_fisher(model, batch):
            """Loss function for LDM Fisher Information computation.
            
            Args:
                model: LDM U-Net model
                batch: Full batch from DataLoader - tuple of (z0, x_raw, features, labels)
            
            Returns:
                Scalar loss tensor with requires_grad=True
            """
            # Validate batch format
            if not isinstance(batch, (list, tuple)):
                raise RuntimeError(
                    f"LDM Fisher loss expects batch to be list/tuple from CachedLatentDataset, "
                    f"got {type(batch)}. Check that DataLoader collation is working correctly."
                )
            
            if len(batch) < 4:
                raise RuntimeError(
                    f"LDM Fisher loss expects batch with 4 elements (z0, x_raw, features, labels), "
                    f"got {len(batch)} elements. CachedLatentDataset may be returning wrong format."
                )
            
            # Unpack batch (DataLoader collates individual tuples into batched tensors)
            z0 = batch[0].to(device)
            x_raw = batch[1].to(device) if len(batch) > 1 else None
            features_cond = batch[2].to(device)
            labels_batch = batch[3].to(device)
            
            # Sample random timesteps
            B = z0.shape[0]
            t = torch.randint(0, LDM_DIFFUSION_TIMESTEPS, (B,), device=device, dtype=torch.long)
            
            # Add noise (simple diffusion forward process)
            eps = torch.randn_like(z0)
            # CRITICAL FIX: z0 is 3D [B, C, L], so alpha_bar should be [B, 1, 1] not [B, 1, 1, 1]
            alpha_bar = alphas_cumprod.index_select(0, t).view(B, 1, 1)
            zt = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * eps
            
            # Predict noise
            pred = model(zt, t, class_labels=labels_batch, feature_cond=features_cond)
            
            # MSE loss
            loss = F.mse_loss(pred, eps, reduction="mean")
            return loss
        
        # Compute Fisher with reduced batches for memory
        new_fisher_ldm = compute_fisher_information(
            model=ldm_unet,
            data_loader=fisher_loader_ldm,
            loss_fn=ldm_loss_for_fisher,
            device=device,
            num_batches=30  # Reduced for LDM (larger model)
        )
        
        # Clear memory
        del fisher_loader_ldm, fisher_latent_dataset
        torch.cuda.empty_cache()
        
        # Merge with previous Fisher if present (EWC accumulation)
        if ewc_ldm_params:
            for name, Fi in new_fisher_ldm.items():
                if name in ewc_ldm_params['fisher_info']:
                    new_fisher_ldm[name] += ewc_ldm_params['fisher_info'][name]
        
        ewc_ldm_params = {
            'old_params': {n: p.detach().clone() for n, p in ldm_unet.named_parameters()},
            'fisher_info': new_fisher_ldm
        }
        
        print(f"    ✓ EWC parameters updated for VAE and LDM")
        
        # Update replay buffer
        if len(X_train_task_raw) > 0:
            num_to_add = min(500, len(X_train_task_raw))
            sample_indices = np.random.choice(len(X_train_task_raw), num_to_add, replace=False)
            for idx in sample_indices:
                experience_replay_buffer.append((X_train_task_raw[idx], y_train_task[idx]))
            # Keep buffer size manageable (max 5000 samples)
            if len(experience_replay_buffer) > 5000:
                experience_replay_buffer = experience_replay_buffer[-5000:]
        
        print(f"    CL Step {cl_step_idx+1} complete. {len(experience_replay_buffer)} samples in replay buffer.")
    
    return vae_model, feature_recon_head, ldm_unet, ewc_vae_params, ewc_ldm_params, experience_replay_buffer


def main_cl_lopo_validation(
    data_root_dir: str = None,
    output_dir: str = None,
    device: str = 'cuda',
    dataset_name: str = 'siena',
    max_patients: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Main CL-LOPO validation framework addressing Reviewer #1's data leakage concern.
    
    This function implements a complete nested-loop validation structure:
    - OUTER LOOP: Iterates through each patient as held-out test set
    - INNER LOOP: Sequentially trains on remaining N-1 patients with EWC+ER
    - Evaluation: Tests final models on completely unseen held-out patient
    
    Args:
        data_root_dir: Root directory containing patient data folders
        output_dir: Directory to save all outputs and results
        device: Device for training ('cuda' or 'cpu')
        dataset_name: Dataset identifier ('siena' or 'chbmit')
        max_patients: Maximum number of patients to process (None = all patients)
    
    Returns:
        List of dictionaries containing results for each LOPO fold
    
    Example:
        >>> # Run complete CL-LOPO validation
        >>> results = main_cl_lopo_validation(
        ...     data_root_dir='/data/siena',
        ...     output_dir='./cl_lopo_results',
        ...     device='cuda'
        ... )
        >>> # Results contains 14 folds (one per patient)
        >>> print(f"Average accuracy: {np.mean([r['accuracy'] for r in results]):.3f}")
    """
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"\n[INFO] Using device: {device}")
    
    # Disable anomaly detection for better performance and to avoid NaN detection errors
    torch.autograd.set_detect_anomaly(False)
    
    # Initialization & Setup
    start_time_main = time.time()
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)
    
    # Create output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join('.', 'cl_lopo_output')
    main_run_output_dir = os.path.join(output_dir, f"run_{run_timestamp}_CL_LOPO")
    os.makedirs(main_run_output_dir, exist_ok=True)
    
    # GenEEG ASCII Art Logo
    print(f"\n{'='*80}")
    print("""
   ████╗ ███████╗███╗   ██╗███████╗███████╗ ██████╗ 
  ██╔══╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝ 
  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  █████╗  ██║  ███╗
  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══╝  ██║   ██║
  ╚██████╔╝███████╗██║ ╚████║███████╗███████╗╚██████╔╝
   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝ 
    """)
    print("  Epileptic EEG Detection via Patient-Adaptive Latent Diffusion")
    print("  & Continual Learning with EWC + Experience Replay")
    print(f"{'='*80}")
    print(f"  CL-LOPO HYBRID VALIDATION FRAMEWORK")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"  All outputs saved to: {main_run_output_dir}")
    print(f"{'='*80}\n")
    
    # Load all patient data
    print("\n--- Section 1: Data Loading & Grouping by Patient ---")
    
    if dataset_name.lower() == 'siena':
        if data_root_dir is None:
            raise ValueError("data_root_dir must be specified for Siena dataset")
        
        # Load Siena data with consistent 16 channels
        loader = SienaLoader(data_root_dir, num_channels=COMMON_EEG_CHANNELS)
        all_patient_data_raw = loader.load_all_patients()
    
    elif dataset_name.lower() in ['chbmit', 'chb-mit', 'chb_mit']:
        if data_root_dir is None:
            raise ValueError("data_root_dir must be specified for CHB-MIT dataset")
        
        # Load CHB-MIT data WITHOUT artificial balancing
        # Class imbalance is natural in epilepsy detection and will be handled via:
        # 1. Class weights in loss function
        # 2. Appropriate evaluation metrics (F1, sensitivity, specificity)
        print(f"Loading CHB-MIT dataset from: {data_root_dir}")
        loader = CHBMITLoader(
            root_dir=data_root_dir,
            target_sampling_rate=TARGET_SFREQ,
            common_channels=None,  # Use all available channels, will be standardized later
            verbose=True
        )
        # Disable artificial balancing to preserve natural class distribution
        all_patient_data_raw = loader.load_all_patients(
            segment_duration_sec=5.0,
            overlap=0.5,
            balance_classes=False  # CRITICAL: Preserve natural imbalance
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    all_patient_ids = list(all_patient_data_raw.keys())
    if max_patients is not None:
        all_patient_ids = all_patient_ids[:max_patients]
    
    print(f"\nLoaded data for {len(all_patient_ids)} patients: {all_patient_ids}")
    
    final_results_all_folds = []
    
    # OUTER LOOP: Leave-One-Patient-Out (LOPO) Validation
    for lopo_fold_idx, test_patient_id in enumerate(all_patient_ids):
        print(f"\n\n{'='*100}")
        print(f"  LOPO FOLD {lopo_fold_idx+1}/{len(all_patient_ids)}: TESTING ON {test_patient_id}")
        print(f"{'='*100}\n")
        
        fold_output_dir = os.path.join(main_run_output_dir, f"LOPO_fold_{lopo_fold_idx+1}_{test_patient_id}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Setup: Sequential CL Training on N-1 Patients
        train_patient_ids = [pid for pid in all_patient_ids if pid != test_patient_id]
        print(f"Training sequentially on {len(train_patient_ids)} patients (excluding test patient {test_patient_id})")
        print(f"Training sequence: {train_patient_ids}\n")
        
        # Initialize FRESH models for this LOPO fold with consistent 16 channels
        print(f"Initializing models with {COMMON_EEG_CHANNELS} channels")
        
        vae_model = DecoupledVAE(
            in_channels=COMMON_EEG_CHANNELS,
            base_channels=128,
            latent_channels=VAE_LATENT_CHANNELS,
            ch_mults=(1, 2, 4, 8),
            blocks_per_stage=2,
            use_feature_cond=True,
            feature_cond_dim=VAE_FEATURE_COND_DIM
        ).to(device)
        feature_recon_head = nn.Sequential(
            nn.Linear(VAE_LATENT_CHANNELS, 64), nn.ReLU(),
            nn.Linear(64, VAE_FEATURE_COND_DIM)
        ).to(device)
        ldm_unet = LatentDiffusionUNetEEG(
            latent_channels=VAE_LATENT_CHANNELS,
            base_unet_channels=256,
            time_emb_dim=256,
            num_classes=NUM_CLASSES
        ).to(device)
        
        # Apply enhanced initialization
        def init_weights(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        vae_model.apply(init_weights)
        feature_recon_head.apply(init_weights)
        ldm_unet.apply(init_weights)
        
        # Initialize FRESH EWC params and Replay Buffer
        ewc_vae_params = None
        experience_replay_buffer = []
        ldm_scaling_factor = 1.0
        
        # CRITICAL: Compute fold-specific normalization (NO TEST LEAKAGE!)
        fold_mean_np, fold_std_np = compute_fold_specific_normalization(
            all_patient_data_raw, 
            train_patient_ids
        )
        
        # Determine latent scaling factor on first patient
        first_patient_data = all_patient_data_raw.get(train_patient_ids[0], None)
        if first_patient_data is not None:
            X_first_full, y_first_full = first_patient_data
            # Standardize channels before processing
            X_first_full = standardize_channels(X_first_full, COMMON_EEG_CHANNELS)
            # Sample subset for efficiency
            num_samples = min(500, len(X_first_full))
            X_first = X_first_full[:num_samples]
            y_first = y_first_full[:num_samples]
            temp_dataset = GenericDataset(((X_first - fold_mean_np) / fold_std_np).astype(np.float32), y_first)
            temp_loader = DataLoader(temp_dataset, batch_size=VAE_BATCH_SIZE*2, shuffle=False, 
                                      num_workers=0, pin_memory=True)  # CRITICAL FIX: Windows compatibility
            ldm_scaling_factor = robust_latent_scaling_from_mu(vae_model, temp_loader, device)
            print(f"    Robust LDM scaling factor: {ldm_scaling_factor:.4f}")
        
        # INNER LOOP: Sequential Training
        vae_model, feature_recon_head, ldm_unet, ewc_vae_params, ewc_ldm_params, experience_replay_buffer = sequential_patient_training(
            vae_model=vae_model,
            feature_recon_head=feature_recon_head,
            ldm_unet=ldm_unet,
            train_patient_ids=train_patient_ids,
            patient_data_dict=all_patient_data_raw,
            fold_mean_np=fold_mean_np,
            fold_std_np=fold_std_np,
            device=device,
            fold_output_dir=fold_output_dir,
            ldm_scaling_factor=ldm_scaling_factor,
            ewc_vae_params=ewc_vae_params,
            ewc_ldm_params=None,  # Initialize fresh for each fold
            experience_replay_buffer=experience_replay_buffer
        )
        
        # ========================================================================
        # MODEL QUALITY ASSESSMENT (CHECK VAE RECONSTRUCTION QUALITY)
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"  MODEL QUALITY ASSESSMENT - VAE Reconstruction")
        print(f"{'='*80}\n")
        
        # Prepare sample data for quality checks
        if len(experience_replay_buffer) > 0:
            X_quality_check_raw, y_quality_check = map(np.array, zip(*experience_replay_buffer[:200]))
        else:
            # Fallback: use last training patient
            last_train_patient = train_patient_ids[-1]
            X_quality_check_raw, y_quality_check = all_patient_data_raw[last_train_patient]
            X_quality_check_raw = standardize_channels(X_quality_check_raw, COMMON_EEG_CHANNELS)
            # Take first 200 samples
            X_quality_check_raw = X_quality_check_raw[:200]
            y_quality_check = y_quality_check[:200]
        
        print(f"  Checking VAE reconstruction quality on {len(X_quality_check_raw)} samples...")
        
        try:
            from utils.vae_visualization import VAEVisualizer
            
            quality_output_dir = os.path.join(fold_output_dir, "quality_assessment")
            os.makedirs(quality_output_dir, exist_ok=True)
            
            # Test VAE reconstruction
            vae_model.eval()
            X_quality_scaled = (X_quality_check_raw - fold_mean_np) / fold_std_np
            X_quality_tensor = torch.tensor(X_quality_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                # Get VAE reconstructions
                recon, mu, logvar, z, _ = vae_model(X_quality_tensor)
                recon_np = recon.cpu().numpy()
                
                # Compute reconstruction error
                recon_error = np.mean(np.abs(X_quality_scaled - recon_np))
                print(f"    VAE Mean Absolute Reconstruction Error: {recon_error:.6f}")
                
                # Compute per-class reconstruction error
                for class_idx in range(NUM_CLASSES):
                    class_mask = (y_quality_check == class_idx)
                    if np.sum(class_mask) > 0:
                        class_recon_error = np.mean(np.abs(X_quality_scaled[class_mask] - recon_np[class_mask]))
                        print(f"      Class {class_idx}: {class_recon_error:.6f}")
            
            # Create visualization
            visualizer = VAEVisualizer(save_dir=quality_output_dir)
            
            # Plot reconstruction samples (3 samples per class)
            for class_idx in range(NUM_CLASSES):
                class_mask = (y_quality_check == class_idx)
                if np.sum(class_mask) >= 3:
                    class_indices = np.where(class_mask)[0][:3]
                    class_samples = X_quality_tensor[class_indices]
                    class_labels_tensor = torch.tensor(y_quality_check[class_indices], dtype=torch.long).to(device)
                    
                    with torch.no_grad():
                        class_recon, _, _, _, _ = vae_model(class_samples)
                    
                    visualizer.plot_reconstruction_samples(
                        original=class_samples,
                        reconstructed=class_recon,
                        labels=class_labels_tensor,
                        epoch=0,
                        n_samples=min(3, len(class_indices)),
                        save_name=f"vae_reconstruction_class{class_idx}_fold{lopo_fold_idx+1}.png"
                    )
            
            print(f"  ✓ Quality assessment complete. Results saved to: {quality_output_dir}")
        except Exception as e:
            print(f"  [WARNING] Quality assessment failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========================================================================
        # SYNTHETIC DATA GENERATION & QUALITY EVALUATION (NEW WORKFLOW STEP)
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"  SYNTHETIC DATA GENERATION & QUALITY EVALUATION")
        print(f"{'='*80}\n")
        
        # Prepare real training data for synthetic generation
        if len(experience_replay_buffer) > 0:
            X_real_for_synth_raw, y_real_for_synth = map(np.array, zip(*experience_replay_buffer))
        else:
            # Fallback: use last training patient
            last_train_patient = train_patient_ids[-1]
            X_real_for_synth_raw, y_real_for_synth = all_patient_data_raw[last_train_patient]
            X_real_for_synth_raw = standardize_channels(X_real_for_synth_raw, COMMON_EEG_CHANNELS)
        
        # Normalize real data
        X_real_for_synth_scaled = (X_real_for_synth_raw - fold_mean_np) / fold_std_np
        
        # Organize real data by class (keep BOTH raw and normalized for feature extraction)
        real_data_by_class = {}
        real_data_by_class_raw = {}  # NEW: Keep raw data for feature extraction
        for class_idx in range(NUM_CLASSES):
            class_mask = (y_real_for_synth == class_idx)
            if np.sum(class_mask) > 0:
                real_data_by_class[class_idx] = X_real_for_synth_scaled[class_mask]
                real_data_by_class_raw[class_idx] = X_real_for_synth_raw[class_mask]
        
        print(f"  Real data distribution:")
        for class_idx in range(NUM_CLASSES):
            count = len(real_data_by_class.get(class_idx, []))
            print(f"    Class {class_idx}: {count} samples")
        
        # Generate balanced synthetic dataset
        print(f"\n  Generating synthetic EEG samples...")
        from utils.generation import generate_balanced_synthetic_dataset
        from evaluation.metrics import evaluate_synthetic_quality
        
        # OPTIMIZATION: Reduce samples for faster generation (500 → 100 per class)
        # This reduces generation time by ~80% while maintaining quality validation
        n_synthetic_per_class = 100  # Generate 100 samples per class (1-2 mins vs 5-10 mins)
        X_synthetic_scaled, y_synthetic = generate_balanced_synthetic_dataset(
            ldm_unet=ldm_unet,
            vae_model=vae_model,
            real_data=real_data_by_class,
            real_data_raw=real_data_by_class_raw,  # NEW: Pass raw data for feature extraction
            n_samples_per_class=n_synthetic_per_class,
            diffusion_timesteps=LDM_DIFFUSION_TIMESTEPS,
            num_inference_steps=50,  # DDIM with 50 steps (20x faster than DDPM)
            device=device,
            ldm_scaling_factor=ldm_scaling_factor,
            data_mean=fold_mean_np,
            data_std=fold_std_np
        )
        
        # Validate synthetic data generation
        if X_synthetic_scaled is None or len(X_synthetic_scaled) == 0:
            raise RuntimeError("Synthetic data generation failed - no samples generated")
        
        print(f"\n  Generated {len(X_synthetic_scaled)} synthetic samples")
        print(f"  Synthetic data distribution:")
        for class_idx in range(NUM_CLASSES):
            count = np.sum(y_synthetic == class_idx)
            print(f"    Class {class_idx}: {count} samples")
        
        # Evaluate synthetic data quality
        print(f"\n  Evaluating synthetic data quality...")
        
        # CRITICAL: Denormalize synthetic data for feature extraction
        # (Feature extraction needs raw scale data for accurate band powers, Hjorth params, etc.)
        X_synthetic_raw = (X_synthetic_scaled * fold_std_np) + fold_mean_np
        
        # Organize synthetic data by class (BOTH scaled for visualization and raw for quality metrics)
        synthetic_data_by_class = {}  # Scaled version for visualization
        synthetic_data_by_class_raw = {}  # Raw version for feature extraction
        for class_idx in range(NUM_CLASSES):
            class_mask = (y_synthetic == class_idx)
            if np.sum(class_mask) > 0:
                synthetic_data_by_class[class_idx] = X_synthetic_scaled[class_mask]
                synthetic_data_by_class_raw[class_idx] = X_synthetic_raw[class_mask]
        
        # Compute quality metrics (using RAW scale data for both real and synthetic)
        quality_metrics = evaluate_synthetic_quality(
            real_data=real_data_by_class_raw,  # Use RAW real data
            synthetic_data=synthetic_data_by_class_raw,  # Use RAW synthetic data
            target_sfreq=TARGET_SFREQ
        )
        
        print(f"\n  Quality Metrics Summary:")
        for class_idx in range(NUM_CLASSES):
            if class_idx in quality_metrics:
                metrics = quality_metrics[class_idx]
                print(f"    Class {class_idx}:")
                print(f"      FID Score: {metrics['fid']:.4f} (lower is better)")
                print(f"      Diversity: {metrics['diversity']:.4f} (higher is better)")
                print(f"      Feature Match Rate: {metrics['feature_match_rate']:.2%}")
                
                # Show feature statistics
                if 'feature_details' in metrics:
                    passed_features = sum(1 for f in metrics['feature_details'].values() if f['passed'])
                    total_features = len(metrics['feature_details'])
                    print(f"      Features Passed: {passed_features}/{total_features}")
        
        # Save quality metrics
        quality_metrics_path = os.path.join(fold_output_dir, "synthetic_quality_metrics.json")
        with open(quality_metrics_path, 'w') as f:
            json.dump(quality_metrics, f, indent=4, cls=NumpyEncoder)
        print(f"\n  Quality metrics saved: {quality_metrics_path}")
        
        # Visualize quality metrics
        try:
            from evaluation.metrics import plot_quality_metrics, plot_synthetic_vs_real_comparison
            
            # Plot quality metrics summary
            quality_plot_path = os.path.join(fold_output_dir, "synthetic_quality_visualization.png")
            plot_quality_metrics(quality_metrics, quality_plot_path)
            
            # Plot real vs synthetic comparison for each class
            for class_idx in range(NUM_CLASSES):
                if class_idx in real_data_by_class and class_idx in synthetic_data_by_class:
                    comparison_plot_path = os.path.join(
                        fold_output_dir, 
                        f"real_vs_synthetic_class{class_idx}.png"
                    )
                    plot_synthetic_vs_real_comparison(
                        real_samples=real_data_by_class[class_idx],
                        synthetic_samples=synthetic_data_by_class[class_idx],
                        class_label=class_idx,
                        save_path=comparison_plot_path,
                        n_samples=3,
                        channel_idx=0,
                        sfreq=TARGET_SFREQ
                    )
        except Exception as e:
            print(f"[WARNING] Quality visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Save synthetic data
        synthetic_data_path = os.path.join(fold_output_dir, "synthetic_data.npz")
        np.savez_compressed(
            synthetic_data_path,
            X_synthetic=X_synthetic_scaled,
            y_synthetic=y_synthetic,
            fold_mean=fold_mean_np,
            fold_std=fold_std_np
        )
        print(f"  Synthetic data saved: {synthetic_data_path}")
        
        # ========================================================================
        # LOPO EVALUATION: Test on Held-Out Patient
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"  EVALUATING on held-out test patient: {test_patient_id}")
        print(f"{'='*80}\n")
        
        # Load test patient's data (already in array format)
        X_test_raw_full, y_test_full = all_patient_data_raw[test_patient_id]
        
        # Standardize channels to 16 (handle CHB-MIT's 23 channels, Siena's 16 channels, etc.)
        X_test_raw_full = standardize_channels(X_test_raw_full, COMMON_EEG_CHANNELS)
        print(f"  Test data standardized to {COMMON_EEG_CHANNELS} channels")
        
        # CRITICAL: Use fold-specific normalization (no test leakage!)
        X_test_scaled = (X_test_raw_full - fold_mean_np) / fold_std_np
        
        # Extract features for classifier training
        X_feat_test = get_vae_encoder_features(X_test_scaled, vae_model, device)
        
        # Use replay buffer for classifier training data
        if len(experience_replay_buffer) > 0:
            X_train_clf_raw, y_train_clf = map(np.array, zip(*experience_replay_buffer))
        else:
            # Fallback: use last training patient (already in array format)
            last_train_patient = train_patient_ids[-1]
            X_train_clf_raw, y_train_clf = all_patient_data_raw[last_train_patient]
            # Standardize channels for fallback case
            X_train_clf_raw = standardize_channels(X_train_clf_raw, COMMON_EEG_CHANNELS)
        
        X_train_clf_scaled = (X_train_clf_raw - fold_mean_np) / fold_std_np
        X_feat_train = get_vae_encoder_features(X_train_clf_scaled, vae_model, device)
        
        # Run classifier evaluation
        fold_results = {
            'fold': lopo_fold_idx + 1, 
            'test_patient': test_patient_id, 
            'classifier_performance': {}
        }
        
        # FIXED: Updated scenarios - Added GenEEG_Augmented for minority class balancing
        scenarios_list = ["Real_Data", "Real_Oversampled", "Real_Synthetic", "GenEEG_Augmented"]
        class_names = ["Normal", "Preictal", "Ictal"]
        
        for scenario_name in scenarios_list:
            print(f"\n  Scenario: {scenario_name}")
            scenario_output_dir = os.path.join(fold_output_dir, f"classifier_{scenario_name}")
            os.makedirs(scenario_output_dir, exist_ok=True)
            
            if scenario_name == "Real_Data":
                # Real data only (baseline)
                X_scen_raw, y_scen = X_train_clf_raw, y_train_clf
            elif scenario_name == "Real_Oversampled":
                # Real data with random oversampling
                try:
                    X_scen_raw, y_scen = oversample_data(X_train_clf_raw, y_train_clf, strategy='random_raw')
                except Exception as e:
                    print(f"    [WARNING] Oversampling failed: {e}. Using original data.")
                    X_scen_raw, y_scen = X_train_clf_raw, y_train_clf
            elif scenario_name == "Real_Synthetic":
                # Real data augmented with synthetic samples (ALL classes)
                if len(X_synthetic_scaled) == 0:
                    print(f"    [WARNING] No synthetic data available. Using real data only.")
                    X_scen_raw, y_scen = X_train_clf_raw, y_train_clf
                else:
                    # Denormalize synthetic data to raw scale
                    X_synthetic_raw = (X_synthetic_scaled * fold_std_np) + fold_mean_np
                    
                    # Combine real and synthetic data
                    X_scen_raw = np.concatenate([X_train_clf_raw, X_synthetic_raw], axis=0)
                    y_scen = np.concatenate([y_train_clf, y_synthetic], axis=0)
                
                # Shuffle combined dataset
                shuffle_indices = np.random.permutation(len(X_scen_raw))
                X_scen_raw = X_scen_raw[shuffle_indices]
                y_scen = y_scen[shuffle_indices]
                
                print(f"    Combined dataset size: {len(X_scen_raw)} samples")
                print(f"      Real: {len(X_train_clf_raw)}, Synthetic: {len(X_synthetic_raw)}")
            elif scenario_name == "GenEEG_Augmented":
                # FIXED: Real data + synthetic ONLY for minority classes (Preictal=1, Ictal=2)
                # This is the key scenario from original GenEEG paper
                
                if len(X_synthetic_scaled) == 0:
                    print(f"    [WARNING] No synthetic data available. Using real data only.")
                    X_scen_raw, y_scen = X_train_clf_raw, y_train_clf
                    continue
                
                # Denormalize synthetic data
                X_synthetic_raw = (X_synthetic_scaled * fold_std_np) + fold_mean_np
                
                # Count samples per class in real data
                real_class_counts = {c: np.sum(y_train_clf == c) for c in range(NUM_CLASSES)}
                majority_count = max(real_class_counts.values())
                
                print(f"    Real class distribution: {real_class_counts}")
                print(f"    Majority class count: {majority_count}")
                
                # Keep all real data
                X_augmented = [X_train_clf_raw]
                y_augmented = [y_train_clf]
                
                # Add synthetic samples ONLY for minority classes
                for class_idx in [1, 2]:  # Preictal and Ictal (minority classes)
                    if real_class_counts[class_idx] < majority_count:
                        # Get synthetic samples for this class
                        synth_mask = y_synthetic == class_idx
                        synth_for_class = X_synthetic_raw[synth_mask]
                        
                        # Calculate how many synthetic samples to add
                        deficit = majority_count - real_class_counts[class_idx]
                        n_to_add = min(deficit, len(synth_for_class))
                        
                        if n_to_add > 0:
                            # Add synthetic samples
                            X_augmented.append(synth_for_class[:n_to_add])
                            y_augmented.append(np.full(n_to_add, class_idx))
                            print(f"      Class {class_idx} ({class_names[class_idx]}): Adding {n_to_add} synthetic samples")
                
                # Combine and shuffle
                X_scen_raw = np.concatenate(X_augmented, axis=0)
                y_scen = np.concatenate(y_augmented, axis=0)
                
                shuffle_indices = np.random.permutation(len(X_scen_raw))
                X_scen_raw = X_scen_raw[shuffle_indices]
                y_scen = y_scen[shuffle_indices]
                
                # Print final distribution
                final_counts = {c: np.sum(y_scen == c) for c in range(NUM_CLASSES)}
                print(f"    Final augmented distribution: {final_counts}")
            
            X_scen_scaled = (X_scen_raw - fold_mean_np) / fold_std_np
            
            # Train and evaluate CNN-BiLSTM
            cnn_bilstm_model = CNNBiLSTM(num_classes=NUM_CLASSES).to(device)
            _, _ = train_pytorch_classifier(
                cnn_bilstm_model, X_scen_scaled, y_scen, 50, 1e-4, device,
                "CNN_BiLSTM", scenario_name, scenario_output_dir, class_names, validation_split=0.2
            )
            cnn_bilstm_results = evaluate_pytorch_classifier(
                cnn_bilstm_model, X_test_scaled, y_test_full, device,
                "CNN_BiLSTM", scenario_name, scenario_output_dir, class_names
            )
            
            scenario_results = {'CNN_BiLSTM': cnn_bilstm_results}
            fold_results['classifier_performance'][scenario_name] = scenario_results
        
        # Visualize classifier comparison across scenarios
        try:
            from evaluation.classifier_plots import plot_classifier_comparison, plot_per_class_f1_comparison
            
            # Prepare scenario data for visualization
            scenarios_viz = {}
            for scenario_name in scenarios_list:
                if scenario_name in fold_results['classifier_performance']:
                    clf_results = fold_results['classifier_performance'][scenario_name]['CNN_BiLSTM']
                    scenarios_viz[scenario_name] = {
                        'accuracy': clf_results.get('accuracy', 0),
                        'precision': clf_results.get('macro_precision', 0),
                        'recall': clf_results.get('macro_recall', 0),
                        'f1': clf_results.get('macro_f1', 0),
                        'f1_per_class': clf_results.get('per_class_f1', [0] * NUM_CLASSES),
                        'confusion_matrix': np.array(clf_results.get('confusion_matrix', [[0]*NUM_CLASSES]*NUM_CLASSES))
                    }
            
            # Plot overall comparison
            comparison_plot_path = os.path.join(fold_output_dir, "classifier_performance_comparison.png")
            plot_classifier_comparison(scenarios_viz, comparison_plot_path)
            
            # Plot per-class F1 comparison
            per_class_plot_path = os.path.join(fold_output_dir, "per_class_f1_comparison.png")
            plot_per_class_f1_comparison(scenarios_viz, class_names, per_class_plot_path)
            
        except Exception as e:
            print(f"[WARNING] Classifier comparison visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        final_results_all_folds.append(fold_results)
        
        # ========================================================================
        # PER-FOLD SUMMARY & PROGRESS REPORT (NEW - Immediate Feedback)
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"  FOLD {lopo_fold_idx+1}/{len(all_patient_ids)} COMPLETED - IMMEDIATE RESULTS")
        print(f"{'='*80}\n")
        
        # Print this fold's results immediately
        print(f"  Test Patient: {test_patient_id}")
        print(f"\n  Classifier Performance Summary:")
        for scenario_name in scenarios_list:
            if scenario_name in fold_results['classifier_performance']:
                clf_results = fold_results['classifier_performance'][scenario_name]['CNN_BiLSTM']
                print(f"\n    {scenario_name}:")
                print(f"      Accuracy:  {clf_results.get('accuracy', 0):.4f}")
                print(f"      F1-Macro:  {clf_results.get('macro_f1', 0):.4f}")
                print(f"      F1-Weighted: {clf_results.get('f1_weighted', 0):.4f}")
                
                # Per-class F1
                per_class_f1 = clf_results.get('per_class_f1', [])
                if len(per_class_f1) == NUM_CLASSES:
                    print(f"      Per-Class F1:")
                    for i, f1 in enumerate(per_class_f1):
                        print(f"        Class {i}: {f1:.4f}")
        
        # Print running averages across completed folds
        if len(final_results_all_folds) > 0:
            print(f"\n  Running Averages Across {len(final_results_all_folds)} Completed Folds:")
            
            for scenario_name in scenarios_list:
                accuracies = []
                f1_macros = []
                
                for fold_res in final_results_all_folds:
                    if scenario_name in fold_res['classifier_performance']:
                        clf_res = fold_res['classifier_performance'][scenario_name]['CNN_BiLSTM']
                        accuracies.append(clf_res.get('accuracy', 0))
                        f1_macros.append(clf_res.get('macro_f1', 0))
                
                if accuracies:
                    print(f"\n    {scenario_name}:")
                    print(f"      Avg Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                    print(f"      Avg F1-Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
        
        # Save fold summary
        fold_report_path = os.path.join(fold_output_dir, f"LOPO_fold_{lopo_fold_idx+1}_summary.json")
        with open(fold_report_path, 'w') as f:
            json.dump(fold_results, f, indent=4, cls=NumpyEncoder)
        print(f"\n  ✓ Fold {lopo_fold_idx+1} summary saved: {fold_report_path}")
        
        # Save running progress report
        progress_report_path = os.path.join(main_run_output_dir, f"progress_after_fold_{lopo_fold_idx+1}.json")
        progress_data = {
            'completed_folds': len(final_results_all_folds),
            'total_folds': len(all_patient_ids),
            'completion_percentage': (len(final_results_all_folds) / len(all_patient_ids)) * 100,
            'results_so_far': final_results_all_folds
        }
        with open(progress_report_path, 'w') as f:
            json.dump(progress_data, f, indent=4, cls=NumpyEncoder)
        print(f"  ✓ Progress report saved: {progress_report_path}")
        
        # Calculate and print estimated time remaining
        if lopo_fold_idx == 0:
            fold_time = time.time() - start_time_main
            remaining_folds = len(all_patient_ids) - 1
            estimated_remaining_time = fold_time * remaining_folds / 3600
            print(f"\n  ⏱  First fold completed in {fold_time/60:.1f} minutes")
            print(f"  ⏱  Estimated time remaining: {estimated_remaining_time:.1f} hours")
        
        print(f"\n{'='*80}\n")
        
        # Save fold summary (MOVED BELOW - keep old one for compatibility)
        fold_report_path_old = os.path.join(fold_output_dir, f"LOPO_fold_{lopo_fold_idx+1}_summary.json")
        # Save fold summary (MOVED BELOW - keep old one for compatibility)
        fold_report_path_old = os.path.join(fold_output_dir, f"LOPO_fold_{lopo_fold_idx+1}_summary.json")
        with open(fold_report_path_old, 'w') as f:
            json.dump(fold_results, f, indent=4, cls=NumpyEncoder)
        print(f"\n  LOPO Fold {lopo_fold_idx+1} complete. Summary saved: {fold_report_path_old}")
    
    # Final Aggregation
    print(f"\n\n{'='*100}")
    print(f"  FINAL AGGREGATION ACROSS ALL {len(all_patient_ids)} LOPO FOLDS")
    print(f"{'='*100}\n")
    
    # Parse and aggregate results
    all_rows = []
    for fold_res in final_results_all_folds:
        fold_num = fold_res['fold']
        test_pat = fold_res['test_patient']
        for scenario_name, scenario_results in fold_res['classifier_performance'].items():
            for clf_name, clf_results in scenario_results.items():
                row = {
                    'Fold': fold_num,
                    'Test_Patient': test_pat,
                    'Scenario': scenario_name,
                    'Classifier': clf_name,
                    'Accuracy': clf_results.get('accuracy', np.nan),
                    'F1_Macro': clf_results.get('f1_macro', np.nan),
                    'F1_Weighted': clf_results.get('f1_weighted', np.nan),
                }
                all_rows.append(row)
    
    aggregated_df = pd.DataFrame(all_rows)
    aggregated_df.to_csv(os.path.join(main_run_output_dir, "CL_LOPO_all_results.csv"), index=False)
    
    # Summary statistics
    summary_stats = aggregated_df.groupby(['Scenario', 'Classifier']).agg({
        'Accuracy': ['mean', 'std'],
        'F1_Macro': ['mean', 'std'],
        'F1_Weighted': ['mean', 'std']
    }).round(4)
    
    print("\n--- CL-LOPO VALIDATION SUMMARY ---")
    print(summary_stats)
    summary_stats.to_csv(os.path.join(main_run_output_dir, "CL_LOPO_summary_statistics.csv"))
    
    total_time_hrs = (time.time() - start_time_main) / 3600
    print(f"\n{'='*100}")
    print(f"  CL-LOPO VALIDATION COMPLETE in {total_time_hrs:.2f} hours")
    print(f"  Results saved to: {main_run_output_dir}")
    print(f"{'='*100}\n")
    
    return final_results_all_folds
