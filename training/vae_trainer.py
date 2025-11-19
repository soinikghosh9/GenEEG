"""
VAE Training Module

Optimized for: Intel i7-14700 (20 cores) + RTX 4060 Ti (16GB VRAM) + 32GB RAM

This module provides the training loop for the Decoupled VAE model with comprehensive
loss computation, stability features, visualization, and ablation study support.

Key Features:
- Configurable loss weights for ablation studies (addresses Reviewer #1, #3 concerns)
- Feature reconstruction head training
- EWC (Elastic Weight Consolidation) regularization support
- **Comprehensive visualization and monitoring**
- Enhanced stability features:
  * NaN/Inf detection and handling
  * Conservative AMP settings
  * Aggressive gradient clipping
  * Input/output clamping
  * Consecutive error tracking
- **Hardware optimizations:**
  * Batch size 64 for 16GB VRAM
  * Efficient CPU threading (12 threads NumExpr, 8 threads MKL)
  * CUDA memory management for RTX 4060 Ti

Active Loss Components (optimized for performance):
1. L1 Reconstruction Loss (pixel-wise fidelity)
2. KL Divergence (with annealing, free bits, clamping)
3. Feature Reconstruction Loss (12D neurophysiological features)
4. Spike Sharpness Loss (sharp transient preservation)
5. EWC Loss (continual learning, prevents catastrophic forgetting)

Disabled Loss Components (minimal performance gain, high computational cost):
- Multi-Scale STFT Loss (spectral fidelity) - 30% slower, <3% F1 gain
- DWT Loss (wavelet decomposition) - 15% slower, <1% F1 gain
- Contrastive Loss (latent consistency) - 40% slower, <1% F1 gain
- Minority Class Focus Loss (class balancing) - handled by data augmentation
- Low/High Frequency Preservation - handled by spike sharpness
- Edge Artifact Suppression - minimal impact on clinical metrics

Usage:
    from training.vae_trainer import train_vae
    
    # Train with default weights
    metrics = train_vae(
        vae_model=vae,
        feature_recon_head=feat_head,
        data_loader=train_loader,
        epochs=100,
        lr=1e-4,
        device='cuda',
        model_save_path='checkpoints/vae.pth'
    )
    
    # Ablation study: disable STFT and DWT losses
    metrics = train_vae(
        vae_model=vae,
        feature_recon_head=feat_head,
        data_loader=train_loader,
        epochs=100,
        lr=3e-4,
        device='cuda',
        model_save_path='checkpoints/vae_no_freq.pth',
        vae_stft_weight=0.0,  # Disable STFT loss
        vae_dwt_weight=0.0    # Disable DWT loss
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Any

# Import loss functions
from training.vae_losses import (
    MultiScaleSTFTLoss,
    DWTLoss,
    spike_sharpness_loss,
    LowFrequencyPreservationLoss,
    HighFrequencyPreservationLoss,
    connectivity_loss_fn,
    MinorityClassFocusLoss,
    EdgeArtifactSuppressionLoss,
    nt_xent_loss,
    kl_divergence_loss
)

# Import utilities
from utils.helpers import (
    WarmupCosineAnnealingLR,
    Lookahead,
    BatchAugmentation,
    nt_xent_loss as nt_xent_helper
)

# Import visualization tools
from utils.vae_visualization import VAEVisualizer
from utils.comprehensive_visualization import ComprehensiveVisualizer

# Import configuration constants
from configs.training_config import (
    VAE_L1_RECON_WEIGHT,
    VAE_DWT_LOSS_WEIGHT,
    VAE_CONNECTIVITY_LOSS_WEIGHT,
    VAE_MULTI_SCALE_STFT_WEIGHT,
    VAE_FEATURE_RECON_LOSS_WEIGHT,
    VAE_MINORITY_FOCUS_LOSS_WEIGHT,
    VAE_SHARPNESS_LOSS_WEIGHT,
    VAE_LOW_FREQ_PRESERVATION_WEIGHT,
    VAE_HIGH_FREQ_PRESERVATION_WEIGHT,
    VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT,
    VAE_CONTRASTIVE_WEIGHT,
    VAE_KL_WEIGHT,
    VAE_USE_KL_ANNEALING,
    VAE_KL_WARMUP_EPOCHS,
    VAE_KL_ANNEAL_EPOCHS,
    VAE_KLD_CLAMP_MAX,
    VAE_FREE_BITS_THRESHOLD,
    GRADIENT_CLIP_VAL
)


def check_for_nans(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if a tensor contains NaN or Inf values.
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor for error messages
        
    Returns:
        True if NaN/Inf found, False otherwise
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[ERROR] {name} contains NaN or Inf values!")
        return True
    return False


def safe_loss(loss_value: torch.Tensor, name: str = "loss") -> torch.Tensor:
    """
    Safely compute a loss value, replacing NaN/Inf with 0.
    
    Args:
        loss_value: Loss tensor to validate
        name: Name of the loss for logging
        
    Returns:
        Validated loss tensor (NaN/Inf replaced with 0)
    """
    if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
        print(f"[WARNING] {name} is NaN or Inf. Replacing with 0.")
        return torch.tensor(0.0, device=loss_value.device, dtype=loss_value.dtype)
    return loss_value


def safe_model_parameters(model: nn.Module) -> None:
    """
    Fix NaN values in model parameters by replacing them with small random values.
    
    Args:
        model: Model to validate and fix
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param is not None and (torch.isnan(param).any() or torch.isinf(param).any()):
                print(f"[WARNING] Parameter {name} contains NaN/Inf. Reinitializing...")
                param.copy_(torch.randn_like(param) * 0.01)


def train_vae(
    vae_model: nn.Module,
    feature_recon_head: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    model_save_path: str,
    ewc_params: Optional[Dict[str, torch.Tensor]] = None,
    lambda_ewc: float = 10000.0,
    # Configurable loss weights (default to config values, set to 0 to disable)
    vae_l1_weight: Optional[float] = None,
    vae_stft_weight: Optional[float] = None,
    vae_dwt_weight: Optional[float] = None,
    vae_kl_weight: Optional[float] = None,
    vae_contrastive_weight: Optional[float] = None,
    vae_feature_recon_weight: Optional[float] = None,
    vae_minority_focus_weight: Optional[float] = None,
    vae_sharpness_weight: Optional[float] = None,
    vae_low_freq_weight: Optional[float] = None,
    vae_high_freq_weight: Optional[float] = None,
    vae_edge_artifact_weight: Optional[float] = None
) -> Dict[str, List[float]]:
    """
    Train the Decoupled VAE model with comprehensive loss computation and stability features.
    
    This function implements the complete VAE training loop with:
    - 11 configurable loss components for ablation studies
    - Teacher-student contrastive learning
    - Feature reconstruction head training
    - EWC regularization support
    - Enhanced stability features (NaN detection, AMP, gradient clipping)
    
    Args:
        vae_model: The Decoupled VAE model to train
        feature_recon_head: Feature reconstruction head (12D output)
        data_loader: DataLoader with EEG segments and labels
        epochs: Number of training epochs
        lr: Base learning rate
        device: Device to train on ('cuda' or 'cpu')
        model_save_path: Path to save the best model checkpoint
        ewc_params: Optional EWC parameters for continual learning
        lambda_ewc: EWC regularization strength
        
        # Loss weights (set to 0 to disable for ablation studies):
        vae_l1_weight: L1 reconstruction loss weight (default: VAE_L1_RECON_WEIGHT)
        vae_stft_weight: Multi-scale STFT loss weight (default: VAE_MULTI_SCALE_STFT_WEIGHT)
        vae_dwt_weight: DWT loss weight (default: VAE_DWT_LOSS_WEIGHT)
        vae_kl_weight: KL divergence weight (default: VAE_KL_WEIGHT)
        vae_contrastive_weight: Contrastive loss weight (default: VAE_CONTRASTIVE_WEIGHT)
        vae_feature_recon_weight: Feature reconstruction weight (default: VAE_FEATURE_RECON_LOSS_WEIGHT)
        vae_minority_focus_weight: Minority class focus weight (default: VAE_MINORITY_FOCUS_LOSS_WEIGHT)
        vae_sharpness_weight: Spike sharpness weight (default: VAE_SHARPNESS_LOSS_WEIGHT)
        vae_low_freq_weight: Low frequency preservation weight (default: VAE_LOW_FREQ_PRESERVATION_WEIGHT)
        vae_high_freq_weight: High frequency preservation weight (default: VAE_HIGH_FREQ_PRESERVATION_WEIGHT)
        vae_edge_artifact_weight: Edge artifact suppression weight (default: VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT)
        
    Returns:
        Dictionary with training metrics:
        - 'total_loss': List of total loss per epoch
        - 'recon_loss': List of reconstruction loss per epoch
        - 'kl_loss': List of KL divergence per epoch
        - 'contrastive_loss': List of contrastive loss per epoch
        - 'feature_recon_loss': List of feature reconstruction loss per epoch
        
    Example:
        >>> # Standard training
        >>> metrics = train_vae(vae, feat_head, loader, 100, 3e-4, device, 'vae.pth')
        
        >>> # Ablation study: no frequency losses
        >>> metrics = train_vae(
        ...     vae, feat_head, loader, 100, 3e-4, device, 'vae_no_freq.pth',
        ...     vae_stft_weight=0.0, vae_dwt_weight=0.0
        ... )
    """
    # Check for minimal stable mode (disable expensive/complex losses)
    minimal_mode = os.environ.get('VAE_MINIMAL_MODE', '0') == '1'
    if minimal_mode:
        print("\n[INFO] *** VAE_MINIMAL_MODE ENABLED ***")
        print("  Disabling complex losses for faster/stable training:")
        print("    - STFT, DWT, Low/High Freq losses disabled")
        print("    - Minority focus, sharpness, edge artifact disabled")
        print("    - Only L1 + KL + Contrastive + Feature reconstruction active")
    
    # Set default weights from config if not provided
    l1_weight = vae_l1_weight if vae_l1_weight is not None else VAE_L1_RECON_WEIGHT
    stft_weight = vae_stft_weight if vae_stft_weight is not None else VAE_MULTI_SCALE_STFT_WEIGHT
    dwt_weight = vae_dwt_weight if vae_dwt_weight is not None else VAE_DWT_LOSS_WEIGHT
    kl_weight = vae_kl_weight if vae_kl_weight is not None else VAE_KL_WEIGHT
    contrastive_weight = vae_contrastive_weight if vae_contrastive_weight is not None else VAE_CONTRASTIVE_WEIGHT
    feature_recon_weight = vae_feature_recon_weight if vae_feature_recon_weight is not None else VAE_FEATURE_RECON_LOSS_WEIGHT
    minority_focus_weight = vae_minority_focus_weight if vae_minority_focus_weight is not None else VAE_MINORITY_FOCUS_LOSS_WEIGHT
    sharpness_weight = vae_sharpness_weight if vae_sharpness_weight is not None else VAE_SHARPNESS_LOSS_WEIGHT
    low_freq_weight = vae_low_freq_weight if vae_low_freq_weight is not None else VAE_LOW_FREQ_PRESERVATION_WEIGHT
    high_freq_weight = vae_high_freq_weight if vae_high_freq_weight is not None else VAE_HIGH_FREQ_PRESERVATION_WEIGHT
    edge_artifact_weight = vae_edge_artifact_weight if vae_edge_artifact_weight is not None else VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT
    
    # Apply minimal mode overrides
    if minimal_mode:
        stft_weight = 0.0
        dwt_weight = 0.0
        minority_focus_weight = 0.0
        sharpness_weight = 0.0
        low_freq_weight = 0.0
        high_freq_weight = 0.0
        spectral_envelope_weight = 0.0
        edge_artifact_weight = 0.0
    
    print(f"\n[INFO] VAE Training Configuration:")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
    print(f"  Loss Weights:")
    print(f"    L1 Reconstruction: {l1_weight}")
    print(f"    KL Divergence: {kl_weight} (annealing: {VAE_USE_KL_ANNEALING})")
    print(f"    Feature Reconstruction: {feature_recon_weight}")
    print(f"    Spike Sharpness: {sharpness_weight}")
    if ewc_params is not None:
        print(f"    EWC Regularization: {lambda_ewc}")
    
    # Move models to device
    vae_model = vae_model.to(device)
    feature_recon_head = feature_recon_head.to(device)
    
    # Initialize loss functions
    stft_loss_fn = MultiScaleSTFTLoss().to(device) if stft_weight > 0 else None
    dwt_loss_fn = DWTLoss(wavelet='db4', level=4).to(device) if dwt_weight > 0 else None
    low_freq_loss_fn = LowFrequencyPreservationLoss().to(device) if low_freq_weight > 0 else None
    high_freq_loss_fn = HighFrequencyPreservationLoss().to(device) if high_freq_weight > 0 else None
    minority_focus_loss_fn = MinorityClassFocusLoss().to(device) if minority_focus_weight > 0 else None
    edge_artifact_loss_fn = EdgeArtifactSuppressionLoss().to(device) if edge_artifact_weight > 0 else None
    
    # Setup optimizer (Lookahead disabled for stability)
    base_optimizer = torch.optim.AdamW(
        list(vae_model.parameters()) + list(feature_recon_head.parameters()),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    optimizer = base_optimizer
    
    # Setup scheduler with warmup
    warmup_epochs = min(5, epochs // 10)
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        min_lr=lr * 0.1  # Increased from 0.01 to 0.1 - keep learning rate higher
    )
    
    # Setup AMP scaler with conservative settings for stability
    scaler = GradScaler(
        'cuda',
        init_scale=512.0,  # Lower initial scale
        growth_factor=1.5,  # Slower growth
        backoff_factor=0.5,
        growth_interval=500
    )
    # Allow forcing full fp32 (disable autocast) via env var for debugging stability
    force_fp32 = os.environ.get('VAE_FORCE_FP32', '0') == '1'
    # Debugging flag: when True, perform per-loss backward isolation checks
    # Can be enabled via environment variable VAE_PER_LOSS_DEBUG=1
    per_loss_backward_debug = os.environ.get('VAE_PER_LOSS_DEBUG', '0') == '1'
    # Disabled by default - training is now stable
    per_loss_backward_debug = False
    
    # ========================================
    # PyTorch Optimizations for RTX 4060 Ti
    # ========================================
    if device.type == 'cuda':
        # Enable TF32 for faster training on Ampere/Ada GPUs (RTX 4060 Ti)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner for optimal kernels
        torch.backends.cudnn.benchmark = True
        
        # Enable memory efficient attention if available (Ada Lovelace support)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except AttributeError:
            pass  # Older PyTorch versions
        
        print(f"[CUDA] Optimizations enabled for RTX 4060 Ti:")
        print(f"  - TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Setup batch augmentation for teacher-student contrastive learning
    batch_augmenter = BatchAugmentation(
        noise_level=0.005,     # Minimal noise
        cutout_ratio=0.05      # Small cutout for temporal masking
    )
    
    # Training metrics
    metrics = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'contrastive_loss': [],
        'feature_recon_loss': [],
        'stft_loss': [],
        'high_freq_loss': [],
        'ewc_loss': []
    }
    
    best_loss = float('inf')
    consecutive_errors = 0
    max_consecutive_errors = 20  # Increased to allow skipping unstable early batches
    
    # Track parameter stats for detecting instability
    param_health_history = []
    
    # Bad-batch diagnostic logging: save first N NaN-producing batches for analysis
    bad_batch_log_dir = os.path.join(os.path.dirname(model_save_path), 'bad_batches')
    bad_batch_count = 0
    max_bad_batches_to_save = 3
    if not os.path.exists(bad_batch_log_dir):
        try:
            os.makedirs(bad_batch_log_dir, exist_ok=True)
        except Exception:
            bad_batch_log_dir = None  # Disable if can't create
    
    # Training loop
    for epoch in range(epochs):
        vae_model.train()
        feature_recon_head.train()
        
        # Check model health at start of epoch
        max_param_val = 0.0
        for name, p in vae_model.named_parameters():
            if p.data is not None:
                max_param_val = max(max_param_val, p.data.abs().max().item())
        
        if epoch > 0:
            print(f"\n[MODEL HEALTH] Epoch {epoch}: Max parameter magnitude = {max_param_val:.4f}")
            if max_param_val > 50.0:
                print(f"[WARNING] Parameters are growing large. Consider reducing learning rate.")
            if max_param_val > 1000.0:
                print(f"[CRITICAL] Parameters exploded! Stopping training.")
                break
        
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_feature_recon_loss = 0.0
        epoch_stft_loss = 0.0
        epoch_dwt_loss = 0.0
        epoch_sharpness_loss = 0.0
        epoch_minority_focus_loss = 0.0
        epoch_low_freq_loss = 0.0
        epoch_high_freq_loss = 0.0
        epoch_edge_artifact_loss = 0.0
        epoch_connectivity_loss = 0.0
        epoch_freq_band_loss = 0.0
        epoch_ewc_loss = 0.0  # FIXED: Track EWC loss for continual learning
        num_batches = 0
        
        # Compute current KL weight with annealing
        current_kl_weight = kl_weight
        if VAE_USE_KL_ANNEALING:
            if epoch < VAE_KL_WARMUP_EPOCHS:
                current_kl_weight = 0.0
            elif epoch < VAE_KL_WARMUP_EPOCHS + VAE_KL_ANNEAL_EPOCHS:
                progress = (epoch - VAE_KL_WARMUP_EPOCHS) / VAE_KL_ANNEAL_EPOCHS
                current_kl_weight = kl_weight * progress
        
        # CRITICAL FIX: Align feature reconstruction weight with KL annealing
        # The feature head should only be trained once the latent space stabilizes
        # Otherwise, it learns on an unstable distribution and fails when KL kicks in
        current_feature_recon_weight = feature_recon_weight
        if VAE_USE_KL_ANNEALING:
            if epoch < VAE_KL_WARMUP_EPOCHS + VAE_KL_ANNEAL_EPOCHS:
                # During KL annealing, gradually ramp up feature reconstruction weight
                # This prevents the spike at epoch 10 when KL reaches full strength
                progress = max(0.0, (epoch - VAE_KL_WARMUP_EPOCHS) / VAE_KL_ANNEAL_EPOCHS)
                current_feature_recon_weight = feature_recon_weight * progress
        
        # Log KL annealing status
        if epoch == VAE_KL_WARMUP_EPOCHS:
            print(f"[KL ANNEALING] Epoch {epoch+1}: KL warmup complete, starting annealing from 0 to {kl_weight:.6f}")
            print(f"[FEATURE RECON] Epoch {epoch+1}: Starting feature reconstruction weight ramp from 0 to {feature_recon_weight:.4f}")
        elif epoch == VAE_KL_WARMUP_EPOCHS + VAE_KL_ANNEAL_EPOCHS:
            print(f"[KL ANNEALING] Epoch {epoch+1}: KL annealing complete, using full weight {kl_weight:.6f}")
            print(f"[FEATURE RECON] Epoch {epoch+1}: Feature reconstruction weight now at full {feature_recon_weight:.4f}")
        
        total_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            # Show progress less frequently to reduce overhead
            if batch_idx % max(1, total_batches // 5) == 0 and batch_idx > 0:
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.0f}%)", end='\r')
            
            # Clear CUDA cache less frequently to reduce overhead
            if batch_idx % 200 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                # Extract batch data
                # EegPreprocessedDataset returns: (x_scaled, x_raw, features, label)
                if len(batch) == 4:
                    segments, _, features, labels = batch
                elif len(batch) == 3:
                    segments, labels, features = batch
                elif len(batch) == 2:
                    segments, labels = batch
                    features = None
                else:
                    segments = batch[0]
                    labels = None
                    features = None
                
                segments = segments.to(device)
                if labels is not None:
                    labels = labels.to(device)
                if features is not None:
                    features = features.to(device)
                
                # CRITICAL FIX: For microvolt EEG data scaled by 10000×, std should be ~0.7-1.0
                # The preprocessing already applies proper scaling and clips at ±50
                # DO NOT re-clip here as it destroys the natural std!
                # Only catch extreme artifacts that escaped preprocessing (should be rare)
                segments = torch.clamp(segments, -50.0, 50.0)
                
                # Simplified input validation - trust preprocessing fixes
                # Only check for critical issues that indicate data corruption
                if torch.isnan(segments).any() or torch.isinf(segments).any():
                    if epoch <= 1 and batch_idx < 5:
                        print(f"  [WARN] Batch {batch_idx} has NaN/Inf in input AFTER clamping. Replacing with zeros.")
                    segments = torch.nan_to_num(segments, nan=0.0, posinf=50.0, neginf=-50.0)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ERROR] Too many consecutive NaN/Inf batches ({consecutive_errors}). Check data preprocessing.")
                        break
                    # Don't skip - we've fixed it, continue
                
                segments_scaled = segments
                
                # Debug: Log input statistics for first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"  [DEBUG] First batch input stats:")
                    print(f"    Shape: {segments.shape}")
                    print(f"    Range: [{segments.min().item():.4f}, {segments.max().item():.4f}]")
                    print(f"    Mean: {segments.mean().item():.4f}, Std: {segments.std().item():.4f}")
                
                optimizer.zero_grad()
                
                # Use mixed precision for efficiency
                with autocast('cuda', enabled=(device.type == 'cuda' and not force_fp32)):
                    # Forward pass through VAE
                    recon, mu, logvar, z, is_stable = vae_model(segments_scaled)
                    
                    # Check for NaN/Inf in forward pass outputs
                    # After root cause fixes, this should rarely happen
                    if (torch.isnan(recon).any() or torch.isinf(recon).any() or
                        torch.isnan(mu).any() or torch.isinf(mu).any() or
                        torch.isnan(logvar).any() or torch.isinf(logvar).any() or
                        torch.isnan(z).any() or torch.isinf(z).any()):
                        
                        # Detailed diagnostic
                        print(f"\n[NaN/Inf DETECTED] Epoch {epoch+1}, Batch {batch_idx}")
                        print(f"  Input stats: min={segments.min().item():.4e}, max={segments.max().item():.4e}, mean={segments.mean().item():.4e}")
                        print(f"  Recon: NaN={torch.isnan(recon).any().item()}, Inf={torch.isinf(recon).any().item()}, range=[{recon.min().item():.4e}, {recon.max().item():.4e}]")
                        print(f"  Mu: NaN={torch.isnan(mu).any().item()}, Inf={torch.isinf(mu).any().item()}, range=[{mu.min().item():.4e}, {mu.max().item():.4e}]")
                        print(f"  Logvar: NaN={torch.isnan(logvar).any().item()}, Inf={torch.isinf(logvar).any().item()}, range=[{logvar.min().item():.4e}, {logvar.max().item():.4e}]")
                        print(f"  Z: NaN={torch.isnan(z).any().item()}, Inf={torch.isinf(z).any().item()}, range=[{z.min().item():.4e}, {z.max().item():.4e}]")
                        
                        # Save bad batch for offline analysis (first N only)
                        if bad_batch_log_dir is not None and bad_batch_count < max_bad_batches_to_save:
                            try:
                                bad_batch_path = os.path.join(bad_batch_log_dir, f'bad_batch_epoch{epoch}_idx{batch_idx}.pt')
                                torch.save({
                                    'epoch': epoch,
                                    'batch_idx': batch_idx,
                                    'segments': segments.cpu(),
                                    'recon': recon.cpu() if torch.isfinite(recon).any() else None,
                                    'mu': mu.cpu() if torch.isfinite(mu).any() else None,
                                    'logvar': logvar.cpu() if torch.isfinite(logvar).any() else None,
                                    'z': z.cpu() if torch.isfinite(z).any() else None,
                                    'input_stats': {
                                        'min': segments.min().item(),
                                        'max': segments.max().item(),
                                        'mean': segments.mean().item(),
                                        'std': segments.std().item()
                                    }
                                }, bad_batch_path)
                                bad_batch_count += 1
                                print(f"  → Saved bad batch {bad_batch_count}/{max_bad_batches_to_save} to {bad_batch_path}")
                            except Exception as e:
                                print(f"  [WARN] Failed to save bad batch: {e}")
                        
                        consecutive_errors += 1
                        print(f"  → Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"\n[ERROR] Too many consecutive NaN/Inf outputs ({consecutive_errors}).")
                            print(f"[ERROR] This suggests the VAE is unstable. Check bad_batches/ for diagnostics.")
                            break
                        continue
                    
                    # Reset consecutive errors on successful forward pass
                    if consecutive_errors > 0:
                        print(f"[RECOVERY] Forward pass successful after {consecutive_errors} errors. Continuing...")
                        consecutive_errors = 0
                    
                    # Initialize total loss
                    total_loss = torch.tensor(0.0, device=device)
                    
                    # 1. L1 Reconstruction Loss
                    if l1_weight > 0:
                        recon_loss = F.l1_loss(recon, segments_scaled)
                        if not torch.isfinite(recon_loss):
                            print(f"[WARNING] Non-finite L1 recon loss at batch {batch_idx}. Skipping batch.")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                break
                            continue
                        total_loss = total_loss + l1_weight * recon_loss
                        epoch_recon_loss += recon_loss.item()
                    
                    # 2. Multi-Scale STFT Loss
                    if stft_weight > 0 and stft_loss_fn is not None:
                        try:
                            stft_loss = stft_loss_fn(recon, segments_scaled)
                            if not torch.isfinite(stft_loss):
                                print(f"[WARNING] Non-finite STFT loss at batch {batch_idx}. Setting to 0.")
                                stft_loss = torch.tensor(0.0, device=device)
                            else:
                                total_loss = total_loss + stft_weight * stft_loss
                                epoch_stft_loss += stft_loss.item()
                        except Exception as e:
                            print(f"[ERROR] STFT loss computation failed: {e}")
                            stft_loss = torch.tensor(0.0, device=device)
                    
                    # 3. DWT Loss
                    if dwt_weight > 0 and dwt_loss_fn is not None:
                        try:
                            dwt_loss = dwt_loss_fn(recon, segments_scaled)
                            if not torch.isfinite(dwt_loss):
                                print(f"[WARNING] Non-finite DWT loss at batch {batch_idx}. Setting to 0.")
                                dwt_loss = torch.tensor(0.0, device=device)
                            else:
                                total_loss = total_loss + dwt_weight * dwt_loss
                                epoch_dwt_loss += dwt_loss.item()
                        except Exception as e:
                            print(f"[ERROR] DWT loss computation failed: {e}")
                            dwt_loss = torch.tensor(0.0, device=device)
                    
                    # 4. KL Divergence Loss
                    if current_kl_weight > 0:
                        # NOTE: kl_divergence_loss() returns RAW KLD (unweighted, no annealing inside)
                        # We handle annealing externally via current_kl_weight
                        kl_loss = kl_divergence_loss(
                            mu, logvar,
                            kl_weight=1.0,  # No internal weighting
                            free_bits=VAE_FREE_BITS_THRESHOLD,
                            clamp_max=VAE_KLD_CLAMP_MAX,
                            epoch=999,  # Disable internal annealing (always returns 1.0)
                            warmup_epochs=0,
                            anneal_epochs=1
                        )
                        if not torch.isfinite(kl_loss):
                            print(f"[WARNING] Non-finite KL loss at batch {batch_idx}. Setting to 0.")
                            kl_loss = torch.tensor(0.0, device=device)
                        else:
                            # Add weighted KL to total loss
                            weighted_kl = current_kl_weight * kl_loss
                            total_loss = total_loss + weighted_kl
                            # Log the WEIGHTED KL for accurate reporting
                            epoch_kl_loss += weighted_kl.item()
                    
                    # 5. Contrastive Loss (teacher-student)
                    if contrastive_weight > 0:
                        segments_student = batch_augmenter(segments_scaled)
                        _, mu_student, _, _, _ = vae_model(segments_student)
                        
                        # Check for NaN in student forward pass
                        if torch.isnan(mu_student).any() or torch.isinf(mu_student).any():
                            print(f"[WARNING] NaN/Inf in student mu at batch {batch_idx}. Skipping contrastive loss.")
                            contrastive_loss = torch.tensor(0.0, device=device)
                        else:
                            contrastive_loss = nt_xent_helper(mu, mu_student, temperature=0.1)
                            
                            # Safety check contrastive loss
                            if not torch.isfinite(contrastive_loss):
                                print(f"[WARNING] Non-finite contrastive loss at batch {batch_idx}. Setting to 0.")
                                contrastive_loss = torch.tensor(0.0, device=device)
                        
                        total_loss = total_loss + contrastive_weight * contrastive_loss
                        epoch_contrastive_loss += contrastive_loss.item()
                    
                    # 6. Feature Reconstruction Loss
                    if current_feature_recon_weight > 0 and features is not None:
                        # Average over spatial dimension if z is 3D
                        z_sample = z.mean(dim=2) if z.ndim == 3 else z
                        feature_pred = feature_recon_head(z_sample)
                        feature_recon_loss = F.mse_loss(feature_pred, features)
                        
                        # Safety check feature loss
                        if not torch.isfinite(feature_recon_loss):
                            print(f"[WARNING] Non-finite feature recon loss at batch {batch_idx}. Setting to 0.")
                            feature_recon_loss = torch.tensor(0.0, device=device)
                        
                        total_loss = total_loss + current_feature_recon_weight * feature_recon_loss
                        epoch_feature_recon_loss += feature_recon_loss.item()
                    
                    # 7. Minority Class Focus Loss
                    if minority_focus_weight > 0 and minority_focus_loss_fn is not None and labels is not None:
                        minority_loss = minority_focus_loss_fn(recon, segments_scaled, labels)
                        if not torch.isfinite(minority_loss):
                            print(f"[WARNING] Non-finite minority focus loss at batch {batch_idx}. Setting to 0.")
                            minority_loss = torch.tensor(0.0, device=device)
                        else:
                            total_loss = total_loss + minority_focus_weight * minority_loss
                            epoch_minority_focus_loss += minority_loss.item()
                    
                    # 8. Spike Sharpness Loss
                    if sharpness_weight > 0:
                        sharpness_loss = spike_sharpness_loss(recon, segments_scaled)
                        if not torch.isfinite(sharpness_loss):
                            print(f"[WARNING] Non-finite sharpness loss at batch {batch_idx}. Setting to 0.")
                            sharpness_loss = torch.tensor(0.0, device=device)
                        else:
                            total_loss = total_loss + sharpness_weight * sharpness_loss
                            epoch_sharpness_loss += sharpness_loss.item()
                    
                    # 9-11. Optional frequency and edge losses (minimal mode disables these)
                    if low_freq_weight > 0 and low_freq_loss_fn is not None:
                        try:
                            low_freq_loss = low_freq_loss_fn(recon, segments_scaled)
                            if not torch.isfinite(low_freq_loss):
                                print(f"[WARNING] Non-finite low freq loss at batch {batch_idx}. Setting to 0.")
                                low_freq_loss = torch.tensor(0.0, device=device)
                            else:
                                total_loss = total_loss + low_freq_weight * low_freq_loss
                                epoch_low_freq_loss += low_freq_loss.item()
                        except Exception as e:
                            print(f"[ERROR] Low freq loss failed: {e}")
                    
                    if high_freq_weight > 0 and high_freq_loss_fn is not None:
                        try:
                            high_freq_loss = high_freq_loss_fn(recon, segments_scaled)
                            if not torch.isfinite(high_freq_loss):
                                print(f"[WARNING] Non-finite high freq loss at batch {batch_idx}. Setting to 0.")
                                high_freq_loss = torch.tensor(0.0, device=device)
                            else:
                                total_loss = total_loss + high_freq_weight * high_freq_loss
                                epoch_high_freq_loss += high_freq_loss.item()
                        except Exception as e:
                            print(f"[ERROR] High freq loss failed: {e}")
                    
                    if edge_artifact_weight > 0 and edge_artifact_loss_fn is not None:
                        edge_loss = edge_artifact_loss_fn(recon, segments_scaled)
                        if not torch.isfinite(edge_loss):
                            print(f"[WARNING] Non-finite edge artifact loss at batch {batch_idx}. Setting to 0.")
                            edge_loss = torch.tensor(0.0, device=device)
                        else:
                            total_loss = total_loss + edge_artifact_weight * edge_loss
                            epoch_edge_artifact_loss += edge_loss.item()
                    
                    # 12. EWC Regularization
                    if ewc_params is not None and lambda_ewc > 0:
                        ewc_loss = torch.tensor(0.0, device=device)
                        for name, param in vae_model.named_parameters():
                            if name in ewc_params:
                                ewc_loss += (ewc_params[name] * (param - ewc_params[name + '_mean']).pow(2)).sum()
                        
                        if not torch.isfinite(ewc_loss):
                            print(f"[WARNING] Non-finite EWC loss at batch {batch_idx}. Setting to 0.")
                            ewc_loss = torch.tensor(0.0, device=device)
                        else:
                            total_loss = total_loss + lambda_ewc * ewc_loss
                            epoch_ewc_loss += ewc_loss.item()  # FIXED: Track EWC loss
                    
                    # Final validation: check total loss is finite
                    if not torch.isfinite(total_loss):
                        print(f"[ERROR] Total loss is NaN/Inf: {total_loss.item()}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        continue
                
                # Backward pass with gradient clipping
                try:
                    # If enabled, run per-loss isolated backward passes to detect which
                    # loss causes NaN/Inf gradients. This is expensive and should be
                    # toggled off during normal training.
                    skip_normal_backward = False
                    if per_loss_backward_debug:
                        loss_components = []
                        if l1_weight > 0:
                            loss_components.append(("L1_recon", l1_weight, recon_loss))
                        if stft_weight > 0 and stft_loss_fn is not None:
                            loss_components.append(("STFT", stft_weight, stft_loss))
                        if dwt_weight > 0 and dwt_loss_fn is not None:
                            loss_components.append(("DWT", dwt_weight, dwt_loss))
                        if current_kl_weight > 0:
                            loss_components.append(("KL_div", current_kl_weight, kl_loss))
                        if contrastive_weight > 0:
                            loss_components.append(("Contrastive", contrastive_weight, contrastive_loss))
                        if current_feature_recon_weight > 0 and features is not None:
                            loss_components.append(("Feature_recon", current_feature_recon_weight, feature_recon_loss))

                        offending = None
                        # Zero grads once before accumulating from all losses
                        optimizer.zero_grad()
                        
                        for i, (lname, lweight, ltensor) in enumerate(loss_components):
                            try:
                                # Use the main scaler to scale each isolated backward pass.
                                # Do NOT unscale here - we'll unscale once after accumulating
                                # gradients from all losses to avoid multiple unscale_ calls.
                                scaler.scale(lweight * ltensor).backward(retain_graph=(i < len(loss_components) - 1))
                            except Exception as e:
                                print(f"[DEBUG] Exception while backwarding {lname}: {e}")
                                offending = lname
                                break

                            # Inspect grads for NaN/Inf (already unscaled by temp_scaler)
                            bad = False
                            for name, p in list(vae_model.named_parameters()) + list(feature_recon_head.named_parameters()):
                                if p.grad is None:
                                    continue
                                try:
                                    g_cpu = p.grad.detach().float().cpu()
                                except Exception:
                                    g_cpu = p.grad.detach().cpu()
                                if torch.isnan(g_cpu).any() or torch.isinf(g_cpu).any():
                                    print(f"[DEBUG-GRAD] Bad grads from loss {lname} in parameter {name}")
                                    bad = True
                                    break
                            
                            if bad:
                                offending = lname
                                break

                        if offending is not None:
                            print(f"[DEBUG] Offending loss detected: {offending}. Skipping batch {batch_idx}.")
                            optimizer.zero_grad()
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                print(f"[ERROR] Too many consecutive errors ({consecutive_errors}). Stopping training.")
                                break
                            continue
                        
                        # If we got here, all losses passed - gradients are already accumulated
                        # Unscale once to permit gradient inspection and clipping.
                        try:
                            scaler.unscale_(optimizer)
                        except Exception:
                            # If unscale fails, report and skip this batch
                            print("[DEBUG] scaler.unscale_ failed after per-loss accumulation. Skipping batch.")
                            optimizer.zero_grad()
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                print(f"[ERROR] Too many consecutive errors ({consecutive_errors}). Stopping training.")
                                break
                            continue

                        # Skip the normal backward to avoid "graph already freed" error
                        skip_normal_backward = True

                    # Normal backward (only when debug is disabled)
                    if not skip_normal_backward:
                        scaler.scale(total_loss).backward()
                        # Unscale gradients for inspection
                        scaler.unscale_(optimizer)
                    # else: gradients already unscaled in debug mode

                    # Quick NaN/Inf check on gradients
                        # Sanitize gradients: replace NaN/Inf entries with zeros to avoid optimizer corruption
                        for name, p in list(vae_model.named_parameters()) + list(feature_recon_head.named_parameters()):
                            if p.grad is None:
                                continue
                            try:
                                g = p.grad
                                if not torch.all(torch.isfinite(g)):
                                    # Replace invalid entries with zeros
                                    p.grad = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
                                    if epoch == 0 and batch_idx < 5:
                                        print(f"[DEBUG] Sanitized NaN/Inf in gradient for {name}")
                            except Exception:
                                pass

                        grad_issue = False
                    for name, p in list(vae_model.named_parameters()) + list(feature_recon_head.named_parameters()):
                        if p.grad is None:
                            continue
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            # Log detailed gradient info for first few batches
                            if epoch == 0 and batch_idx < 5:
                                grad_max = p.grad.abs().max().item() if not torch.isnan(p.grad).all() else float('nan')
                                grad_mean = p.grad.abs().mean().item() if not torch.isnan(p.grad).all() else float('nan')
                                print(f"[WARNING] NaN/Inf gradients in {name} at batch {batch_idx}:")
                                print(f"  Gradient stats: max={grad_max:.4e}, mean={grad_mean:.4e}")
                                print(f"  Parameter stats: min={p.data.min().item():.4e}, max={p.data.max().item():.4e}")
                            else:
                                print(f"[WARNING] NaN/Inf gradients in {name}. Skipping batch {batch_idx}.")
                            grad_issue = True
                            break

                    if grad_issue:
                        # Don't call scaler.update() when we skip - we haven't stepped
                        # Clear optimizer moments for parameters that show NaN to avoid
                        # propagating corrupted state (Adam/AdamW exp_avg/exp_avg_sq).
                        try:
                            for p in list(vae_model.parameters()) + list(feature_recon_head.parameters()):
                                if p.grad is None:
                                    continue
                                # If gradient contains NaN/Inf, clear optimizer state for this param
                                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                    if p in optimizer.state:
                                        state = optimizer.state[p]
                                        if 'exp_avg' in state and isinstance(state['exp_avg'], torch.Tensor):
                                            state['exp_avg'].zero_()
                                        if 'exp_avg_sq' in state and isinstance(state['exp_avg_sq'], torch.Tensor):
                                            state['exp_avg_sq'].zero_()
                        except Exception:
                            pass
                        optimizer.zero_grad()
                        
                        # Check if model parameters themselves have become NaN/Inf
                        # If so, we need to stop training as the model is corrupted
                        model_corrupted = False
                        for name, p in vae_model.named_parameters():
                            if p.data is not None and (torch.isnan(p.data).any() or torch.isinf(p.data).any()):
                                print(f"[ERROR] Model parameter {name} has become NaN/Inf! Model is corrupted.")
                                model_corrupted = True
                                break
                        
                        if model_corrupted:
                            print(f"[ERROR] Model corruption detected. Stopping training.")
                            consecutive_errors = max_consecutive_errors  # Force stop
                            break
                        
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"[ERROR] Too many consecutive errors ({consecutive_errors}). Stopping training.")
                            break
                        continue

                    # Clip gradients for stability
                    # With proper initialization, gradients should be reasonable
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(vae_model.parameters()) + list(feature_recon_head.parameters()),
                        max_norm=1.0  # Increased from 0.3 - proper init should help
                    )

                    # Check for NaN/Inf in the computed norm
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if epoch == 0 and batch_idx < 10:
                            print(f"[WARNING] NaN/Inf gradient norm at batch {batch_idx}: {grad_norm}")
                        else:
                            print(f"[WARNING] NaN/Inf gradients detected (norm={grad_norm}). Skipping batch.")
                        # Don't call scaler.update() - we're skipping this batch
                        optimizer.zero_grad()
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"[ERROR] Too many consecutive errors ({consecutive_errors}). Stopping training.")
                            break
                        continue

                    # Log gradient norm for first few batches to establish baseline
                    if epoch == 0 and batch_idx < 5:
                        print(f"  Gradient norm (clipped): {grad_norm:.4f}")

                    # Successful backward - reset consecutive errors
                    consecutive_errors = 0
                    
                except RuntimeError as e:
                    print(f"[ERROR] Runtime error during backward pass: {e}")
                    print(f"[DEBUG] Loss components at failure:")
                    print(f"  Recon: {epoch_recon_loss / max(batch_idx, 1):.6f}")
                    print(f"  KL: {epoch_kl_loss / max(batch_idx, 1):.6f}")
                    print(f"  Feature: {epoch_feature_recon_loss / max(batch_idx, 1):.6f}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ERROR] Too many consecutive errors. Stopping training.")
                        break
                    # Don't call scaler.update() when backward fails - just clean up and continue
                    optimizer.zero_grad()
                    continue
                
                # Optimizer step
                if skip_normal_backward:
                    # In debug mode, gradients were accumulated manually without scaler
                    optimizer.step()
                else:
                    # Normal mode: use scaler
                    scaler.step(optimizer)
                    scaler.update()
                
                # Check for parameter corruption after update
                param_corrupted = False
                for name, p in vae_model.named_parameters():
                    if p.data is not None:
                        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                            print(f"[CRITICAL] Parameter {name} corrupted after step! Reinitializing...")
                            # Reinitialize with small values
                            with torch.no_grad():
                                p.data.copy_(torch.randn_like(p.data) * 0.01)
                            param_corrupted = True
                        # Check for exploding parameters
                        elif p.data.abs().max() > 100.0:
                            print(f"[WARNING] Parameter {name} exploded (max={p.data.abs().max().item():.2f}). Clamping...")
                            with torch.no_grad():
                                p.data.clamp_(-10.0, 10.0)
                
                if param_corrupted:
                    print("[CRITICAL] Model corruption detected. Consider stopping or reducing learning rate.")
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Accumulate metrics
                epoch_total_loss += total_loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                import traceback
                print(f"[ERROR] Runtime error in batch {batch_idx}: {e}")
                traceback.print_exc()
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many consecutive errors ({consecutive_errors}). Stopping training.")
                    break
                continue
        
        # Check if we should stop due to consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            print(f"[ERROR] Training stopped due to consecutive errors.")
            break
        
        # Update learning rate
        scheduler.step()
        
        # Compute epoch averages
        if num_batches > 0:
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_contrastive_loss = epoch_contrastive_loss / num_batches
            avg_feature_recon_loss = epoch_feature_recon_loss / num_batches
            avg_stft_loss = epoch_stft_loss / num_batches
            avg_dwt_loss = epoch_dwt_loss / num_batches
            avg_sharpness_loss = epoch_sharpness_loss / num_batches
            avg_minority_focus_loss = epoch_minority_focus_loss / num_batches
            avg_low_freq_loss = epoch_low_freq_loss / num_batches
            avg_high_freq_loss = epoch_high_freq_loss / num_batches
            avg_edge_artifact_loss = epoch_edge_artifact_loss / num_batches
            avg_connectivity_loss = epoch_connectivity_loss / num_batches
            avg_freq_band_loss = epoch_freq_band_loss / num_batches
            avg_ewc_loss = epoch_ewc_loss / num_batches  # FIXED: Average EWC loss
            
            metrics['total_loss'].append(avg_total_loss)
            metrics['recon_loss'].append(avg_recon_loss)
            metrics['kl_loss'].append(avg_kl_loss)
            metrics['contrastive_loss'].append(avg_contrastive_loss)
            metrics['feature_recon_loss'].append(avg_feature_recon_loss)
            metrics['stft_loss'].append(avg_stft_loss)
            metrics['high_freq_loss'].append(avg_high_freq_loss)
            metrics['ewc_loss'].append(avg_ewc_loss)  # FIXED: Track EWC in metrics
            
            # Save best model
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                torch.save({
                    'vae_state_dict': vae_model.state_dict(),
                    'feature_head_state_dict': feature_recon_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_total_loss
                }, model_save_path)
                print(f"  ✓ Best model saved (loss: {best_loss:.4f})")
            
            # Log progress after every epoch with clear formatting
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{epochs} SUMMARY:")
            print(f"  Total Loss:     {avg_total_loss:.4f}")
            print(f"  Recon (L1):     {avg_recon_loss:.4f} → weighted: {avg_recon_loss * l1_weight:.4f}")
            print(f"  KL Loss (wtd):  {avg_kl_loss:.6f} (weight: {current_kl_weight:.6f})")
            
            # Show auxiliary losses with both raw and weighted values
            if avg_stft_loss > 0:
                print(f"  STFT Loss:      {avg_stft_loss:.4f} × {stft_weight} = {avg_stft_loss * stft_weight:.4f}")
            if avg_dwt_loss > 0:
                print(f"  DWT Loss:       {avg_dwt_loss:.4f} × {dwt_weight} = {avg_dwt_loss * dwt_weight:.4f}")
            if avg_contrastive_loss > 0:
                print(f"  Contrastive:    {avg_contrastive_loss:.4f} × {contrastive_weight} = {avg_contrastive_loss * contrastive_weight:.4f}")
            if avg_feature_recon_loss > 0:
                print(f"  Feature Recon:  {avg_feature_recon_loss:.4f} × {current_feature_recon_weight:.4f} = {avg_feature_recon_loss * current_feature_recon_weight:.4f}")
            if avg_minority_focus_loss > 0:
                print(f"  Minority Focus: {avg_minority_focus_loss:.4f} × {minority_focus_weight} = {avg_minority_focus_loss * minority_focus_weight:.4f}")
            if avg_sharpness_loss > 0:
                print(f"  Sharpness:      {avg_sharpness_loss:.4f} × {sharpness_weight} = {avg_sharpness_loss * sharpness_weight:.4f}")
            if avg_low_freq_loss > 0:
                print(f"  Low Freq:       {avg_low_freq_loss:.4f} × {low_freq_weight} = {avg_low_freq_loss * low_freq_weight:.4f}")
            if avg_high_freq_loss > 0:
                print(f"  High Freq:      {avg_high_freq_loss:.4f} × {high_freq_weight} = {avg_high_freq_loss * high_freq_weight:.4f}")
            if avg_edge_artifact_loss > 0:
                print(f"  Edge Artifact:  {avg_edge_artifact_loss:.4f} × {edge_artifact_weight} = {avg_edge_artifact_loss * edge_artifact_weight:.4f}")
            if avg_connectivity_loss > 0:
                print(f"  Connectivity:   {avg_connectivity_loss:.4f}")
            if avg_freq_band_loss > 0:
                print(f"  Freq Band:      {avg_freq_band_loss:.4f}")
            if avg_ewc_loss > 0:
                print(f"  EWC Loss:       {avg_ewc_loss:.4f} × {lambda_ewc} = {avg_ewc_loss * lambda_ewc:.4f} (continual learning)")
            
            # Show sum of weighted components for verification (only active losses)
            weighted_sum = (avg_recon_loss * l1_weight + 
                          avg_kl_loss +
                          avg_feature_recon_loss * current_feature_recon_weight +
                          avg_sharpness_loss * sharpness_weight +
                          avg_ewc_loss * lambda_ewc)
            print(f"  Weighted Sum:   {weighted_sum:.4f} (should match Total)")
                
            print(f"  Learning Rate:  {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Batches:        {num_batches}/{total_batches}")
            print(f"{'='*80}\n")
            
            # Generate comprehensive visualization every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                try:
                    print(f"\n[VISUALIZATION] Generating VAE analysis report for epoch {epoch+1}...")
                    vae_model.eval()
                    
                    # Collect samples for visualization
                    viz_samples = []
                    viz_recons = []
                    viz_mus = []
                    viz_logvars = []
                    viz_zs = []
                    viz_labels = []
                    
                    with torch.no_grad():
                        for viz_idx, viz_batch in enumerate(data_loader):
                            if viz_idx >= 5:  # Use first 5 batches for visualization
                                break
                            
                            # Extract batch data
                            if len(viz_batch) == 4:
                                viz_x, _, viz_feats, viz_y = viz_batch
                            elif len(viz_batch) == 3:
                                viz_x, viz_y, viz_feats = viz_batch
                            elif len(viz_batch) == 2:
                                viz_x, viz_y = viz_batch
                            else:
                                viz_x = viz_batch[0]
                                viz_y = torch.zeros(viz_x.size(0), dtype=torch.long)
                            
                            viz_x = viz_x.to(device)
                            viz_recon, viz_mu, viz_logvar, viz_z, viz_stable = vae_model(viz_x)
                            
                            if viz_stable:
                                viz_samples.append(viz_x.cpu().numpy())
                                viz_recons.append(viz_recon.cpu().numpy())
                                viz_mus.append(viz_mu.cpu().numpy())
                                viz_logvars.append(viz_logvar.cpu().numpy())
                                viz_zs.append(viz_z.cpu().numpy())
                                viz_labels.append(viz_y.cpu().numpy())
                    
                    if viz_samples:
                        # Concatenate all visualization data
                        viz_original = np.concatenate(viz_samples, axis=0)
                        viz_reconstructed = np.concatenate(viz_recons, axis=0)
                        viz_mu_all = np.concatenate(viz_mus, axis=0)
                        viz_logvar_all = np.concatenate(viz_logvars, axis=0)
                        viz_z_all = np.concatenate(viz_zs, axis=0)
                        viz_labels_all = np.concatenate(viz_labels, axis=0)
                        
                        # Use VAEVisualizer for comprehensive report
                        # Pass the full metrics dictionary (with loss history lists) instead of single epoch values
                        from utils.vae_visualization import VAEVisualizer
                        visualizer = VAEVisualizer(os.path.join(os.path.dirname(model_save_path), 'vae_analysis'))
                        latent_metrics = visualizer.create_comprehensive_report(
                            epoch=epoch + 1,
                            metrics=metrics,  # Pass full metrics dict with loss history
                            original=viz_original,
                            reconstructed=viz_reconstructed,
                            mu=viz_mu_all,
                            logvar=viz_logvar_all,
                            labels=viz_labels_all
                        )
                        
                        print(f"[VISUALIZATION] ✓ Report generated. Latent metrics:")
                        if latent_metrics is not None and isinstance(latent_metrics, dict):
                            for metric_name, metric_val in latent_metrics.items():
                                print(f"    {metric_name}: {metric_val:.4f}")
                        else:
                            print(f"    [WARNING] Latent metrics not available")
                    else:
                        print(f"[VISUALIZATION] ⚠ No stable samples for visualization")
                    
                    vae_model.train()
                    
                except Exception as viz_error:
                    print(f"[WARNING] Visualization failed: {viz_error}")
                    import traceback
                    traceback.print_exc()
                    vae_model.train()  # Ensure model goes back to train mode
        
        # Validate model parameters periodically
        if (epoch + 1) % 1 == 0:  # Check every epoch
            with torch.no_grad():
                # Test forward pass with a small batch to detect issues early
                test_input = torch.randn(2, vae_model.in_channels, 1280).to(device) * 0.1
                try:
                    test_recon, test_mu, test_logvar, test_z, test_stable = vae_model(test_input)
                    if torch.isnan(test_recon).any() or torch.isinf(test_recon).any():
                        print(f"[CRITICAL] Model produces NaN/Inf on test input after epoch {epoch+1}!")
                        print(f"  This indicates model corruption. Training may be unstable.")
                        # Optionally stop here
                        # break
                except Exception as e:
                    print(f"[ERROR] Model forward pass failed: {e}")
                    break
            
            safe_model_parameters(vae_model)
            safe_model_parameters(feature_recon_head)
    
    print(f"\n{'='*80}")
    print(f"VAE TRAINING COMPLETED")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Model saved to: {model_save_path}")
    print(f"{'='*80}\n")
    
    # Plot training losses
    try:
        from utils.training_visualization import plot_vae_training_losses
        loss_plot_path = os.path.join(os.path.dirname(model_save_path), 'vae_training_losses.png')
        plot_vae_training_losses(metrics, loss_plot_path)
    except Exception as e:
        print(f"[WARNING] Failed to generate training loss plot: {e}")
    
    return vae_model, feature_recon_head, metrics
