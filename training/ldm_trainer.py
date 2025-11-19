"""
Latent Diffusion Model (LDM) Training Module

This module provides the training loop for the Latent Diffusion UNet model with
advanced features for high-quality EEG generation.

Key Features:
- V-prediction or epsilon-prediction loss targets
- Min-SNR loss weighting for improved sample quality
- P2 weighting for better high-frequency preservation
- Moment matching loss (μ/σ stability)
- Auxiliary losses (low-frequency preservation, seizure fidelity)
- Exponential Moving Average (EMA) of model weights
- Memory-safe EMA implementation
- Cosine learning rate schedule with warmup
- Log-normal timestep sampling
- Mixed precision training (BF16/FP16)

Usage:
    from training.ldm_trainer import train_latent_diffusion_unet
    from models.ldm import LatentDiffusionUNetEEG
    from models.vae import DecoupledVAE
    
    # Create models
    vae = DecoupledVAE(...)
    ldm = LatentDiffusionUNetEEG(...)
    
    # Train
    trained_ldm, history = train_latent_diffusion_unet(
        unet_model=ldm,
        vae_model_frozen=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        lr=2e-4,
        device='cuda',
        save_path='checkpoints/ldm.pth'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import math
import copy
import inspect
import os
from typing import Optional, Dict, List, Tuple

# Import loss functions
from training.vae_losses import (
    LowFrequencyPreservationLoss,
    compute_seizure_fidelity_loss
)

# Import utilities
from utils.helpers import get_cosine_schedule

# Import configuration constants
from configs.training_config import (
    LDM_DIFFUSION_TIMESTEPS,
    LDM_LOSS_TARGET,
    LDM_MIN_SNR_GAMMA,
    LDM_P2_K,
    LDM_P2_GAMMA,
    LDM_MOMENT_LOSS_WEIGHT,
    LDM_LOW_FREQ_PRESERVATION_WEIGHT,
    LDM_SEIZURE_LOSS_WEIGHT,
    LDM_AUX_EVERY,
    LDM_AUX_BATCH_FRAC,
    LDM_AUX_HUBER_DELTA,
    LDM_AUX_CAP,
    LDM_AUX_WARMUP_STEPS,
    LDM_GRAD_CLIP,
    LDM_SCALING_FACTOR,
    LDM_EMA_DECAY,
    LDM_LR_WARMUP_STEPS,
    LDM_USE_LOGNORMAL_T
)

from configs.dataset_config import TARGET_SFREQ


def train_latent_diffusion_unet(
    unet_model: nn.Module,
    vae_model_frozen: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    *,
    epochs: int = 150,
    lr: float = 2e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    diffusion_timesteps: Optional[int] = None,
    device: Optional[torch.device] = None,
    log_every: int = 100,
    save_path: Optional[str] = None,
    # Targets & weighting
    loss_target: Optional[str] = None,
    min_snr_gamma: Optional[float] = None,
    p2_k: Optional[float] = None,
    p2_gamma: Optional[float] = None,
    moment_loss_weight: Optional[float] = None,
    # Auxiliary losses
    low_freq_preservation_weight: Optional[float] = None,
    seizure_loss_weight: Optional[float] = None,
    aux_every: Optional[int] = None,
    aux_batch_frac: Optional[float] = None,
    aux_huber_delta: Optional[float] = None,
    aux_cap: Optional[float] = None,
    aux_warmup_steps: Optional[int] = None,
    # Hygiene
    grad_clip: Optional[float] = None,
    target_sfreq: Optional[float] = None,
    ldm_scaling_factor: Optional[float] = None,
    ema_decay: Optional[float] = None,
    # Schedule
    lr_warmup_steps: Optional[int] = None,
    # Timestep sampling
    use_lognormal_t: Optional[bool] = None,
    # EWC for Continual Learning
    ewc_params: Optional[Dict] = None,
    lambda_ewc: float = 5000.0,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the Latent Diffusion UNet model for EEG generation.
    
    This function implements a complete training loop with:
    - V-prediction or epsilon-prediction
    - Min-SNR and P2 loss weighting
    - Moment matching for distribution stability
    - Auxiliary losses (low-frequency preservation, seizure fidelity)
    - Exponential Moving Average (EMA)
    - Mixed precision training
    
    Args:
        unet_model: The Latent Diffusion UNet model to train
        vae_model_frozen: Frozen VAE model for encoding/decoding
        train_loader: DataLoader with CachedLatentDataset format (z0, x_raw, features, labels)
        val_loader: Optional validation DataLoader
        epochs: Number of training epochs
        lr: Base learning rate
        min_lr: Minimum learning rate for cosine annealing
        weight_decay: Weight decay for AdamW
        diffusion_timesteps: Number of diffusion timesteps (default: LDM_DIFFUSION_TIMESTEPS)
        device: Device to train on (default: auto-detect)
        log_every: Log progress every N steps
        save_path: Path to save best checkpoint
        
        # Loss configuration
        loss_target: 'v' for v-prediction or 'eps' for epsilon (default: LDM_LOSS_TARGET)
        min_snr_gamma: Min-SNR gamma value (default: LDM_MIN_SNR_GAMMA)
        p2_k: P2 weighting k parameter (default: LDM_P2_K)
        p2_gamma: P2 weighting gamma parameter (default: LDM_P2_GAMMA)
        moment_loss_weight: Weight for moment matching loss (default: LDM_MOMENT_LOSS_WEIGHT)
        
        # Auxiliary losses
        low_freq_preservation_weight: Weight for low-freq loss (default: LDM_LOW_FREQ_PRESERVATION_WEIGHT)
        seizure_loss_weight: Weight for seizure fidelity loss (default: LDM_SEIZURE_LOSS_WEIGHT)
        aux_every: Apply auxiliary losses every N steps (default: LDM_AUX_EVERY)
        aux_batch_frac: Fraction of batch for auxiliary losses (default: LDM_AUX_BATCH_FRAC)
        aux_huber_delta: Huber loss delta (default: LDM_AUX_HUBER_DELTA)
        aux_cap: Cap for auxiliary losses (default: LDM_AUX_CAP)
        aux_warmup_steps: Warmup steps for auxiliary losses (default: LDM_AUX_WARMUP_STEPS)
        
        # Training hygiene
        grad_clip: Gradient clipping value (default: LDM_GRAD_CLIP)
        target_sfreq: Target sampling frequency (default: TARGET_SFREQ)
        ldm_scaling_factor: Latent scaling factor (default: LDM_SCALING_FACTOR)
        ema_decay: EMA decay rate (default: LDM_EMA_DECAY)
        lr_warmup_steps: Learning rate warmup steps (default: LDM_LR_WARMUP_STEPS)
        use_lognormal_t: Use log-normal timestep sampling (default: LDM_USE_LOGNORMAL_T)
        
    Returns:
        Tuple of (trained_model, history_dict):
        - trained_model: EMA model if EMA is enabled, otherwise the trained model
        - history_dict: Dictionary with training metrics per epoch
    
    Example:
        >>> vae = DecoupledVAE(...)
        >>> ldm = LatentDiffusionUNetEEG(...)
        >>> trained_ldm, hist = train_latent_diffusion_unet(
        ...     unet_model=ldm,
        ...     vae_model_frozen=vae,
        ...     train_loader=train_loader,
        ...     epochs=150,
        ...     lr=2e-4,
        ...     device='cuda'
        ... )
    """
    # Set defaults from config
    diffusion_timesteps = diffusion_timesteps or LDM_DIFFUSION_TIMESTEPS
    loss_target = loss_target or LDM_LOSS_TARGET
    min_snr_gamma = min_snr_gamma if min_snr_gamma is not None else LDM_MIN_SNR_GAMMA
    p2_k = p2_k if p2_k is not None else LDM_P2_K
    p2_gamma = p2_gamma if p2_gamma is not None else LDM_P2_GAMMA
    moment_loss_weight = moment_loss_weight if moment_loss_weight is not None else LDM_MOMENT_LOSS_WEIGHT
    low_freq_preservation_weight = low_freq_preservation_weight if low_freq_preservation_weight is not None else LDM_LOW_FREQ_PRESERVATION_WEIGHT
    seizure_loss_weight = seizure_loss_weight if seizure_loss_weight is not None else LDM_SEIZURE_LOSS_WEIGHT
    aux_every = aux_every or LDM_AUX_EVERY
    aux_batch_frac = aux_batch_frac if aux_batch_frac is not None else LDM_AUX_BATCH_FRAC
    aux_huber_delta = aux_huber_delta if aux_huber_delta is not None else LDM_AUX_HUBER_DELTA
    aux_cap = aux_cap if aux_cap is not None else LDM_AUX_CAP
    aux_warmup_steps = aux_warmup_steps or LDM_AUX_WARMUP_STEPS
    grad_clip = grad_clip if grad_clip is not None else LDM_GRAD_CLIP
    target_sfreq = target_sfreq or TARGET_SFREQ
    ldm_scaling_factor = ldm_scaling_factor if ldm_scaling_factor is not None else LDM_SCALING_FACTOR
    ema_decay = ema_decay if ema_decay is not None else LDM_EMA_DECAY
    lr_warmup_steps = lr_warmup_steps or LDM_LR_WARMUP_STEPS
    use_lognormal_t = use_lognormal_t if use_lognormal_t is not None else LDM_USE_LOGNORMAL_T
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = True
    
    print(f"\n[INFO] LDM Training Configuration:")
    print(f"  Epochs: {epochs}, LR: {lr} -> {min_lr}, Device: {device}")
    print(f"  Diffusion timesteps: {diffusion_timesteps}")
    print(f"  Loss target: {loss_target}")
    print(f"  Min-SNR gamma: {min_snr_gamma}, P2 (k={p2_k}, γ={p2_gamma})")
    print(f"  Moment loss weight: {moment_loss_weight}")
    print(f"  Low-freq preservation weight: {low_freq_preservation_weight}")
    print(f"  Seizure fidelity weight: {seizure_loss_weight}")
    print(f"  EMA decay: {ema_decay}")
    print(f"  Gradient clip: {grad_clip}")
    if ewc_params is not None:
        print(f"  EWC lambda: {lambda_ewc} (Continual Learning ENABLED)")
    else:
        print(f"  EWC: Disabled (First patient training)")
    print(f"  ╔════════════════════════════════════════════════════════════════╗")
    print(f"  ║  [CRITICAL FIX] CFG DROPOUT: 10%                              ║")
    print(f"  ║  (Enables classifier-free guidance - MANDATORY for CFG!)      ║")
    print(f"  ╚════════════════════════════════════════════════════════════════╝")
    
    # Setup models
    unet_model.to(device).train()
    for p in unet_model.parameters():
        p.requires_grad_(True)
    
    vae_model_frozen.to(device).eval()
    for p in vae_model_frozen.parameters():
        p.requires_grad_(False)
    
    # Optimizer
    try:
        optimizer = torch.optim.AdamW(
            unet_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=(device.type == "cuda")
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            unet_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
    
    # Learning rate schedule
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    
    def lr_at(step):
        """Cosine annealing with warmup."""
        if lr_warmup_steps > 0 and step < lr_warmup_steps:
            return lr * (step / float(lr_warmup_steps))
        t = (step - lr_warmup_steps) / max(1, total_steps - lr_warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
    
    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))
    
    # Diffusion schedule (cosine)
    betas = get_cosine_schedule(diffusion_timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0).clamp_(1e-5, 1.0)
    sqrt_ab, sqrt_omab = alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()
    
    # Loss weighting schedules
    snr = alpha_bar / (1.0 - alpha_bar)
    min_snr_w = (torch.minimum(snr, torch.tensor(min_snr_gamma, device=device)) / (snr + 1e-8)) if (min_snr_gamma > 0) else torch.ones_like(snr)
    p2_w = torch.pow(p2_k + snr, -p2_gamma) if (p2_gamma > 0) else torch.ones_like(snr)
    w_t_all = (min_snr_w * p2_w).detach()
    
    # Memory-safe EMA setup
    use_ema = (ema_decay and ema_decay > 0)
    ema_model = None
    
    if use_ema:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            ema_model = copy.deepcopy(unet_model).to(device).eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)
            print(f"[INFO] EMA enabled with decay {ema_decay}")
        except RuntimeError as e:
            print(f"[WARNING] Not enough memory for EMA model. Training without EMA. Error: {e}")
            use_ema = False
            ema_model = None
    
    @torch.no_grad()
    def ema_update(dst, src, decay):
        """Update EMA model with memory safety."""
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            for p_dst, p_src in zip(dst.parameters(), src.parameters()):
                p_dst.data.mul_(decay).add_(p_src.data, alpha=1 - decay)
            return True
        except RuntimeError as e:
            print(f"[WARNING] EMA update failed: {e}")
            return False
    
    # Helper functions
    use_bf16 = False
    if device.type == "cuda":
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except RuntimeError:
            use_bf16 = False
    
    ac_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    def view_like_t(x, B):
        """Reshape tensor for broadcasting with batch."""
        return (B,) + (1,) * (x.ndim - 1)
    
    def cosine_warmup_weight(step, warmup_steps):
        """Cosine warmup for auxiliary losses."""
        if warmup_steps <= 0 or step >= warmup_steps:
            return 1.0
        return 0.5 * (1 - math.cos(math.pi * step / warmup_steps))
    
    def sample_timesteps(B, T, device, mean=None, sigma=0.5):
        """Sample timesteps with optional log-normal distribution."""
        if not use_lognormal_t:
            return torch.randint(0, T, (B,), device=device, dtype=torch.long)
        mean = math.log(0.6 * T) if mean is None else mean
        u = torch.randn(B, device=device) * sigma + mean
        return torch.clamp(u.exp().long(), 1, T - 1)
    
    # Initialize auxiliary loss function
    lf_loss_fn = LowFrequencyPreservationLoss(sfreq=target_sfreq, device=device) if low_freq_preservation_weight > 0 else None
    
    # Logging
    hist = {"total": [], "denoise": [], "moment": [], "aux": [], "ewc": [], "val_denoise": [], "lr": []}
    ema_log = {"total": None, "denoise": None, "moment": None, "aux": None, "ewc": None}
    
    def upd(name, v, a=0.98):
        """Update exponential moving average for logging."""
        ema_log[name] = float(v) if ema_log[name] is None else a * ema_log[name] + (1 - a) * float(v)
    
    best_val = float("inf")
    global_step = 0
    patience_counter = 0
    patience_threshold = 15  # Stop if no improvement for 15 validation checks (75 epochs)
    
    # Training loop
    for ep in range(1, epochs + 1):
        run = {"total": 0.0, "denoise": 0.0, "moment": 0.0, "aux": 0.0, "ewc": 0.0}
        nb = 0
        
        for batch in train_loader:
            global_step += 1
            lr_step = lr_at(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr_step
            
            # Extract batch data (CachedLatentDataset format)
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise RuntimeError("train_loader must return (z0, x_raw, features_cond, labels_batch)")
            
            z0, x_raw, features_cond, labels_batch = batch
            
            z0 = z0.to(device)
            x_raw = x_raw.to(device)
            features_cond = features_cond.to(device)
            labels_batch = labels_batch.to(device)
            
            B = z0.shape[0]
            t = sample_timesteps(B, diffusion_timesteps, device)
            eps = torch.randn_like(z0)
            vshape = view_like_t(z0, B)
            sqrt_ab_t = sqrt_ab.index_select(0, t).view(*vshape)
            sqrt_omab_t = sqrt_omab.index_select(0, t).view(*vshape)
            zt = sqrt_ab_t * z0 + sqrt_omab_t * eps
            
            # CRITICAL FIX: Classifier-Free Guidance Dropout
            # Randomly drop conditioning 10% of the time to enable CFG during inference
            # This teaches the model to generate unconditionally, which is essential for CFG
            cfg_dropout_prob = 0.1  # Drop conditioning 10% of the time
            uncond_mask = torch.rand(B, device=device) < cfg_dropout_prob
            
            # DIAGNOSTIC: Log CFG dropout activity in first batch
            if global_step == 1:
                num_uncond = uncond_mask.sum().item()
                print(f"  [CFG DROPOUT ACTIVE] Batch 1: {num_uncond}/{B} samples set to unconditional ({num_uncond/B*100:.1f}%)")
            
            # Apply dropout: set labels to num_classes (reserved for unconditional)
            labels_for_training = labels_batch.clone()
            labels_for_training[uncond_mask] = unet_model.num_classes
            
            # Apply dropout: set features to zero for unconditional samples
            features_for_training = features_cond.clone()
            features_for_training[uncond_mask] = 0.0
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type, dtype=ac_dtype):
                # Forward pass with CFG-aware conditioning
                pred = unet_model(zt, t, class_labels=labels_for_training, feature_cond=features_for_training)
                
                # Compute target based on loss type
                if loss_target.lower() == "v":
                    v_tgt = sqrt_ab_t * eps - sqrt_omab_t * z0
                    base = F.mse_loss(pred, v_tgt, reduction="none")
                    pred_for_moment, tgt_for_moment = pred, v_tgt
                else:
                    base = F.mse_loss(pred, eps, reduction="none")
                    pred_for_moment, tgt_for_moment = pred, eps
                
                # Reduce spatial dimensions
                while base.ndim > 1:
                    base = base.mean(dim=-1)
                
                # Apply timestep-dependent weighting
                wt = w_t_all.index_select(0, t)
                loss_denoise = (base * wt).mean()
                
                # Moment stability loss (μ/σ matching)
                mu_pred = pred_for_moment.flatten(1).mean(1)
                mu_tgt = tgt_for_moment.flatten(1).mean(1)
                sig_pred = pred_for_moment.flatten(1).std(1)
                sig_tgt = tgt_for_moment.flatten(1).std(1)
                loss_moment = F.mse_loss(mu_pred, mu_tgt) + F.mse_loss(sig_pred, sig_tgt)
                loss_moment *= moment_loss_weight
                
                total_loss = loss_denoise + loss_moment
                
                # Auxiliary losses (every aux_every steps on subset of batch)
                loss_aux = torch.zeros((), device=device, dtype=total_loss.dtype)
                if global_step % aux_every == 0:
                    aux_warmup_steps_adjusted = min(aux_warmup_steps, total_steps // 4)
                    warmup_w = cosine_warmup_weight(global_step, aux_warmup_steps_adjusted)
                    
                    if warmup_w > 1e-6:
                        sub_b = max(1, int(B * aux_batch_frac))
                        
                        # Decode z0 subset for auxiliary losses
                        with torch.no_grad():
                            z0_sub = z0[:sub_b] * ldm_scaling_factor
                            
                            # Check if VAE has skip connections
                            if hasattr(vae_model_frozen, 'decode') and 'skip' in str(inspect.signature(vae_model_frozen.decode)):
                                recon_sub = vae_model_frozen.decode(z0_sub, feature_cond=features_cond[:sub_b], skip_features=None)
                            else:
                                recon_sub = vae_model_frozen.decode(z0_sub, feature_cond=features_cond[:sub_b])
                            
                            x_sub = x_raw[:sub_b]
                        
                        # Low-frequency preservation loss
                        if lf_loss_fn is not None and low_freq_preservation_weight > 0:
                            lf_loss = lf_loss_fn(recon_sub.float(), x_sub.float())
                            loss_aux_lf = torch.clamp(
                                F.huber_loss(lf_loss, torch.zeros_like(lf_loss), delta=aux_huber_delta),
                                0, aux_cap
                            )
                            loss_aux_lf *= low_freq_preservation_weight * warmup_w
                            loss_aux = loss_aux + loss_aux_lf
                        
                        # Seizure fidelity loss
                        if seizure_loss_weight > 0:
                            try:
                                loss_seizure = compute_seizure_fidelity_loss(x_sub.float(), recon_sub.float(), sfreq=target_sfreq)
                                loss_seizure = torch.clamp(loss_seizure, 0, aux_cap)
                                loss_seizure *= seizure_loss_weight * warmup_w
                                
                                if torch.isfinite(loss_seizure).all():
                                    loss_aux = loss_aux + loss_seizure
                                else:
                                    print(f"[WARN] Non-finite seizure loss at step {global_step}")
                            except Exception as e:
                                print(f"[WARN] Seizure loss computation failed: {e}")
                        
                        # Safety check for auxiliary loss
                        if not torch.isfinite(loss_aux).all():
                            print(f"[WARN] Non-finite aux loss at step {global_step}, setting to 0")
                            loss_aux = torch.zeros_like(loss_aux)
                
                total_loss = total_loss + loss_aux
                
                # EWC regularization for Continual Learning
                loss_ewc = torch.zeros((), device=device, dtype=total_loss.dtype)
                if ewc_params is not None:
                    # Compute EWC loss to prevent catastrophic forgetting
                    for name, param in unet_model.named_parameters():
                        if name in ewc_params['fisher_info'] and name in ewc_params['old_params']:
                            # Move EWC tensors to device (they're stored on CPU to save memory)
                            fisher = ewc_params['fisher_info'][name].to(device)
                            old_param = ewc_params['old_params'][name].to(device)
                            # EWC penalty: λ/2 * F_i * (θ_i - θ*_i)^2
                            loss_ewc += (fisher * (param - old_param).pow(2)).sum()
                    
                    loss_ewc *= (lambda_ewc / 2.0)
                    total_loss = total_loss + loss_ewc
            
            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            # EMA update
            if use_ema and ema_model is not None:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                try:
                    continue_ema = ema_update(ema_model, unet_model, ema_decay)
                    if not continue_ema:
                        use_ema = False
                        try:
                            del ema_model
                            torch.cuda.empty_cache()
                            ema_model = None
                        except:
                            pass
                except RuntimeError as e:
                    print(f"[ERROR] EMA update failed, disabling EMA: {e}")
                    use_ema = False
                    try:
                        del ema_model
                        torch.cuda.empty_cache()
                        ema_model = None
                    except:
                        pass
            
            # Logging
            upd("total", total_loss)
            upd("denoise", loss_denoise)
            upd("moment", loss_moment)
            upd("aux", loss_aux)
            upd("ewc", loss_ewc)
            run["total"] += float(total_loss)
            run["denoise"] += float(loss_denoise)
            run["moment"] += float(loss_moment)
            run["aux"] += float(loss_aux)
            run["ewc"] += float(loss_ewc)
            nb += 1
            
            if global_step % log_every == 0:
                ewc_str = f" | EWC={ema_log['ewc']:.6f}" if ewc_params is not None else ""
                print(f"[Ep {ep}/{epochs} | Step {global_step}] LR={lr_step:.2e} | "
                      f"Total={ema_log['total']:.4f} | Denoise={ema_log['denoise']:.4f} | "
                      f"Moment={ema_log['moment']:.6f} | Aux={ema_log['aux']:.6f}{ewc_str}")
        
        # End-of-epoch logging with more detail
        for k in run:
            run[k] /= max(1, nb)
        hist["lr"].append(lr_step)
        for k in ["total", "denoise", "moment", "aux", "ewc"]:
            hist[k].append(run[k])
        
        print(f"\n{'='*80}")
        print(f"EPOCH {ep}/{epochs} SUMMARY:")
        print(f"  Total Loss:     {run['total']:.4f}")
        print(f"  Denoise Loss:   {run['denoise']:.4f} (main diffusion objective)")
        print(f"  Moment Loss:    {run['moment']:.6f} (μ/σ stability)")
        print(f"  Auxiliary Loss: {run['aux']:.6f} (low-freq + seizure)")
        if ewc_params is not None and run['ewc'] > 0:
            print(f"  EWC Loss:       {run['ewc']:.6f} (continual learning)")
        print(f"  Learning Rate:  {lr_step:.6e}")
        print(f"  Batches:        {nb}")
        print(f"{'='*80}\n")
        
        # Validation
        if val_loader is not None and ep % 5 == 0:
            unet_model.eval()
            tgt = ema_model if (use_ema and ema_model is not None) else unet_model
            val_total, vb = 0.0, 0
            
            with torch.no_grad(), autocast(device_type=device.type, dtype=ac_dtype):
                for batch in val_loader:
                    if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                        raise RuntimeError("val_loader must return (x, _, features_cond, labels_batch)")
                    xv, _, fv, yv = batch
                    xv, fv, yv = xv.to(device), fv.to(device), yv.to(device)
                    
                    # Encode validation data
                    mu, _ = vae_model_frozen.encode(xv)
                    zv = mu / float(ldm_scaling_factor)
                    
                    tv = sample_timesteps(zv.shape[0], diffusion_timesteps, device)
                    epsv = torch.randn_like(zv)
                    vshape = view_like_t(zv, zv.shape[0])
                    sqrt_ab_v = sqrt_ab.index_select(0, tv).view(*vshape)
                    sqrt_omab_v = sqrt_omab.index_select(0, tv).view(*vshape)
                    ztv = sqrt_ab_v * zv + sqrt_omab_v * epsv
                    
                    pred = tgt(ztv, tv, class_labels=yv, feature_cond=fv)
                    v_tgt = sqrt_ab_v * epsv - sqrt_omab_v * zv if loss_target.lower() == "v" else epsv
                    l = F.mse_loss(pred, v_tgt, reduction="none")
                    while l.ndim > 1:
                        l = l.mean(dim=-1)
                    wt = w_t_all.index_select(0, tv)
                    val_total += float((l * wt).mean())
                    vb += 1
            
            if vb > 0:
                val_avg = val_total / vb
                improvement = ""
                if len(hist["val_denoise"]) > 0:
                    prev_val = hist["val_denoise"][-1]
                    delta = ((val_avg - prev_val) / prev_val) * 100
                    improvement = f" ({delta:+.2f}%)"
                
                print(f"  [VALIDATION] Loss: {val_avg:.4f}{improvement} | Model: {'EMA' if use_ema and ema_model is not None else 'Main'}")
                hist["val_denoise"].append(val_avg)
                
                if save_path and val_avg < best_val:
                    best_val = val_avg
                    patience_counter = 0  # Reset patience on improvement
                    model_to_save = ema_model if (use_ema and ema_model is not None) else unet_model
                    try:
                        torch.save({
                            "unet": model_to_save.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "epoch": ep,
                            "val_denoise": val_avg,
                            "ema": (use_ema and ema_model is not None)
                        }, save_path)
                        print(f"  ✓ Best model saved (val_loss: {val_avg:.4f}) -> {save_path}")
                    except Exception as e:
                        print(f"  [WARN] Failed to save checkpoint: {e}")
                else:
                    patience_counter += 1
                    print(f"  [PATIENCE] No improvement for {patience_counter}/{patience_threshold} validation checks")
                    
                    if patience_counter >= patience_threshold:
                        print(f"\n[EARLY STOP] No improvement for {patience_threshold} checks. Best val: {best_val:.4f}")
                        break
            
            unet_model.train()
    
    print("[SUCCESS] LDM training complete.")
    
    # Plot training losses
    try:
        from utils.training_visualization import plot_ldm_training_losses
        loss_plot_path = os.path.join(os.path.dirname(save_path), 'ldm_training_losses.png') if save_path else 'ldm_training_losses.png'
        plot_ldm_training_losses(hist, loss_plot_path)
    except Exception as e:
        print(f"[WARNING] Failed to generate training loss plot: {e}")
    
    final_model = ema_model if (use_ema and ema_model is not None) else unet_model
    return final_model, hist
