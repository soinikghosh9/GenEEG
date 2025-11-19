"""
Training Package

This package contains all training-related functionality:
- vae_losses: 11 specialized loss components for VAE training
- vae_trainer: VAE training loop with stability enhancements
- ldm_trainer: LDM training with DDPM/DDIM sampling
- ewc_utils: EWC utilities for continual learning
- classifier_trainer: Classifier training with class balancing
- continual_learning: CL-LOPO validation with EWC and ER
"""

from .vae_losses import (
    MultiScaleSTFTLoss,
    DWTLoss,
    spike_sharpness_loss,
    LowFrequencyPreservationLoss,
    HighFrequencyPreservationLoss,
    connectivity_loss_fn,
    MinorityClassFocusLoss,
    EdgeArtifactSuppressionLoss,
    nt_xent_loss,
    kl_divergence_loss,
    compute_seizure_fidelity_loss,
    VAELossComputer
)

from .vae_trainer import train_vae
from .ldm_trainer import train_latent_diffusion_unet
from .ewc_utils import (
    compute_fisher_information,
    ewc_loss_fn,
    save_fisher_params,
    load_fisher_params,
    accumulate_fisher
)
from .classifier_trainer import (
    train_pytorch_classifier,
    evaluate_pytorch_classifier,
    get_class_weights
)

__all__ = [
    # VAE Losses
    'MultiScaleSTFTLoss',
    'DWTLoss',
    'spike_sharpness_loss',
    'LowFrequencyPreservationLoss',
    'HighFrequencyPreservationLoss',
    'connectivity_loss_fn',
    'MinorityClassFocusLoss',
    'EdgeArtifactSuppressionLoss',
    'nt_xent_loss',
    'kl_divergence_loss',
    'compute_seizure_fidelity_loss',
    'VAELossComputer',
    # Trainers
    'train_vae',
    'train_latent_diffusion_unet',
    'train_pytorch_classifier',
    'evaluate_pytorch_classifier',
    # EWC utilities
    'compute_fisher_information',
    'ewc_loss_fn',
    'save_fisher_params',
    'load_fisher_params',
    'accumulate_fisher',
    # Classifier utilities
    'get_class_weights',
]
