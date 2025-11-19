"""Utility package for GenEEG."""

from .common import cleanup_gpu_memory, safe_tensor_operation, unscale_data
from .feature_extraction import (
    eeg_feature_vector,
    expand_feature_vector,
    compute_band_powers,
    compute_hjorth_parameters,
    compute_spectral_entropy,
    _welch_psd,
)
from .helpers import (
    WarmupCosineAnnealingLR,
    Lookahead,
    BatchAugmentation,
    get_cosine_schedule,
    robust_latent_scaling_from_mu,
    get_vae_encoder_features,
    nt_xent_loss,
)
from .generation import (
    generate_synthetic_eeg_batch,
    generate_balanced_synthetic_dataset,
    extract_features_from_real_samples,
    ddim_sampling,
)
from .training_visualization import (
    plot_vae_training_losses,
    plot_ldm_training_losses,
    plot_combined_training_summary,
)

__all__ = [
    # Common utilities
    'cleanup_gpu_memory',
    'safe_tensor_operation',
    'unscale_data',
    # Feature extraction
    'eeg_feature_vector',
    'expand_feature_vector',
    'compute_band_powers',
    'compute_hjorth_parameters',
    'compute_spectral_entropy',
    '_welch_psd',
    # Helpers
    'WarmupCosineAnnealingLR',
    'Lookahead',
    'BatchAugmentation',
    'get_cosine_schedule',
    'robust_latent_scaling_from_mu',
    'get_vae_encoder_features',
    'nt_xent_loss',
    # Generation
    'generate_synthetic_eeg_batch',
    'generate_balanced_synthetic_dataset',
    'extract_features_from_real_samples',
    'ddim_sampling',
    # Training Visualization
    'plot_vae_training_losses',
    'plot_ldm_training_losses',
    'plot_combined_training_summary',
]
