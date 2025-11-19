"""
Models Package

This package contains all neural network architectures used in the GenEEG project:
- building_blocks: Reusable components (attention, ResNet blocks, etc.)
- vae: Variational Autoencoder for EEG signal compression
- ldm: Latent Diffusion Model for conditional generation
- classifier: CNN-BiLSTM classifier for seizure detection
- eegnet: EEGNet baseline classifier
"""

from .building_blocks import (
    TemporalSelfAttention1D,
    CrossAttention,
    AdaGN,
    ResNetBlock,
    SpatialChannelAttention,
    ChannelAttention,
    EnhancedTemporalAttention,
    Downsample1D,
    Upsample1D,
    FeatureReconstructionHead,
    sinusoidal_time_embedding
)

from .vae import DecoupledVAE, get_cosine_schedule
from .classifier import CNNBiLSTM
from .ldm import LatentDiffusionUNetEEG, DDPMScheduler
from .eegnet import EEGNet

# Backwards compatibility alias
LatentDiffusionModel = LatentDiffusionUNetEEG

__all__ = [
    # Building blocks
    'TemporalSelfAttention1D',
    'CrossAttention',
    'AdaGN',
    'ResNetBlock',
    'SpatialChannelAttention',
    'ChannelAttention',
    'EnhancedTemporalAttention',
    'Downsample1D',
    'Upsample1D',
    'FeatureReconstructionHead',
    'sinusoidal_time_embedding',
    # VAE
    'DecoupledVAE',
    'get_cosine_schedule',
    # Classifier
    'CNNBiLSTM',
    # LDM
    'LatentDiffusionUNetEEG',
    'DDPMScheduler',
    # EEGNet
    'EEGNet',
]
