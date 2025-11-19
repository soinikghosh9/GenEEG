"""Data loading package for GenEEG.

Provides unified interface for loading Siena and CHB-MIT datasets,
as well as preprocessing utilities and dataset classes.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.chbmit_loader import load_chbmit_dataset, CHBMITLoader
from data.siena_loader import load_siena_dataset, SienaLoader
from data.preprocessing import (
    GenericDataset,
    EegPreprocessedDataset,
    CachedLatentDataset,
    balance_dataset_by_undersampling,
    balance_dataset_with_smote,
    oversample_data,
    calculate_eeg_features,
    create_optimized_dataloader,
)
from configs.dataset_config import DatasetConfig


__all__ = [
    # Dataset loaders
    'get_dataset',
    'load_siena_dataset',
    'load_chbmit_dataset',
    'SienaLoader',
    'CHBMITLoader',
    # Dataset classes
    'GenericDataset',
    'EegPreprocessedDataset',
    'CachedLatentDataset',
    # Balancing functions
    'balance_dataset_by_undersampling',
    'balance_dataset_with_smote',
    'oversample_data',
    # Utilities
    'calculate_eeg_features',
    'create_optimized_dataloader',
]


def get_dataset(
    dataset_name: str = None,
    **kwargs
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load dataset based on configuration.
    
    Args:
        dataset_name: 'siena' or 'chbmit' (uses config default if None)
        **kwargs: Additional arguments passed to dataset loaders
    
    Returns:
        Dictionary mapping patient_id to (segments, labels):
        {
            'patient_01': (segments_array, labels_array),
            'patient_02': (segments_array, labels_array),
            ...
        }
        
        segments_array shape: (n_segments, n_channels, n_samples)
        labels_array shape: (n_segments,) - 0=Normal, 1=Preictal, 2=Ictal
    
    Example:
        >>> from data import get_dataset
        >>> patient_data = get_dataset('chbmit')
        >>> segments, labels = patient_data['chb01']
        >>> print(f"Shape: {segments.shape}")
        Shape: (1500, 18, 1280)  # 1500 segments, 18 channels, 1280 samples (5s @ 256Hz)
    """
    if dataset_name is None:
        dataset_name = DatasetConfig.DATASET_NAME
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'siena':
        # Set default parameters from config for consistent 16 channels
        default_params = {
            'root_dir': DatasetConfig.SIENA_ROOT_DIR,
            'target_sampling_rate': DatasetConfig.SIENA_TARGET_SAMPLING_RATE,
            'num_channels': DatasetConfig.COMMON_EEG_CHANNELS,  # Use 16 channels consistently
            'segment_duration_sec': DatasetConfig.SEGMENT_DURATION_SEC,
            'overlap': DatasetConfig.SEGMENT_OVERLAP,
            'preictal_duration_sec': DatasetConfig.SIENA_PREICTAL_DURATION_SEC,
            'verbose': True
        }
        # Override with user-provided kwargs
        default_params.update(kwargs)
        return load_siena_dataset(**default_params)
    
    elif dataset_name == 'chbmit':
        # Set default parameters from config
        default_params = {
            'root_dir': DatasetConfig.CHBMIT_ROOT_DIR,
            'target_sampling_rate': DatasetConfig.CHBMIT_SAMPLING_RATE,
            'common_channels': DatasetConfig.CHBMIT_COMMON_CHANNELS if DatasetConfig.USE_COMMON_CHANNELS_ONLY else None,
            'segment_duration_sec': DatasetConfig.SEGMENT_DURATION_SEC,
            'overlap': DatasetConfig.SEGMENT_OVERLAP,
            'preictal_duration_sec': DatasetConfig.CHBMIT_PREICTAL_DURATION_SEC,
            'verbose': True
        }
        # Override with user-provided kwargs
        default_params.update(kwargs)
        
        return load_chbmit_dataset(**default_params)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'siena' or 'chbmit'")


def load_siena_dataset(**kwargs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load Siena dataset.
    
    Args:
        **kwargs: Dataset loading parameters
    
    Returns:
        Dictionary mapping patient_id to (segments, labels)
    """
    from .siena_loader import load_siena_dataset as _load_siena
    
    # Set default parameters from config
    default_params = {
        'root_dir': DatasetConfig.SIENA_ROOT_DIR,
        'original_sampling_rate': DatasetConfig.SIENA_SAMPLING_RATE,
        'target_sampling_rate': DatasetConfig.SIENA_TARGET_SAMPLING_RATE,
        'num_channels': DatasetConfig.get_num_channels(),
        'segment_duration_sec': DatasetConfig.SEGMENT_DURATION_SEC,
        'overlap': DatasetConfig.SEGMENT_OVERLAP,
        'preictal_duration_sec': DatasetConfig.SIENA_PREICTAL_DURATION_SEC,
        'verbose': True
    }
    # Override with user-provided kwargs
    default_params.update(kwargs)
    
    return _load_siena(**default_params)


__all__ = ['get_dataset', 'load_chbmit_dataset', 'load_siena_dataset']
