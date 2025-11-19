"""Dataset configuration for GenEEG."""

import os
from typing import Literal


class DatasetConfig:
    """Configuration for dataset selection and paths."""
    
    # ========================================
    # GENERAL CONFIGURATION
    # ========================================
    TARGET_SFREQ = 256.0  # Target sampling frequency for all datasets
    COMMON_EEG_CHANNELS = 16  # Number of channels used in models
    SEGMENT_SAMPLES = 1280  # Number of samples per segment (5 sec * 256 Hz)
    
    # ========================================
    # DATASET SELECTION
    # ========================================
    DATASET_NAME: Literal['siena', 'chbmit'] = 'siena'  # Switch between datasets
    
    # ========================================
    # SIENA DATASET CONFIGURATION
    # ========================================
    SIENA_ROOT_DIR = r"D:\DAtasets\siena-scalp-eeg-database-3.0.0"
    SIENA_SAMPLING_RATE = 512  # Original sampling rate
    SIENA_TARGET_SAMPLING_RATE = 256  # Downsampled rate
    SIENA_NUM_CHANNELS = 29
    SIENA_PREICTAL_DURATION_SEC = 20 * 60  # 20 minutes before seizure
    
    # ========================================
    # CHB-MIT DATASET CONFIGURATION
    # ========================================
    CHBMIT_ROOT_DIR = r"D:\DAtasets\CHB\chb1"
    CHBMIT_SAMPLING_RATE = 256  # Native sampling rate
    CHBMIT_NUM_CHANNELS = 23  # Maximum channels available
    CHBMIT_PREICTAL_DURATION_SEC = 20 * 60  # 20 minutes before seizure
    
    # Common channels across both datasets (16 channels following 10-20 system)
    # Selected channels that are consistently available in both Siena and CHB-MIT
    COMMON_CHANNEL_NAMES = [
        'FP1', 'FP2', 'F7', 'F3', 'F4', 'F8',
        'T3', 'C3', 'CZ', 'C4', 'T4',
        'T5', 'P3', 'P4', 'T6',
        'O2'  # 16 channels total (removed O1 to get exactly 16)
    ]
    
    # Backwards compatibility
    CHBMIT_COMMON_CHANNELS = COMMON_CHANNEL_NAMES
    
    # ========================================
    # PREPROCESSING CONFIGURATION
    # ========================================
    BANDPASS_LOW = 0.5  # Hz
    BANDPASS_HIGH = 70.0  # Hz
    NOTCH_FREQ = 50.0  # Hz (60 for US, 50 for Europe)
    SEGMENT_DURATION_SEC = 5.0  # 5-second windows
    SEGMENT_OVERLAP = 0.5  # 50% overlap
    
    # ========================================
    # PREICTAL WINDOW SENSITIVITY ANALYSIS
    # ========================================
    PREICTAL_WINDOWS_TO_TEST = [5, 10, 20, 30]  # Minutes
    
    # ========================================
    # CLASS DEFINITIONS
    # ========================================
    CLASS_NORMAL = 0
    CLASS_PREICTAL = 1
    CLASS_ICTAL = 2
    NUM_CLASSES = 3
    
    # ========================================
    # EXPERIMENTAL FLAGS
    # ========================================
    RUN_CHBMIT_VALIDATION = False  # Set True to run CHB-MIT experiments
    RUN_PREICTAL_SENSITIVITY = False  # Set True for window sensitivity analysis
    USE_COMMON_CHANNELS_ONLY = True  # Use 18 common channels for cross-dataset compatibility
    
    @classmethod
    def get_root_dir(cls) -> str:
        """Get root directory for currently selected dataset."""
        if cls.DATASET_NAME == 'siena':
            return cls.SIENA_ROOT_DIR
        elif cls.DATASET_NAME == 'chbmit':
            return cls.CHBMIT_ROOT_DIR
        else:
            raise ValueError(f"Unknown dataset: {cls.DATASET_NAME}")
    
    @classmethod
    def get_sampling_rate(cls) -> int:
        """Get sampling rate for currently selected dataset."""
        if cls.DATASET_NAME == 'siena':
            return cls.SIENA_TARGET_SAMPLING_RATE
        elif cls.DATASET_NAME == 'chbmit':
            return cls.CHBMIT_SAMPLING_RATE
        else:
            raise ValueError(f"Unknown dataset: {cls.DATASET_NAME}")
    
    @classmethod
    def get_num_channels(cls) -> int:
        """Get number of channels for currently selected dataset."""
        if cls.USE_COMMON_CHANNELS_ONLY:
            return len(cls.CHBMIT_COMMON_CHANNELS)
        if cls.DATASET_NAME == 'siena':
            return cls.SIENA_NUM_CHANNELS
        elif cls.DATASET_NAME == 'chbmit':
            return cls.CHBMIT_NUM_CHANNELS
        else:
            raise ValueError(f"Unknown dataset: {cls.DATASET_NAME}")
    
    @classmethod
    def get_preictal_duration(cls) -> int:
        """Get preictal duration in seconds for currently selected dataset."""
        if cls.DATASET_NAME == 'siena':
            return cls.SIENA_PREICTAL_DURATION_SEC
        elif cls.DATASET_NAME == 'chbmit':
            return cls.CHBMIT_PREICTAL_DURATION_SEC
        else:
            raise ValueError(f"Unknown dataset: {cls.DATASET_NAME}")
    
    @classmethod
    def validate_paths(cls) -> bool:
        """Validate that dataset paths exist."""
        if cls.DATASET_NAME == 'siena':
            return os.path.exists(cls.SIENA_ROOT_DIR)
        elif cls.DATASET_NAME == 'chbmit':
            return os.path.exists(cls.CHBMIT_ROOT_DIR)
        return False


# Export constant for easy access
TARGET_SFREQ = DatasetConfig.TARGET_SFREQ
