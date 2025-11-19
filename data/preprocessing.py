"""
Data Preprocessing Utilities for GenEEG

This module contains dataset classes and data balancing utilities used for
preparing EEG data for training VAE, LDM, and classifier models.

Classes:
    - GenericDataset: Simple wrapper for tensor data and labels
    - EegPreprocessedDataset: Dataset with normalization, augmentation, and feature computation
    - CachedLatentDataset: Pre-computed VAE latent codes for efficient LDM training
    
Functions:
    - balance_dataset_by_undersampling: Balance classes via undersampling majority
    - balance_dataset_with_smote: Balance classes using SMOTE
    - oversample_data: Oversample minority classes
    - calculate_eeg_features: Compute 12D neurophysiological feature vector
    - create_optimized_dataloader: Create optimized DataLoader with OS-specific settings
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from typing import Tuple, List, Optional
from tqdm import tqdm
import platform
import os

# Try to import SMOTE and RandomOverSampler
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not available. SMOTE and RandomOverSampler will not work.")

# Try to import scipy and antropy for feature computation
try:
    from scipy.signal import welch
    from scipy import stats
    import antropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy/antropy not available. Feature computation may be limited.")


# =============================================================================
# Constants (imported from configs)
# =============================================================================

# These should be imported from configs, but we define defaults here for standalone use
try:
    from configs.dataset_config import (
        DatasetConfig,
        SEED_VALUE,
        INITIAL_UNDERSAMPLE_RATIO,
    )
    from configs.model_config import ModelConfig
    
    COMMON_EEG_CHANNELS = DatasetConfig.COMMON_EEG_CHANNELS
    TARGET_SFREQ = DatasetConfig.TARGET_SFREQ
    SEGMENT_SAMPLES = DatasetConfig.SEGMENT_SAMPLES
    VAE_FEATURE_COND_DIM = ModelConfig.VAE_FEATURE_COND_DIM
    VAE_LATENT_CHANNELS = ModelConfig.VAE_LATENT_DIM
except ImportError:
    # Fallback defaults
    COMMON_EEG_CHANNELS = 16  # Changed from 18 to 16
    TARGET_SFREQ = 256
    SEGMENT_SAMPLES = 1280  # 5s @ 256Hz
    SEED_VALUE = 42
    INITIAL_UNDERSAMPLE_RATIO = 3.0
    VAE_FEATURE_COND_DIM = 12
    VAE_LATENT_CHANNELS = 128
    VAE_LATENT_CHANNELS = 64


# =============================================================================
# Simple Dataset Wrapper
# =============================================================================

class GenericDataset(Dataset):
    """
    Simple wrapper for tensor data and labels.
    
    Args:
        data: NumPy array or tensor of shape (N, ...) 
        labels: NumPy array or tensor of shape (N,)
    """
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None
) -> DataLoader:
    """
    Create optimized DataLoader with OS-specific settings.
    
    CRITICAL FIX: Windows multiprocessing with spawn causes EOFError.
    Solution: Use num_workers=0 on Windows for stability.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of workers (default: 0 for Windows, auto for Unix)
    
    Returns:
        DataLoader instance
    """
    is_windows = platform.system().lower().startswith("win")

    if num_workers is None:
        if is_windows:
            # CRITICAL: Windows spawn multiprocessing causes EOFError
            # Use single-process loading (num_workers=0) for stability
            num_workers = 0
        else:
            # Unix/Linux: Safe to use multiprocessing
            try:
                cpu_count = os.cpu_count() or 2
                num_workers = max(2, min(6, cpu_count // 2))
            except:
                num_workers = 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Always enable for CUDA (RTX 4060 Ti)
        persistent_workers=False,  # Disabled for Windows compatibility
        prefetch_factor=None,  # Disabled when num_workers=0
        drop_last=True  # More stable training
    )


# =============================================================================
# Feature Computation
# =============================================================================

def calculate_eeg_features(eeg_segment_np: np.ndarray, sfreq: float = TARGET_SFREQ) -> np.ndarray:
    """
    Calculate 12-dimensional neurophysiological feature vector with robust numerical stability.
    
    Features:
        - 5 band powers (delta, theta, alpha, beta, gamma)
        - 2 Hjorth parameters (mobility, complexity)
        - 5 complexity metrics (spectral entropy, permutation entropy, 
          Higuchi fractal dimension, kurtosis, decorrelation time)
    
    Args:
        eeg_segment_np: EEG segment of shape (C, L) or averaged to (L,)
        sfreq: Sampling frequency in Hz
    
    Returns:
        Feature vector of shape (12,) - guaranteed no NaN/Inf
    """
    # Initialize with zeros (safe default)
    features = np.zeros(12, dtype=np.float32)
    
    if not SCIPY_AVAILABLE:
        return features
    
    try:
        # Convert to float64 for numerical precision
        eeg_segment_np = eeg_segment_np.astype(np.float64)
        
        # Average across channels if multi-channel
        if eeg_segment_np.ndim > 1:
            eeg_segment_mean = np.mean(eeg_segment_np, axis=0)
        else:
            eeg_segment_mean = eeg_segment_np
        
        # Validate input: must have reasonable values and length
        if len(eeg_segment_mean) < 64:
            return features  # Too short to compute features
        
        # Remove extreme outliers first (beyond 10 std)
        signal_std = np.std(eeg_segment_mean)
        signal_mean = np.mean(eeg_segment_mean)
        if signal_std > 1e-10:  # Only clip if there's actual variation
            eeg_segment_mean = np.clip(eeg_segment_mean, 
                                       signal_mean - 10 * signal_std, 
                                       signal_mean + 10 * signal_std)
        
        # 1. Spectral Features (5 features: band powers)
        try:
            nperseg = min(256, len(eeg_segment_mean) // 4)
            nperseg = max(64, nperseg)  # Minimum segment size
            freqs, psd = welch(eeg_segment_mean, fs=sfreq, nperseg=nperseg)
            
            # Add epsilon to prevent division by zero
            total_power = np.sum(psd) + 1e-12
            
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 70)
            }
            
            for idx, band in enumerate(bands.values()):
                band_mask = (freqs >= band[0]) & (freqs < band[1])
                if np.any(band_mask):
                    band_power = np.sum(psd[band_mask])
                    # Normalize and clamp to [0, 1]
                    normalized_power = np.clip(band_power / total_power, 0.0, 1.0)
                    features[idx] = normalized_power
                else:
                    features[idx] = 0.0
        except Exception:
            # Band powers default to 0
            pass

        # 2. Hjorth Parameters (2 features: mobility, complexity)
        try:
            diff1 = np.diff(eeg_segment_mean, n=1)
            diff2 = np.diff(eeg_segment_mean, n=2)
            
            var0 = np.var(eeg_segment_mean) + 1e-12
            var1 = np.var(diff1) + 1e-12
            var2 = np.var(diff2) + 1e-12
            
            mobility = np.sqrt(var1 / var0)
            complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
            
            # Clamp to reasonable ranges
            features[5] = np.clip(mobility, 0.0, 10.0)
            features[6] = np.clip(complexity, 0.0, 10.0)
        except Exception:
            features[5] = 1.0  # Default mobility
            features[6] = 1.0  # Default complexity

        # 3. Complexity & Statistical Features (5 features)
        # Spectral entropy
        try:
            spec_ent = antropy.spectral_entropy(eeg_segment_mean, sf=sfreq, method='welch', normalize=True)
            features[7] = np.clip(spec_ent, 0.0, 1.0)
        except Exception:
            features[7] = 0.5
        
        # Permutation entropy
        try:
            perm_ent = antropy.perm_entropy(eeg_segment_mean, order=3, delay=1, normalize=True)
            features[8] = np.clip(perm_ent, 0.0, 1.0)
        except Exception:
            features[8] = 0.5
        
        # Higuchi fractal dimension
        try:
            hfd = antropy.higuchi_fd(eeg_segment_mean, kmax=10)
            # Typical range is 1-2, normalize to ~0-1
            features[9] = np.clip((hfd - 1.0), 0.0, 2.0)
        except Exception:
            features[9] = 0.5
        
        # Kurtosis (normalized)
        try:
            kurt = stats.kurtosis(eeg_segment_mean, fisher=True)
            # Kurtosis can be very large, clip to [-10, 10]
            features[10] = np.clip(kurt / 10.0, -1.0, 1.0)
        except Exception:
            features[10] = 0.0
        
        # Decorrelation time (normalized)
        try:
            centered = eeg_segment_mean - np.mean(eeg_segment_mean)
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[len(centered) - 1:]
            
            # Normalize autocorrelation
            if autocorr[0] > 1e-12:
                autocorr_norm = autocorr / autocorr[0]
                # Find where it drops below 1/e
                threshold_idx = np.argmax(autocorr_norm < (1.0 / np.e))
                if threshold_idx > 0:
                    decorr_time = threshold_idx / len(eeg_segment_mean)
                    features[11] = np.clip(decorr_time, 0.0, 1.0)
                else:
                    features[11] = 0.5
            else:
                features[11] = 0.5
        except Exception:
            features[11] = 0.5
        
        # Final safety: replace any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Final clamp to reasonable range
        features = np.clip(features, -5.0, 5.0)
        
    except Exception as e:
        # If anything fails catastrophically, return zeros
        print(f"[WARN] Feature computation failed: {e}")
        features = np.zeros(12, dtype=np.float32)
    
    return features.astype(np.float32)


# =============================================================================
# Data Balancing Functions
# =============================================================================

def balance_dataset_by_undersampling(
    segments_list: List,
    labels_list: List,
    majority_class_labels: List[int] = [0],
    minority_class_labels: List[int] = [1, 2],
    ratio_to_minority: float = INITIAL_UNDERSAMPLE_RATIO
) -> Tuple[List, List]:
    """
    Balance dataset by undersampling majority class.
    
    Args:
        segments_list: List of EEG segments
        labels_list: List of labels
        majority_class_labels: Labels of majority classes (default: [0])
        minority_class_labels: Labels of minority classes (default: [1, 2])
        ratio_to_minority: Target ratio of majority to minority (default: 3.0)
    
    Returns:
        Tuple of (balanced_segments, balanced_labels)
    """
    print(f"\n--- Balancing Dataset by Undersampling (Ratio: {ratio_to_minority}) ---")
    original_counts = Counter(labels_list)
    print(f"Original class distribution: {original_counts}")
    
    minority_counts = {label: original_counts.get(label, 0) for label in minority_class_labels}
    if not any(minority_counts.values()):
        print("No minority class samples found. No undersampling performed.")
        return segments_list, labels_list
    
    largest_minority_count = max(minority_counts.values()) if minority_counts else 0
    if largest_minority_count == 0:
        print("Minority classes have 0 samples. No undersampling performed.")
        return segments_list, labels_list
    
    target_majority_count = int(largest_minority_count * ratio_to_minority)
    
    balanced_segments, balanced_labels = [], []
    class_data_map = {label: [] for label in original_counts.keys()}
    for segment, label in zip(segments_list, labels_list):
        class_data_map[label].append(segment)
    
    rng = np.random.RandomState(SEED_VALUE)
    
    for class_label, class_segments in class_data_map.items():
        if class_label in majority_class_labels and len(class_segments) > target_majority_count:
            print(f"Undersampling class {class_label} from {len(class_segments)} to {target_majority_count}")
            rng.shuffle(class_segments)
            balanced_segments.extend(class_segments[:target_majority_count])
            balanced_labels.extend([class_label] * target_majority_count)
        else:
            balanced_segments.extend(class_segments)
            balanced_labels.extend([class_label] * len(class_segments))
    
    final_counts = Counter(balanced_labels)
    print(f"Balanced class distribution: {final_counts}")
    
    # Shuffle
    combined = list(zip(balanced_segments, balanced_labels))
    rng.shuffle(combined)
    balanced_segments, balanced_labels = zip(*combined) if combined else ([], [])
    
    return list(balanced_segments), list(balanced_labels)


def balance_dataset_with_smote(
    segments_list: List,
    labels_list: List,
    target_ratio: float = 0.8,
    smote_k_neighbors: int = 3
) -> Tuple[List, List]:
    """
    Balance dataset using SMOTE (Synthetic Minority Over-sampling Technique).
    
    Args:
        segments_list: List of EEG segments
        labels_list: List of labels
        target_ratio: Target ratio of minority to majority (default: 0.8)
        smote_k_neighbors: Number of neighbors for SMOTE (default: 3)
    
    Returns:
        Tuple of (balanced_segments, balanced_labels)
    """
    if not IMBLEARN_AVAILABLE:
        print("[WARN] imbalanced-learn not available. Returning original data.")
        return segments_list, labels_list
    
    print(f"\n--- Balancing Dataset with SMOTE (Target Ratio: {target_ratio}) ---")
    original_counts = Counter(labels_list)
    print(f"Original class distribution: {original_counts}")

    if not segments_list:
        print("Segment list is empty. Skipping balancing.")
        return [], []

    segments_np = np.array(segments_list)
    labels_np = np.array(labels_list)
    
    # Flatten data for SMOTE: (n_samples, n_features)
    n_samples, n_channels, n_timesteps = segments_np.shape
    data_reshaped = segments_np.reshape(n_samples, -1)

    # Determine target number of samples for minority classes
    majority_class_label = original_counts.most_common(1)[0][0]
    n_majority = original_counts[majority_class_label]
    
    sampling_strategy = {}
    for label, count in original_counts.items():
        if label != majority_class_label:
            target_count = int(n_majority * target_ratio)
            if count < target_count:
                # Ensure k_neighbors is not greater than number of samples in class
                k = min(smote_k_neighbors, count - 1)
                if k > 0:
                    sampling_strategy[label] = target_count
                else:
                    print(f"Skipping SMOTE for class {label} due to insufficient samples ({count}).")

    if not sampling_strategy:
        print("No classes eligible for SMOTE. Using original data.")
        return segments_list, labels_list

    print(f"Applying SMOTE with strategy: {sampling_strategy}")
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=SEED_VALUE)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(data_reshaped, labels_np)
    except Exception as e:
        print(f"SMOTE failed: {e}. Returning original data.")
        return segments_list, labels_list

    # Reshape back to (n_samples, n_channels, n_timesteps)
    X_resampled_reshaped = X_resampled.reshape(-1, n_channels, n_timesteps)

    final_counts = Counter(y_resampled)
    print(f"Balanced class distribution after SMOTE: {final_counts}")
    
    # Shuffle
    combined = list(zip(X_resampled_reshaped.tolist(), y_resampled.tolist()))
    np.random.RandomState(SEED_VALUE).shuffle(combined)
    balanced_segments, balanced_labels = zip(*combined) if combined else ([], [])
    
    return list(balanced_segments), list(balanced_labels)


def oversample_data(
    X_data: np.ndarray,
    y_data: np.ndarray,
    strategy: str = 'smote_features',
    random_state: int = SEED_VALUE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes.
    
    Args:
        X_data: Features or raw EEG data
        y_data: Labels
        strategy: 'smote_features' (for feature data) or 'random_raw' (for raw EEG)
        random_state: Random seed
    
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not IMBLEARN_AVAILABLE:
        print("[WARN] imbalanced-learn not available. Returning original data.")
        return X_data, y_data
    
    original_counts = Counter(y_data)
    print(f"  Oversampling: Original class distribution: {original_counts}")

    if strategy == 'smote_features':
        # Expects X_data to be 2D (samples, features)
        if X_data.ndim > 2:
            print(f"  [Oversample WARN] X_data has {X_data.ndim} dims for SMOTE, expecting 2. Attempting reshape.")
            X_data_reshaped = X_data.reshape(X_data.shape[0], -1)
        else:
            X_data_reshaped = X_data
        
        # Check if any class has too few samples for SMOTE
        min_samples_in_class = min(original_counts.values())
        k_neighbors = min(5, max(1, min_samples_in_class - 1))
        
        if k_neighbors < 1:
            print(f"  [Oversample WARN] Smallest class has {min_samples_in_class} samples. Using RandomOverSampler.")
            oversampler = RandomOverSampler(random_state=random_state)
            X_resampled, y_resampled = oversampler.fit_resample(X_data_reshaped, y_data)
        else:
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X_data_reshaped, y_data)
            except ValueError as e:
                print(f"  [Oversample ERROR] SMOTE failed: {e}. Falling back to RandomOverSampler.")
                oversampler = RandomOverSampler(random_state=random_state)
                X_resampled, y_resampled = oversampler.fit_resample(X_data_reshaped, y_data)
        
    elif strategy == 'random_raw':
        # Expects X_data to be (samples, channels, length) or (samples, features)
        ros = RandomOverSampler(random_state=random_state)
        original_shape_X = X_data.shape
        
        if X_data.ndim == 3:
            X_data_flat = X_data.reshape(original_shape_X[0], -1)
            X_resampled_flat, y_resampled = ros.fit_resample(X_data_flat, y_data)
            X_resampled = X_resampled_flat.reshape(X_resampled_flat.shape[0], original_shape_X[1], original_shape_X[2])
        else:
            X_resampled, y_resampled = ros.fit_resample(X_data, y_data)
    else:
        raise ValueError(f"Unknown oversampling strategy: {strategy}")

    final_counts = Counter(y_resampled)
    print(f"  Oversampled class distribution: {final_counts}")
    
    return X_resampled, y_resampled


# =============================================================================
# Preprocessed EEG Dataset
# =============================================================================

class EegPreprocessedDataset(Dataset):
    """
    Dataset with normalization, augmentation, and feature pre-computation.
    
    This dataset:
    - Normalizes EEG data using provided mean and std
    - Optionally augments minority classes with time warping
    - Pre-computes neurophysiological features for all samples
    
    Args:
        raw_data_np: Raw EEG data of shape (N, C, L)
        labels_np: Labels of shape (N,)
        mean_np: Normalization mean of shape (1, C, 1)
        std_np: Normalization std of shape (1, C, 1)
        augment: Whether to augment minority classes (default: False)
        augment_classes: Which classes to augment (default: [1, 2])
    """
    
    def __init__(
        self,
        raw_data_np: np.ndarray,
        labels_np: np.ndarray,
        mean_np: np.ndarray,
        std_np: np.ndarray,
        augment: bool = False,
        augment_classes: List[int] = [1, 2]
    ):
        print("\n--- Initializing Preprocessed Dataset ---")

        current_raw_data = torch.from_numpy(raw_data_np).float()
        current_labels = torch.from_numpy(labels_np).long()

        # Minority augmentation
        if augment and len(current_raw_data) > 0:
            print(f"  Performing augmentation for minority classes {augment_classes}...")
            minority_mask = torch.zeros_like(current_labels, dtype=torch.bool)
            for cls in augment_classes:
                minority_mask |= (current_labels == cls)

            segments_to_augment = current_raw_data[minority_mask]
            labels_to_augment = current_labels[minority_mask]

            if segments_to_augment.shape[0] > 0:
                augmented_segments = []
                C, L = segments_to_augment.shape[1], segments_to_augment.shape[2]
                stretch_factors = torch.rand(segments_to_augment.shape[0]) * 0.1 + 0.95

                original_x_coords = np.arange(L)
                segments_to_augment_np = segments_to_augment.numpy()

                for i in range(segments_to_augment.shape[0]):
                    stretch_factor = stretch_factors[i].item()
                    new_len = int(L * stretch_factor)
                    new_x_coords = np.linspace(0, L - 1, new_len)

                    warped_channels = [
                        np.interp(new_x_coords, original_x_coords, ch)
                        for ch in segments_to_augment_np[i]
                    ]
                    warped_segment = np.stack(warped_channels)

                    final_segment = np.zeros((C, L), dtype=np.float32)
                    if new_len > L:
                        start = np.random.randint(0, new_len - L + 1)
                        final_segment = warped_segment[:, start:start + L]
                    else:
                        final_segment[:, :new_len] = warped_segment

                    augmented_segments.append(final_segment)

                augmented_data_tensor = torch.from_numpy(np.array(augmented_segments)).float()
                current_raw_data = torch.cat([current_raw_data, augmented_data_tensor], dim=0)
                current_labels = torch.cat([current_labels, labels_to_augment], dim=0)
                print(f"  Augmentation complete. New sample count: {current_raw_data.shape[0]}")

        # Store tensors
        self.raw_data = current_raw_data
        self.labels = current_labels

        # Normalize with epsilon guard
        mean_t = torch.from_numpy(mean_np).float()
        std_t = torch.from_numpy(std_np).float()
        
        # Add epsilon to prevent division by zero
        std_t = torch.clamp(std_t, min=1e-8)
        
        # CRITICAL FIX for microvolt data normalization:
        # For EEG data in microvolts (std ~0.00007), the pipeline sets fold_std_np = 0.0001
        # This effectively scales UP by 10,000×, bringing data to std ~0.7-1.0
        # This is PERFECT for neural networks (no further normalization needed)
        # 
        # Best practice for microvolt EEG:
        # 1. Scale by dividing by 0.0001 (multiply by 10,000×)
        # 2. Result: mean≈0, std≈0.7-1.0 (neural network friendly)
        # 3. NO aggressive clipping (destroys std)
        # 4. Only safety clip at ±50 to catch rare artifacts
        
        self.scaled_data = (self.raw_data - mean_t) / std_t
        
        # Safety clipping ONLY for extreme artifacts (should affect <0.01% of data)
        # Clipping at ±50 instead of ±5 preserves the natural std of scaled data
        # For properly scaled data (std~0.7), 99.99% of values are within ±3, so ±50 is safe
        self.scaled_data = torch.clamp(self.scaled_data, -50.0, 50.0)
        
        # Check for extreme values after normalization
        if len(self.scaled_data) > 0:
            data_min = self.scaled_data.min().item()
            data_max = self.scaled_data.max().item()
            data_mean = self.scaled_data.mean().item()
            data_std = self.scaled_data.std().item()
            print(f"  [DEBUG] Scaled data stats (after 10000× upscaling):")
            print(f"    Range: [{data_min:.4f}, {data_max:.4f}]")
            print(f"    Mean: {data_mean:.4f}, Std: {data_std:.4f}")
            
            # For microvolts EEG scaled by 10000×, expect std ~0.7-1.0
            # This is GOOD for neural networks (no need for std=1.0 exactly)
            if data_std < 0.3 or data_std > 3.0:
                print(f"  [WARNING] Std={data_std:.4f} outside expected range [0.3, 3.0] for scaled EEG")
            if abs(data_mean) > 0.5:
                print(f"  [WARNING] Mean={data_mean:.4f} has significant offset (should be ~0)")
            
            # Final NaN/Inf check
            if torch.isnan(self.scaled_data).any() or torch.isinf(self.scaled_data).any():
                print(f"  [WARNING] NaN/Inf detected in scaled data! Replacing with zeros.")
                self.scaled_data = torch.nan_to_num(self.scaled_data, nan=0.0, posinf=50.0, neginf=-50.0)

        # Pre-compute features
        print("  Pre-computing neurophysiological features...")
        batch_size = 256
        all_features = []
        raw_data_cpu_np = self.raw_data.cpu().numpy()

        if len(raw_data_cpu_np) > 0:
            for i in tqdm(range(0, len(raw_data_cpu_np), batch_size), desc="Calculating Features", leave=False):
                batch_end = min(i + batch_size, len(raw_data_cpu_np))
                batch_features = [
                    calculate_eeg_features(raw_data_cpu_np[j])
                    for j in range(i, batch_end)
                ]
                all_features.extend(batch_features)

        self.features = torch.from_numpy(np.array(all_features)).float() if all_features else torch.empty(0, VAE_FEATURE_COND_DIM)
        
        # Robust feature normalization with proper handling of outliers
        if len(self.features) > 0 and self.features.numel() > 0:
            # Simple robust normalization: per-feature percentile clipping
            for feat_idx in range(self.features.shape[1]):
                feat_col = self.features[:, feat_idx]
                
                # Use percentiles for robust statistics
                p01 = torch.quantile(feat_col, 0.01)
                p99 = torch.quantile(feat_col, 0.99)
                
                # Clip to percentile range
                feat_col = torch.clamp(feat_col, p01, p99)
                
                # Z-score normalization
                feat_mean = feat_col.mean()
                feat_std = feat_col.std()
                if feat_std > 1e-6:
                    feat_col = (feat_col - feat_mean) / feat_std
                else:
                    feat_col = feat_col - feat_mean  # Just center if no variance
                
                # Final safety clamp
                self.features[:, feat_idx] = torch.clamp(feat_col, -5.0, 5.0)
            
            print(f"  [INFO] Features normalized. Stats:")
            print(f"    Mean: {self.features.mean().item():.6f}, Std: {self.features.std().item():.6f}")
            print(f"    Min: {self.features.min().item():.4f}, Max: {self.features.max().item():.4f}")
        
        self.features_matrix = self.features.numpy() if len(self.features) > 0 else None

        # Sanity checks
        assert len(self.raw_data) == len(self.labels), "Mismatch raw_data vs labels!"
        assert len(self.raw_data) == len(self.scaled_data), "Mismatch raw_data vs scaled_data!"
        assert len(self.raw_data) == len(self.features), "Mismatch raw_data vs features!"

        print(f"--- Dataset Initialized. Total samples: {len(self.raw_data)} ---")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        x_raw = self.raw_data[idx]
        x_scaled = self.scaled_data[idx]
        feat = self.features[idx] if len(self.features) > 0 else torch.zeros(VAE_FEATURE_COND_DIM, dtype=torch.float32)
        y = self.labels[idx]

        # Ensure 2D shape [C, L]
        while x_raw.ndim > 2:
            x_raw = x_raw.squeeze(0)
        while x_scaled.ndim > 2:
            x_scaled = x_scaled.squeeze(0)
        
        # Reshape if flattened to 1D
        if x_raw.ndim == 1:
            expected_length = len(x_raw) // COMMON_EEG_CHANNELS
            x_raw = x_raw.view(COMMON_EEG_CHANNELS, expected_length)
        if x_scaled.ndim == 1:
            expected_length = len(x_scaled) // COMMON_EEG_CHANNELS
            x_scaled = x_scaled.view(COMMON_EEG_CHANNELS, expected_length)
        
        # Final validation
        assert x_scaled.ndim == 2, f"x_scaled bad shape {x_scaled.shape}, expected 2D [C, L]"
        assert x_raw.ndim == 2, f"x_raw bad shape {x_raw.shape}, expected 2D [C, L]"
        assert x_scaled.shape[0] == COMMON_EEG_CHANNELS, f"Expected {COMMON_EEG_CHANNELS} channels"
        assert x_raw.shape[0] == COMMON_EEG_CHANNELS, f"Expected {COMMON_EEG_CHANNELS} channels"
        
        # Safety check: replace any NaN/Inf (should not happen, but be defensive)
        if torch.isnan(x_scaled).any() or torch.isinf(x_scaled).any():
            x_scaled = torch.nan_to_num(x_scaled, nan=0.0, posinf=50.0, neginf=-50.0)
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            feat = torch.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0)

        return (
            x_scaled,  # Normalized input [C, L]
            x_raw,     # Raw waveform [C, L]
            feat,      # Features [F]
            y          # Label
        )


# =============================================================================
# Cached Latent Dataset
# =============================================================================

class CachedLatentDataset(Dataset):
    """
    Dataset with pre-computed VAE latent codes for efficient LDM training.
    
    This dataset:
    - Encodes all EEG data through VAE encoder once
    - Caches z0 = mu / ldm_scaling_factor for each sample
    - Stores raw data and features for auxiliary losses
    
    Args:
        base_loader: DataLoader with EEG data
        vae: Trained VAE model
        device: Device to run encoding on
        ldm_scaling_factor: Scaling factor for latent codes
        data_mean: Normalization mean
        data_std: Normalization std
    """
    
    def __init__(
        self,
        base_loader,
        vae,
        device,
        ldm_scaling_factor: float,
        data_mean: np.ndarray,
        data_std: np.ndarray
    ):
        self.records = []
        vae.eval()
        mean_t = torch.as_tensor(data_mean, dtype=torch.float32, device=device).view(1, -1, 1)
        std_t = torch.as_tensor(data_std, dtype=torch.float32, device=device).view(1, -1, 1)
        
        import time
        start_time = time.time()
        total_batches = len(base_loader)
        print(f"Building cached latent dataset from {total_batches} batches...")
        print(f"  (This may take a few minutes for large datasets - please wait)")
        with torch.no_grad():
            for batch_idx, batch in enumerate(base_loader):
                # More frequent progress updates
                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    print(f"  Caching latents: {batch_idx+1}/{total_batches} ({progress_pct:.1f}%)", flush=True)
                
                # Handle batch format
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    x_scaled, x_raw, feat, y = batch
                    if x_scaled.ndim == 4:
                        x_scaled = x_scaled.squeeze(1)
                    if x_scaled.ndim == 2:
                        B = x_scaled.shape[0]
                        x_scaled = x_scaled.view(B, data_mean.shape[1], -1)
                else:
                    x_scaled = batch
                    x_raw = None
                    feat = None
                    y = None
                    if x_scaled.ndim == 4:
                        x_scaled = x_scaled.squeeze(1)
                    if x_scaled.ndim == 2:
                        B = x_scaled.shape[0]
                        x_scaled = x_scaled.view(B, data_mean.shape[1], -1)
                
                x_scaled = x_scaled.to(device)
                assert x_scaled.ndim == 3, f"VAE encode expects 3D input (B,C,L), got {x_scaled.shape}"
                
                # Encode through VAE
                encode_result = vae.encode(x_scaled)
                if isinstance(encode_result, (tuple, list)):
                    mu = encode_result[0]  # Always take mu
                else:
                    mu = encode_result
                
                z0 = mu / float(ldm_scaling_factor)
                if batch_idx == 0:
                    print(f"[DEBUG] CachedLatentDataset batch 0: x_scaled={x_scaled.shape}, z0={z0.shape}")
                
                # Compute raw data if not provided
                if x_raw is None:
                    x_raw_computed = x_scaled * std_t + mean_t
                else:
                    x_raw_computed = x_raw.to(device)
                
                # Store individual samples
                for i in range(z0.shape[0]):
                    feat_i = feat[i].cpu() if feat is not None and i < len(feat) else torch.zeros(VAE_FEATURE_COND_DIM, dtype=torch.float32)
                    y_i = y[i].cpu() if y is not None and i < len(y) else torch.tensor(0, dtype=torch.long)
                    
                    self.records.append({
                        "z0": z0[i].cpu(),
                        "x_raw": x_raw_computed[i].cpu(),
                        "feat": feat_i,
                        "y": y_i,
                    })
        
        elapsed = time.time() - start_time
        print(f"  Cached {len(self.records)} latent codes in {elapsed:.1f}s")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            record["z0"],      # Latent z0
            record["x_raw"],   # Raw EEG data
            record["feat"],    # Features
            record["y"]        # Label
        )


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Testing preprocessing utilities...")
    
    # Test GenericDataset
    print("\n1. Testing GenericDataset...")
    data = np.random.randn(100, 18, 3840).astype(np.float32)
    labels = np.random.randint(0, 3, 100)
    dataset = GenericDataset(data, labels)
    assert len(dataset) == 100, "Dataset length mismatch"
    x, y = dataset[0]
    assert x.shape == (18, 3840), "Data shape mismatch"
    print("   GenericDataset: PASS")
    
    # Test create_optimized_dataloader
    print("\n2. Testing create_optimized_dataloader...")
    loader = create_optimized_dataloader(dataset, batch_size=16)
    batch = next(iter(loader))
    assert len(batch) == 2, "Batch should have 2 elements"
    print("   DataLoader: PASS")
    
    # Test calculate_eeg_features
    if SCIPY_AVAILABLE:
        print("\n3. Testing calculate_eeg_features...")
        segment = np.random.randn(18, 3840).astype(np.float32)
        features = calculate_eeg_features(segment, sfreq=256)
        assert features.shape == (12,), f"Feature shape mismatch: {features.shape}"
        assert not np.isnan(features).any(), "Features contain NaN"
        print(f"   Features computed: {features.shape}")
        print("   calculate_eeg_features: PASS")
    else:
        print("\n3. Skipping calculate_eeg_features test (scipy not available)")
    
    # Test balancing functions
    if IMBLEARN_AVAILABLE:
        print("\n4. Testing balance_dataset_by_undersampling...")
        segments = [np.random.randn(18, 100) for _ in range(100)]
        labels = [0] * 70 + [1] * 20 + [2] * 10
        balanced_segs, balanced_labs = balance_dataset_by_undersampling(segments, labels)
        print(f"   Original: 100, Balanced: {len(balanced_segs)}")
        print("   balance_dataset_by_undersampling: PASS")
    else:
        print("\n4. Skipping balancing tests (imbalanced-learn not available)")
    
    print("\n[SUCCESS] All preprocessing tests passed!")
