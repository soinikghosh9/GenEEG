"""
Data Processing Utilities

Functions for data balancing, oversampling, and DataLoader creation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_dataset_by_undersampling(
    X: List[np.ndarray],
    y: List[int],
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Balance dataset by undersampling majority class.
    
    Args:
        X: List of data samples
        y: List of labels
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    # Convert to numpy arrays
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Reshape for sklearn if needed
    original_shape = X_array.shape
    if X_array.ndim > 2:
        n_samples = X_array.shape[0]
        X_flat = X_array.reshape(n_samples, -1)
    else:
        X_flat = X_array
    
    # Apply random undersampling
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_flat, y_array)
    
    # Reshape back to original shape
    if X_array.ndim > 2:
        X_resampled = X_resampled.reshape(-1, *original_shape[1:])
    
    # Convert back to lists
    X_balanced = list(X_resampled)
    y_balanced = list(y_resampled)
    
    print(f"  Undersampled: {len(X)} → {len(X_balanced)} samples")
    print(f"  Class distribution: {Counter(y_balanced)}")
    
    return X_balanced, y_balanced


def balance_dataset_with_smote(
    X: List[np.ndarray],
    y: List[int],
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Balance dataset using SMOTE (Synthetic Minority Over-sampling Technique).
    
    Args:
        X: List of data samples
        y: List of labels
        random_state: Random seed for reproducibility
        k_neighbors: Number of nearest neighbors for SMOTE
    
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    # Convert to numpy arrays
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Check if we have enough samples
    min_class_count = min(Counter(y_array).values())
    if min_class_count < k_neighbors:
        k_neighbors = max(1, min_class_count - 1)
        print(f"  Warning: Reducing k_neighbors to {k_neighbors} due to small class size")
    
    if min_class_count < 2:
        print(f"  Warning: Cannot apply SMOTE with < 2 samples per class")
        return X, y
    
    # Reshape for sklearn if needed
    original_shape = X_array.shape
    if X_array.ndim > 2:
        n_samples = X_array.shape[0]
        X_flat = X_array.reshape(n_samples, -1)
    else:
        X_flat = X_array
    
    try:
        # Apply SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_flat, y_array)
        
        # Reshape back to original shape
        if X_array.ndim > 2:
            X_resampled = X_resampled.reshape(-1, *original_shape[1:])
        
        # Convert back to lists
        X_balanced = list(X_resampled)
        y_balanced = list(y_resampled)
        
        print(f"  SMOTE applied: {len(X)} → {len(X_balanced)} samples")
        print(f"  Class distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    except Exception as e:
        print(f"  Warning: SMOTE failed ({e}), returning original data")
        return X, y


def oversample_data(
    X: np.ndarray,
    y: np.ndarray,
    target_samples_per_class: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes to reach target samples per class.
    
    Args:
        X: Data array of shape (N, ...)
        y: Labels array of shape (N,)
        target_samples_per_class: Target number of samples for each class
        random_state: Random seed
    
    Returns:
        Tuple of (X_oversampled, y_oversampled)
    """
    np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    X_oversampled = []
    y_oversampled = []
    
    for cls in unique_classes:
        # Get samples for this class
        cls_mask = (y == cls)
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]
        
        n_cls_samples = len(X_cls)
        
        if n_cls_samples >= target_samples_per_class:
            # Already enough samples, just add them
            X_oversampled.append(X_cls[:target_samples_per_class])
            y_oversampled.append(y_cls[:target_samples_per_class])
        else:
            # Need to oversample
            n_to_add = target_samples_per_class - n_cls_samples
            
            # Random sampling with replacement
            additional_indices = np.random.choice(n_cls_samples, n_to_add, replace=True)
            X_additional = X_cls[additional_indices]
            y_additional = y_cls[additional_indices]
            
            # Combine original and additional
            X_oversampled.append(np.concatenate([X_cls, X_additional], axis=0))
            y_oversampled.append(np.concatenate([y_cls, y_additional], axis=0))
    
    # Concatenate all classes
    X_final = np.concatenate(X_oversampled, axis=0)
    y_final = np.concatenate(y_oversampled, axis=0)
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(X_final))
    X_final = X_final[shuffle_indices]
    y_final = y_final[shuffle_indices]
    
    print(f"  Oversampled: {len(X)} → {len(X_final)} samples")
    print(f"  Class distribution: {Counter(y_final)}")
    
    return X_final, y_final


def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create an optimized DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (default: 0 for Windows stability)
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    # Windows multiprocessing can be unstable, default to 0 workers
    # RTX 4060 Ti has fast PCIe transfer, so pin_memory helps
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=False if num_workers == 0 else False,  # Disabled for stability
        prefetch_factor=None if num_workers == 0 else 2
    )
    
    return loader


def get_dataset(dataset_name: str, data_root: str):
    """
    Get dataset loader based on name.
    
    Args:
        dataset_name: Name of dataset ('siena' or 'chbmit')
        data_root: Root directory of dataset
    
    Returns:
        Dataset loader instance (SienaLoader or CHBMITLoader)
    """
    from .loaders import SienaLoader, CHBMITLoader
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'siena':
        return SienaLoader(data_root)
    elif dataset_name in ['chbmit', 'chb-mit', 'chb_mit']:
        return CHBMITLoader(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'siena' or 'chbmit'")
