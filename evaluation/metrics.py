"""
Evaluation Metrics and Plotting Module

This module provides comprehensive evaluation functions for seizure detection,
including confusion matrices, ROC curves, and various performance metrics.

Key Components:
    - plot_confusion_matrix: Visualize confusion matrix with annotations
    - plot_roc_auc_curves: Multi-class ROC curves with micro/macro averaging
    - Supports both sklearn and PyTorch model evaluation

Author: GenEEG Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names_list: list[str],
    title: str = 'Confusion Matrix',
    cmap = plt.cm.Blues,
    save_path: str = None,
    normalize: bool = False
) -> None:
    """
    Plot and save a publication-quality confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix as numpy array (num_classes, num_classes)
        class_names_list: List of class names for axis labels
        title: Plot title (default: "Confusion Matrix")
        cmap: Colormap for the confusion matrix (default: Blues)
        save_path: Path to save the plot (None = no saving)
        normalize: If True, normalize confusion matrix by row sums
    
    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(
        ...     cm, ["Ictal", "Interictal", "Preictal"],
        ...     title="Test Set Confusion Matrix",
        ...     save_path="./cm_test.png"
        ... )
    
    Note:
        - Normalized matrices show percentages (format: .2f)
        - Raw matrices show counts (format: d)
        - Figure size scales with number of classes
        - Uses high DPI (300) for publication quality
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # Format for normalized numbers
        print("Normalized confusion matrix")
    else:
        fmt = 'd'  # Format for raw counts
        print('Confusion matrix, without normalization')
    
    # Set aesthetic styling with soft colors
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'axes.linewidth': 1.2,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
    })
    
    # Dynamic figure sizing based on number of classes
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names_list) * 1.5),
                max(7, len(class_names_list) * 1.3))
    )
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=17, fontweight='bold', pad=20, color='#2C3E50')
    
    # Create colorbar with soft styling
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    if normalize:
        cbar.set_label('Proportion', rotation=270, labelpad=20, fontsize=14, fontweight='600', color='#2C3E50')
    else:
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=14, fontweight='600', color='#2C3E50')
    
    # Set tick labels with soft styling
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names_list, fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_yticklabels(class_names_list, fontsize=13, fontweight='600', color='#2C3E50')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Turn off grid for cleaner cells
    ax.grid(False)
    
    # Add text annotations with better visibility
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "#2C3E50",
                   fontsize=12, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=15, fontweight='600', labelpad=10, color='#2C3E50')
    ax.set_xlabel('Predicted Label', fontsize=15, fontweight='600', labelpad=10, color='#2C3E50')
    
    # Add soft border
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(2)
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    [VISUALIZATION] Confusion matrix saved to: {save_path}")
    plt.close(fig)


def plot_roc_auc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
    class_names_list: list[str],
    title_prefix: str = "ROC Curve",
    save_path: str = None
) -> dict:
    """
    Plot ROC curves for multi-class classification (One-vs-Rest) with AUC scores.
    
    Args:
        y_true: True labels (N_samples,) as integer class indices
        y_pred_proba: Predicted probabilities (N_samples, N_classes)
        num_classes: Number of classes
        class_names_list: List of class names
        title_prefix: Prefix for plot title (default: "ROC Curve")
        save_path: Path to save the plot (None = no saving)
    
    Returns:
        Dictionary containing AUC scores:
            - auc_class_<name>: Per-class AUC scores
            - auc_micro_avg: Micro-averaged AUC
            - auc_macro_avg: Macro-averaged AUC
            - auc_weighted_avg_ovr: Weighted average AUC (OvR)
    
    Example:
        >>> auc_scores = plot_roc_auc_curves(
        ...     y_test, y_proba, num_classes=3,
        ...     class_names_list=["Ictal", "Interictal", "Preictal"],
        ...     title_prefix="Test Set ROC",
        ...     save_path="./roc_test.png"
        ... )
        >>> print(f"Weighted AUC: {auc_scores['auc_weighted_avg_ovr']:.4f}")
    
    Note:
        - Handles missing classes gracefully
        - Computes micro-average (global pooling)
        - Computes macro-average (arithmetic mean)
        - Uses One-vs-Rest (OvR) strategy for multi-class
        - Skips classes with no positive samples
    """
    if num_classes <= 1:
        print(f"  [WARN ROC] ROC AUC not supported for single-class ({num_classes}) data. Skipping for {title_prefix}.")
        return None
    
    # Set aesthetic styling with soft colors
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'grid.alpha': 0.3,
    })
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(11, 9))
    ax = plt.gca()
    
    # Binarize the true labels for One-vs-Rest
    y_true_binarized = np.zeros((len(y_true), num_classes))
    for i in range(len(y_true)):
        if y_true[i] < num_classes:  # Ensure label is valid
            y_true_binarized[i, y_true[i]] = 1
        else:
            print(f"[WARN ROC] Invalid true label {y_true[i]} encountered for class count {num_classes}. Skipping this sample for OvR.")
    
    # Soft aesthetic color palette for different classes
    colors = ['#E85D75', '#5DADE2', '#48C9B0', '#F39C6B', '#BB8FCE', 
              '#85C1E9', '#F8B88B', '#73C6B6', '#AED6F1', '#D7BDE2']
    
    # Compute ROC curve for each class
    for i in range(num_classes):
        # Check if the current class has any positive samples
        if np.sum(y_true_binarized[:, i]) == 0:
            print(f"  [WARN ROC] Class '{class_names_list[i]}' has no positive samples in y_true. Cannot compute ROC for this class for {title_prefix}.")
            roc_auc[i] = np.nan
            continue
        
        # Check if all samples belong to this class (no negative samples)
        if np.sum(y_true_binarized[:, i]) == len(y_true_binarized):
            print(f"  [WARN ROC] All samples belong to class '{class_names_list[i]}' in y_true. Cannot compute meaningful ROC for this class for {title_prefix}.")
            roc_auc[i] = np.nan
            continue
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        if not np.isnan(roc_auc[i]):
            color = colors[i % len(colors)]
            ax.plot(fpr[i], tpr[i], lw=2.5, alpha=0.85, color=color,
                    label=f'{class_names_list[i]} (AUC = {roc_auc[i]:.3f})')
        else:
            ax.plot([], [], lw=2.5,
                    label=f'{class_names_list[i]} (AUC = N/A)')
    
    # Micro-average ROC curve (global pooling) with soft color
    if np.any(y_true_binarized.ravel()):
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_true_binarized.ravel(),
                y_pred_proba.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            ax.plot(fpr["micro"], tpr["micro"],
                    label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})',
                    color='#E85D75', linestyle=':', linewidth=3.5, alpha=0.85)
        except ValueError as e_micro:
            print(f"  [WARN ROC] Could not compute micro-average ROC for {title_prefix}: {e_micro}")
            roc_auc["micro"] = np.nan
    else:
        print(f"  [WARN ROC] No positive samples in y_true_binarized.ravel() for {title_prefix}. Cannot compute micro-average ROC.")
        roc_auc["micro"] = np.nan
    
    # Macro-average ROC curve (arithmetic mean) with soft color
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes_for_macro = 0
    
    for i in range(num_classes):
        if i in fpr and i in tpr and not np.isnan(roc_auc[i]):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            valid_classes_for_macro += 1
    
    if valid_classes_for_macro > 0:
        mean_tpr /= valid_classes_for_macro
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        ax.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})',
                color='#5DADE2', linestyle=':', linewidth=3.5, alpha=0.85)
    else:
        print(f"  [WARN ROC] No valid classes to compute macro-average ROC for {title_prefix}.")
        roc_auc["macro"] = np.nan
    
    # Plot random chance line with soft styling
    ax.plot([0, 1], [0, 1], color='#BDBDBD', linestyle='--', lw=2, alpha=0.6, label='Chance (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_title(f'{title_prefix} - Multi-class ROC Curves', 
                fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    ax.legend(loc="lower right", fontsize=10, frameon=True, shadow=False, fancybox=True, 
             framealpha=0.95, edgecolor='#BDBDBD')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    
    # Add soft border
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    [VISUALIZATION] ROC AUC curves saved to: {save_path}")
    plt.close()
    
    # Return dictionary of AUC scores
    auc_scores_dict = {
        f"auc_class_{class_names_list[i]}": roc_auc.get(i, np.nan)
        for i in range(num_classes)
    }
    auc_scores_dict["auc_micro_avg"] = roc_auc.get("micro", np.nan)
    auc_scores_dict["auc_macro_avg"] = roc_auc.get("macro", np.nan)
    
    # Compute weighted AUC using sklearn (overall performance metric)
    try:
        auc_weighted = roc_auc_score(
            y_true, y_pred_proba,
            multi_class='ovr', average='weighted'
        )
        auc_scores_dict["auc_weighted_avg_ovr"] = auc_weighted
    except ValueError as e_sklearn_auc:
        print(f"  [WARN ROC] sklearn roc_auc_score (weighted OvR) failed for {title_prefix}: {e_sklearn_auc}. Setting to NaN.")
        auc_scores_dict["auc_weighted_avg_ovr"] = np.nan
    
    return auc_scores_dict


# =============================================================================
# Synthetic Data Quality Metrics
# =============================================================================

def calculate_fid_score(
    real_samples: np.ndarray,
    synthetic_samples: np.ndarray,
    device: str = 'cuda'
) -> float:
    """
    Calculate Fréchet Inception Distance (FID) for EEG data.
    
    Uses a simplified approach: compute statistics in flattened space.
    For EEG, we use spectral features instead of CNN features.
    
    Args:
        real_samples: Real EEG (N, C, T)
        synthetic_samples: Synthetic EEG (M, C, T)
        device: Device for computation
    
    Returns:
        fid: FID score (lower is better)
    """
    import torch
    from scipy.linalg import sqrtm
    
    # CRITICAL FIX: Limit samples to prevent memory overflow
    # FID computation with N samples requires O(N^2) memory for covariance matrix
    # For 20480-dim features, even 1000 samples = 20480x20480 matrix = 3.15 GB
    MAX_SAMPLES_FOR_FID = 500  # Limit to 500 samples max to keep memory under 1 GB
    
    if len(real_samples) > MAX_SAMPLES_FOR_FID:
        # Randomly sample without replacement
        real_indices = np.random.choice(len(real_samples), MAX_SAMPLES_FOR_FID, replace=False)
        real_samples = real_samples[real_indices]
        print(f"    [FID] Subsampled real data: {len(real_samples)} samples")
    
    if len(synthetic_samples) > MAX_SAMPLES_FOR_FID:
        synthetic_indices = np.random.choice(len(synthetic_samples), MAX_SAMPLES_FOR_FID, replace=False)
        synthetic_samples = synthetic_samples[synthetic_indices]
        print(f"    [FID] Subsampled synthetic data: {len(synthetic_samples)} samples")
    
    # Flatten to (N, C*T)
    real_flat = real_samples.reshape(len(real_samples), -1)
    synthetic_flat = synthetic_samples.reshape(len(synthetic_samples), -1)
    
    # Compute statistics
    mu_real = np.mean(real_flat, axis=0)
    mu_synthetic = np.mean(synthetic_flat, axis=0)
    
    cov_real = np.cov(real_flat, rowvar=False)
    cov_synthetic = np.cov(synthetic_flat, rowvar=False)
    
    # FID formula: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    diff = mu_real - mu_synthetic
    diff_squared = np.sum(diff ** 2)
    
    # Matrix square root
    covmean = sqrtm(cov_real @ cov_synthetic)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff_squared + np.trace(cov_real + cov_synthetic - 2 * covmean)
    
    return float(fid)


def calculate_sample_diversity(
    samples: np.ndarray,
    n_pairs: int = 1000
) -> float:
    """
    Calculate intra-class diversity (average pairwise distance).
    
    Args:
        samples: EEG samples (N, C, T)
        n_pairs: Number of random pairs to sample
    
    Returns:
        diversity: Average L2 distance (higher is better)
    """
    n_samples = len(samples)
    if n_samples < 2:
        return 0.0
    
    # Flatten
    samples_flat = samples.reshape(n_samples, -1)
    
    # Sample random pairs
    n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
    distances = []
    
    for _ in range(n_pairs):
        i, j = np.random.choice(n_samples, size=2, replace=False)
        dist = np.linalg.norm(samples_flat[i] - samples_flat[j])
        distances.append(dist)
    
    return float(np.mean(distances))


def compute_feature_statistics(
    real_samples: np.ndarray,
    synthetic_samples: np.ndarray,
    target_sfreq: int = 256
) -> dict:
    """
    Compute neurophysiological feature statistics and KS tests.
    
    Args:
        real_samples: Real EEG (N, C, T)
        synthetic_samples: Synthetic EEG (M, C, T)
        target_sfreq: Sampling frequency
    
    Returns:
        stats: Dictionary with KS test results
    """
    from scipy.stats import ks_2samp
    from data.preprocessing import calculate_eeg_features
    
    # Extract features
    real_features = []
    for sample in real_samples:
        feats = calculate_eeg_features(sample, sfreq=target_sfreq)
        real_features.append(feats)
    real_features = np.array(real_features)
    
    synthetic_features = []
    for sample in synthetic_samples:
        feats = calculate_eeg_features(sample, sfreq=target_sfreq)
        synthetic_features.append(feats)
    synthetic_features = np.array(synthetic_features)
    
    # Feature names (12 features)
    feature_names = [
        'Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'Gamma Power',
        'Mean Amp', 'Std Amp', 'Kurtosis',
        'Hjorth Activity', 'Hjorth Mobility', 'Hjorth Complexity', 'HFD'
    ]
    
    # KS tests
    ks_results = {}
    n_passed = 0
    
    for i, name in enumerate(feature_names):
        if i < real_features.shape[1]:
            stat, pval = ks_2samp(real_features[:, i], synthetic_features[:, i])
            passed = pval > 0.05  # Pass if p > 0.05 (distributions similar)
            ks_results[name] = {'statistic': float(stat), 'p_value': float(pval), 'passed': passed}
            if passed:
                n_passed += 1
    
    return {
        'feature_tests': ks_results,
        'feature_details': ks_results,  # Alias for backward compatibility
        'n_passed': n_passed,
        'n_total': len(feature_names)
    }


def evaluate_synthetic_quality(
    real_data: dict,
    synthetic_data: dict,
    target_sfreq: int = 256,
    device: str = 'cuda'
) -> dict:
    """
    Comprehensive synthetic quality evaluation.
    
    Args:
        real_data: Dict {class_label: real_samples (N, C, T)}
        synthetic_data: Dict {class_label: synthetic_samples (M, C, T)}
        target_sfreq: Sampling frequency
        device: Device
    
    Returns:
        metrics: Dictionary of quality metrics per class
    """
    quality_metrics = {}
    
    for class_label in sorted(real_data.keys()):
        if class_label not in synthetic_data:
            continue
        
        real_samples = real_data[class_label]
        synthetic_samples = synthetic_data[class_label]
        
        if len(real_samples) == 0 or len(synthetic_samples) == 0:
            continue
        
        print(f"    Evaluating quality for class {class_label}...")
        
        # FID
        fid = calculate_fid_score(real_samples, synthetic_samples, device)
        
        # Diversity
        diversity = calculate_sample_diversity(synthetic_samples)
        
        # Feature statistics
        feature_stats = compute_feature_statistics(real_samples, synthetic_samples, target_sfreq)
        
        quality_metrics[class_label] = {
            'fid': fid,
            'diversity': diversity,
            'feature_match_rate': feature_stats['n_passed'] / feature_stats['n_total'],
            'feature_details': feature_stats['feature_tests'],  # Use feature_tests, not full stats dict
            'n_passed': feature_stats['n_passed'],
            'n_total': feature_stats['n_total']
        }
        
        print(f"      FID: {fid:.2f}")
        print(f"      Diversity: {diversity:.3f}")
        print(f"      Feature Match: {feature_stats['n_passed']}/{feature_stats['n_total']} passed KS test")
    
    return quality_metrics


def plot_quality_metrics(
    quality_metrics: dict,
    save_path: str,
    dpi: int = 300,
    figsize: tuple = (16, 11)
) -> None:
    """
    Plot comprehensive quality metrics visualization.
    
    Creates multi-panel figure showing:
    - FID scores per class (bar chart)
    - Diversity metrics per class (bar chart)
    - Feature match rates per class (bar chart)
    - KS test heatmap (features × classes)
    
    Args:
        quality_metrics: Dictionary from evaluate_synthetic_quality()
            Format: {class_label: {'fid': float, 'diversity': float, 
                                   'feature_match_rate': float, 
                                   'feature_details': dict}}
        save_path: Path to save the plot
        dpi: Resolution for saved figure
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> quality = evaluate_synthetic_quality(real_data, synthetic_data)
        >>> plot_quality_metrics(quality, 'results/quality_metrics.png')
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set aesthetic styling with soft colors
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
    })
    
    # Extract data
    classes = sorted(quality_metrics.keys())
    fid_scores = [quality_metrics[c]['fid'] for c in classes]
    diversity_scores = [quality_metrics[c]['diversity'] for c in classes]
    feature_match_rates = [quality_metrics[c]['feature_match_rate'] * 100 for c in classes]  # Convert to percentage
    
    # Create figure with 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. FID Scores (Lower is better) with soft color
    ax = axes[0, 0]
    bars = ax.bar(range(len(classes)), fid_scores, color='#85C1E9', alpha=0.8, 
                  edgecolor='#BDBDBD', linewidth=1.5)
    ax.set_xlabel('Class', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_ylabel('FID Score', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_title('Fréchet Inception Distance (Lower = Better)', fontsize=14, 
                fontweight='bold', pad=12, color='#2C3E50')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontweight='600')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6, color='#BDBDBD')
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, fid_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='600', color='#2C3E50')
    
    # 2. Diversity Scores (Higher is better) with soft color
    ax = axes[0, 1]
    bars = ax.bar(range(len(classes)), diversity_scores, color='#73C6B6', alpha=0.8, 
                  edgecolor='#BDBDBD', linewidth=1.5)
    ax.set_xlabel('Class', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Diversity Score', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_title('Sample Diversity (Higher = Better)', fontsize=14, 
                fontweight='bold', pad=12, color='#2C3E50')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontweight='600')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6, color='#BDBDBD')
    ax.set_ylim(0, 1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, diversity_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='600', color='#2C3E50')
    
    # 3. Feature Match Rates (Higher is better) with soft color
    ax = axes[1, 0]
    bars = ax.bar(range(len(classes)), feature_match_rates, color='#F8B88B', alpha=0.8, 
                  edgecolor='#BDBDBD', linewidth=1.5)
    ax.set_xlabel('Class', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Feature Match Rate (%)', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_title('Feature Matching (KS Test Pass Rate)', fontsize=14, 
                fontweight='bold', pad=12, color='#2C3E50')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontweight='600')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6, color='#BDBDBD')
    ax.set_ylim(0, 100)
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, feature_match_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='600', color='#2C3E50')
    
    # 4. KS Test Heatmap (Features × Classes)
    ax = axes[1, 1]
    
    # Build KS test matrix
    feature_names = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Peak-to-Peak', 
                     'Line Length', 'Energy', 'Zero Cross', 'Hjorth Mob', 
                     'Hjorth Comp', 'Spec Entropy', 'Samp Entropy']
    
    # Extract KS test p-values for each class and feature
    ks_matrix = []
    for c in classes:
        feature_details = quality_metrics[c]['feature_details']
        p_values = feature_details.get('p_values', [])
        
        # If p_values is available, use it; otherwise mark as unavailable
        if len(p_values) == len(feature_names):
            ks_matrix.append(p_values)
        else:
            ks_matrix.append([0.5] * len(feature_names))  # Neutral value if unavailable
    
    ks_matrix = np.array(ks_matrix).T  # Shape: (n_features, n_classes)
    
    # Create heatmap with soft styling
    im = ax.imshow(ks_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels with soft styling
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight='600')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10, fontweight='600')
    
    ax.set_xlabel('Class', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Feature', fontsize=13, fontweight='600', color='#2C3E50')
    ax.set_title('KS Test P-values (Green = Good Match)', fontsize=14, 
                fontweight='bold', pad=12, color='#2C3E50')
    
    # Add colorbar with soft styling
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P-value', fontsize=11, fontweight='600', color='#2C3E50')
    cbar.ax.tick_params(labelsize=10)
    
    # Add soft grid
    ax.set_xticks(np.arange(len(classes)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(feature_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='#BDBDBD', linestyle='-', linewidth=1.0)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Main title with soft color
    fig.suptitle('Synthetic EEG Quality Assessment', fontsize=18, 
                fontweight='bold', y=0.995, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VISUALIZATION] Quality metrics plot saved to: {save_path}")


def plot_synthetic_vs_real_comparison(
    real_samples: np.ndarray,
    synthetic_samples: np.ndarray,
    class_label: int,
    save_path: str,
    n_samples: int = 3,
    channel_idx: int = 0,
    sfreq: int = 256,
    dpi: int = 300
) -> None:
    """
    Plot side-by-side comparison of real vs synthetic EEG signals.
    
    Creates comprehensive visualization with:
    - Time-domain signals
    - Power spectral density (PSD)
    - Time-frequency representation (spectrogram)
    
    Args:
        real_samples: Real EEG data (N, C, T)
        synthetic_samples: Synthetic EEG data (M, C, T)
        class_label: Class label for title
        save_path: Path to save the plot
        n_samples: Number of sample pairs to show
        channel_idx: EEG channel to visualize
        sfreq: Sampling frequency (Hz)
        dpi: Resolution for saved figure
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> plot_synthetic_vs_real_comparison(
        ...     real_data[0], synthetic_data[0], class_label=0,
        ...     save_path='results/real_vs_synthetic_class0.png'
        ... )
    """
    import os
    from scipy import signal
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set aesthetic styling with soft colors
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.2,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
    })
    
    # Sample random examples
    n_samples = min(n_samples, len(real_samples), len(synthetic_samples))
    real_indices = np.random.choice(len(real_samples), n_samples, replace=False)
    synth_indices = np.random.choice(len(synthetic_samples), n_samples, replace=False)
    
    # Create figure: n_samples rows × 3 columns (time, PSD, spectrogram)
    fig, axes = plt.subplots(n_samples, 3, figsize=(19, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    time_axis = np.arange(real_samples.shape[2]) / sfreq
    
    for i in range(n_samples):
        real_sig = real_samples[real_indices[i], channel_idx, :]
        synth_sig = synthetic_samples[synth_indices[i], channel_idx, :]
        
        # 1. Time-domain comparison with soft colors
        ax = axes[i, 0]
        ax.plot(time_axis, real_sig, label='Real', color='#5DADE2', alpha=0.85, linewidth=2.0)
        ax.plot(time_axis, synth_sig, label='Synthetic', color='#E85D75', alpha=0.85, linewidth=2.0)
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Amplitude (µV)', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_title(f'Sample {i+1}: Time Domain', fontsize=14, fontweight='bold', pad=10, color='#2C3E50')
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, color='#BDBDBD')
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDBDBD')
            spine.set_linewidth(1.2)
        
        # 2. Power Spectral Density with soft colors
        ax = axes[i, 1]
        freqs_real, psd_real = signal.welch(real_sig, fs=sfreq, nperseg=min(256, len(real_sig)))
        freqs_synth, psd_synth = signal.welch(synth_sig, fs=sfreq, nperseg=min(256, len(synth_sig)))
        
        ax.semilogy(freqs_real, psd_real, label='Real', color='#5DADE2', alpha=0.85, linewidth=2.0)
        ax.semilogy(freqs_synth, psd_synth, label='Synthetic', color='#E85D75', alpha=0.85, linewidth=2.0)
        ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Power Spectral Density', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_title(f'Sample {i+1}: Power Spectrum', fontsize=14, fontweight='bold', pad=10, color='#2C3E50')
        ax.set_xlim(0, 70)  # Focus on 0-70 Hz for EEG
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, color='#BDBDBD')
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDBDBD')
            spine.set_linewidth(1.2)
        
        # 3. Spectrogram (time-frequency) with soft styling
        ax = axes[i, 2]
        f, t, Sxx = signal.spectrogram(synth_sig, fs=sfreq, nperseg=64, noverlap=32)
        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='600', color='#2C3E50')
        ax.set_title(f'Sample {i+1}: Spectrogram (Synthetic)', fontsize=14, fontweight='bold', pad=10, color='#2C3E50')
        ax.set_ylim(0, 70)
        cbar = plt.colorbar(pcm, ax=ax, label='Power (dB)')
        cbar.set_label('Power (dB)', fontsize=12, fontweight='600', color='#2C3E50')
        cbar.ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDBDBD')
            spine.set_linewidth(1.2)
    
    # Main title with soft color
    fig.suptitle(f'Real vs Synthetic EEG Comparison - Class {class_label} - Channel {channel_idx}',
                 fontsize=17, fontweight='bold', y=0.998, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[VISUALIZATION] Real vs Synthetic comparison saved to: {save_path}")
