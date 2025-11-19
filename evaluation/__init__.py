"""
Evaluation Package

This package contains evaluation metrics and plotting functions for
seizure detection model performance assessment.

Components:
    - metrics: Confusion matrix and ROC curve plotting
    - synthetic quality: FID, diversity, and feature statistics for synthetic EEG
    - visualization: Quality metrics plots, real vs synthetic comparisons
    - classifier_plots: Performance comparison across training scenarios
"""

from .metrics import (
    plot_confusion_matrix, 
    plot_roc_auc_curves,
    calculate_fid_score,
    calculate_sample_diversity,
    compute_feature_statistics,
    evaluate_synthetic_quality,
    plot_quality_metrics,
    plot_synthetic_vs_real_comparison
)

from .classifier_plots import (
    plot_classifier_comparison,
    plot_per_class_f1_comparison,
    plot_confusion_matrices_comparison
)

__all__ = [
    'plot_confusion_matrix',
    'plot_roc_auc_curves',
    'calculate_fid_score',
    'calculate_sample_diversity',
    'compute_feature_statistics',
    'evaluate_synthetic_quality',
    'plot_quality_metrics',
    'plot_synthetic_vs_real_comparison',
    'plot_classifier_comparison',
    'plot_per_class_f1_comparison',
    'plot_confusion_matrices_comparison',
]
