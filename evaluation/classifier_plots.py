"""
Classifier Performance Comparison Plots

Provides visualization functions to compare classifier performance across different
training scenarios: Real-only, Oversampled, and Real+Synthetic.

Features:
- Accuracy comparison bar charts
- F1-score comparison per class
- ROC curves overlay
- Confusion matrix comparison
- Performance improvement metrics

Usage:
    from evaluation.classifier_plots import plot_classifier_comparison
    
    scenarios = {
        'Real Only': real_results,
        'Oversampled': oversampled_results,
        'Real + Synthetic': synthetic_results
    }
    
    plot_classifier_comparison(scenarios, 'results/classifier_comparison.png')
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_classifier_comparison(
    scenarios: Dict[str, Dict],
    save_path: str,
    metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    dpi: int = 150,
    figsize: tuple = (20, 12)
) -> None:
    """
    Plot comprehensive classifier performance comparison across scenarios.
    
    Creates multi-panel figure showing:
    - Overall accuracy comparison
    - F1-score per class comparison
    - Precision/Recall/F1 grouped bars
    - Performance improvement relative to baseline
    
    Args:
        scenarios: Dictionary mapping scenario name to results dict
            Example: {
                'Real Only': {'accuracy': 0.75, 'f1': 0.72, 'precision': 0.73, ...},
                'Oversampled': {'accuracy': 0.78, 'f1': 0.75, ...},
                'Real + Synthetic': {'accuracy': 0.82, 'f1': 0.80, ...}
            }
        save_path: Path to save the plot
        metric_names: List of metric names to plot
        dpi: Resolution for saved figure
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> scenarios = {
        ...     'Real Only': baseline_metrics,
        ...     'Real + Synthetic': synthetic_metrics
        ... }
        >>> plot_classifier_comparison(scenarios, 'results/comparison.png')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    scenario_names = list(scenarios.keys())
    n_scenarios = len(scenario_names)
    
    # Define colors for each scenario
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:n_scenarios]
    
    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Overall Accuracy Comparison
    ax = axes[0, 0]
    accuracies = [scenarios[s].get('accuracy', 0) * 100 for s in scenario_names]
    
    bars = ax.bar(range(n_scenarios), accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Overall Classification Accuracy', fontsize=15, fontweight='bold')
    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(scenario_names, fontsize=11, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. F1-Score Comparison
    ax = axes[0, 1]
    f1_scores = [scenarios[s].get('f1', 0) * 100 for s in scenario_names]
    
    bars = ax.bar(range(n_scenarios), f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('F1-Score (%)', fontsize=13)
    ax.set_title('Macro-Averaged F1-Score', fontsize=15, fontweight='bold')
    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(scenario_names, fontsize=11, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Grouped Bar Chart (Precision, Recall, F1)
    ax = axes[1, 0]
    
    metrics_to_compare = ['precision', 'recall', 'f1']
    x = np.arange(len(scenario_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_compare):
        values = [scenarios[s].get(metric, 0) * 100 for s in scenario_names]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=11, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Performance Improvement (relative to baseline)
    ax = axes[1, 1]
    
    if len(scenario_names) > 1:
        baseline_name = scenario_names[0]
        baseline_acc = scenarios[baseline_name].get('accuracy', 0)
        baseline_f1 = scenarios[baseline_name].get('f1', 0)
        
        improvement_scenarios = scenario_names[1:]
        acc_improvements = []
        f1_improvements = []
        
        for s in improvement_scenarios:
            acc_imp = ((scenarios[s].get('accuracy', 0) - baseline_acc) / baseline_acc) * 100
            f1_imp = ((scenarios[s].get('f1', 0) - baseline_f1) / baseline_f1) * 100
            acc_improvements.append(acc_imp)
            f1_improvements.append(f1_imp)
        
        x = np.arange(len(improvement_scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, acc_improvements, width, label='Accuracy', color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, f1_improvements, width, label='F1-Score', color='coral', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Improvement (%)', fontsize=13)
        ax.set_title(f'Performance Improvement vs {baseline_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(improvement_scenarios, fontsize=11, rotation=15, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.legend(loc='best', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                label_y = height + 0.5 if height > 0 else height - 0.5
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{height:.1f}%',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Need at least 2 scenarios\nfor improvement comparison',
                ha='center', va='center', fontsize=13, transform=ax.transAxes)
        ax.axis('off')
    
    # Main title
    fig.suptitle('Classifier Performance Comparison Across Training Scenarios',
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Classifier comparison saved to: {save_path}")


def plot_per_class_f1_comparison(
    scenarios: Dict[str, Dict],
    class_names: List[str],
    save_path: str,
    dpi: int = 150,
    figsize: tuple = (14, 8)
) -> None:
    """
    Plot per-class F1-score comparison across scenarios.
    
    Args:
        scenarios: Dictionary mapping scenario name to results dict
            Results dict should have 'f1_per_class' key with list of F1 scores
        class_names: List of class names for x-axis labels
        save_path: Path to save the plot
        dpi: Resolution
        figsize: Figure size
    
    Returns:
        None (saves plot to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    scenario_names = list(scenarios.keys())
    n_classes = len(class_names)
    n_scenarios = len(scenario_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_classes)
    width = 0.8 / n_scenarios
    
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:n_scenarios]
    
    for i, scenario_name in enumerate(scenario_names):
        f1_per_class = scenarios[scenario_name].get('f1_per_class', [0] * n_classes)
        f1_per_class = np.array(f1_per_class) * 100  # Convert to percentage
        
        offset = (i - n_scenarios/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_per_class, width, label=scenario_name,
                      color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, f1_per_class):
            height = bar.get_height()
            if height > 5:  # Only show label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('F1-Score (%)', fontsize=13)
    ax.set_title('Per-Class F1-Score Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Per-class F1 comparison saved to: {save_path}")


def plot_confusion_matrices_comparison(
    scenarios: Dict[str, Dict],
    class_names: List[str],
    save_path: str,
    dpi: int = 150,
    figsize: tuple = (18, 6)
) -> None:
    """
    Plot confusion matrices side-by-side for multiple scenarios.
    
    Args:
        scenarios: Dictionary mapping scenario name to results dict
            Results dict should have 'confusion_matrix' key with 2D array
        class_names: List of class names for axis labels
        save_path: Path to save the plot
        dpi: Resolution
        figsize: Figure size
    
    Returns:
        None (saves plot to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    scenario_names = list(scenarios.keys())
    n_scenarios = len(scenario_names)
    
    fig, axes = plt.subplots(1, n_scenarios, figsize=figsize)
    if n_scenarios == 1:
        axes = [axes]
    
    for ax, scenario_name in zip(axes, scenario_names):
        cm = scenarios[scenario_name].get('confusion_matrix', np.zeros((len(class_names), len(class_names))))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=True, square=True)
        
        ax.set_title(f'{scenario_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    fig.suptitle('Confusion Matrix Comparison (Normalized)', fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Confusion matrices comparison saved to: {save_path}")
