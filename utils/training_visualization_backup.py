"""
Training Visualization Module

Provides comprehensive visualization functions for training progress of VAE and LDM models.
Generates publication-quality plots showing loss evolution across epochs.

Features:
- Multi-panel loss plots for all loss components
- Separate plots for VAE (11 loss components) and LDM (4 loss components)
- Automatic subplot layout
- Saves plots as high-resolution PNG files
- Handles missing or zero losses gracefully

Usage:
    from utils.training_visualization import plot_vae_training_losses, plot_ldm_training_losses
    
    # Plot VAE losses
    plot_vae_training_losses(
        metrics=vae_metrics,
        save_path='checkpoints/vae_losses.png'
    )
    
    # Plot LDM losses
    plot_ldm_training_losses(
        history=ldm_history,
        save_path='checkpoints/ldm_losses.png'
    )
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional


def plot_vae_training_losses(
    metrics: Dict[str, List[float]],
    save_path: str,
    dpi: int = 150,
    figsize: tuple = (16, 10)
) -> None:
    """
    Plot comprehensive VAE training losses with ALL components normalized on same axes.
    
    Creates a figure showing ALL loss components normalized to [0,1] on same plot,
    with TOTAL LOSS highlighted with thicker line.
    
    Args:
        metrics: Dictionary with training metrics from train_vae()
            Expected keys: 'total_loss', 'recon_loss', 'kl_loss', 
                          'contrastive_loss', 'feature_recon_loss', 
                          'high_freq_loss', 'stft_loss', 'sharpness_loss', etc.
        save_path: Path to save the plot (e.g., 'checkpoints/vae_losses.png')
        dpi: Resolution for saved figure (default: 150)
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> metrics = train_vae(vae, feat_head, loader, 100, 3e-4, device, 'vae.pth')
        >>> plot_vae_training_losses(metrics, 'checkpoints/vae_training_losses.png')
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract metrics
    total_epochs = len(metrics.get('total_loss', []))
    if total_epochs == 0:
        print(f"[WARNING] No training metrics to plot for VAE")
        return
    
    epochs = range(1, total_epochs + 1)
    
    # Define all possible loss components with colors and line styles
    # TOTAL LOSS is first and will be highlighted
    all_components = [
        ('total_loss', 'Total Loss', '#000000', '-', 3.5),  # Black, thick
        ('recon_loss', 'Reconstruction (L1)', '#FF6B6B', '-', 2.0),  # Red
        ('kl_loss', 'KL Divergence', '#4ECDC4', '-', 2.0),  # Teal
        ('feature_recon_loss', 'Feature Recon', '#95E1D3', '--', 2.0),  # Light green
        ('sharpness_loss', 'Sharpness', '#F38181', '--', 2.0),  # Pink
        ('contrastive_loss', 'Contrastive', '#AA96DA', ':', 1.8),  # Purple
        ('stft_loss', 'Multi-Scale STFT', '#FCBAD3', ':', 1.8),  # Light pink
        ('high_freq_loss', 'High-Freq', '#FFFAAA', ':', 1.8),  # Yellow
        ('low_freq_loss', 'Low-Freq', '#A8D8EA', ':', 1.8),  # Light blue
        ('dwt_loss', 'DWT', '#FFA07A', '-.', 1.8),  # Light salmon
        ('connectivity_loss', 'Connectivity', '#98D8C8', '-.', 1.8),  # Mint
        ('minority_focus_loss', 'Minority Focus', '#F7DC6F', '-.', 1.8),  # Gold
        ('edge_artifact_loss', 'Edge Artifact', '#BB8FCE', '-.', 1.8),  # Lavender
        ('freq_band_loss', 'Freq Band', '#85C1E2', '-.', 1.8)  # Sky blue
    ]
    
    # Filter active components (non-zero, present in metrics)
    active_components = []
    for key, label, color, style, width in all_components:
        if key in metrics and len(metrics[key]) > 0:
            if np.max(np.abs(metrics[key])) > 1e-10:
                active_components.append((key, label, color, style, width))
    
    if not active_components:
        print(f"[WARNING] No active loss components to plot for VAE")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot ALL normalized losses on same axes
    legend_entries = []
    for key, label, color, style, width in active_components:
        values = np.array(metrics[key])
        val_min, val_max = values.min(), values.max()
        val_range = val_max - val_min
        
        # Min-max normalization
        if val_range > 1e-10:
            normalized = (values - val_min) / val_range
        else:
            normalized = np.zeros_like(values)
        
        # Special handling for total loss (highlighted)
        alpha = 1.0 if key == 'total_loss' else 0.75
        zorder = 100 if key == 'total_loss' else 10
        
        # Plot
        line, = ax.plot(epochs, normalized, 
                       color=color, linestyle=style, linewidth=width, 
                       alpha=alpha, zorder=zorder)
        
        # Legend entry with original range
        if val_range > 0.01:
            legend_text = f'{label} [{val_min:.3f} – {val_max:.3f}]'
        else:
            legend_text = f'{label} [{val_min:.2e} – {val_max:.2e}]'
        legend_entries.append((line, legend_text))
    
    # Configure plot
    ax.set_title('VAE Training: All Loss Components (Normalized Scale)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Loss [0 → 1]', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(1, total_epochs)
    
    # Legend with original value ranges
    lines, labels = zip(*legend_entries)
    ax.legend(lines, labels, loc='upper right', fontsize=9, 
             framealpha=0.95, ncol=2, columnspacing=0.8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add annotation for total loss
    if 'total_loss' in metrics:
        total_values = np.array(metrics['total_loss'])
        final_total = total_values[-1]
        best_total = total_values.min()
        best_epoch = np.argmin(total_values) + 1
        
        textstr = f'Total Loss:\n  Best: {best_total:.4f} @ Epoch {best_epoch}\n  Final: {final_total:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZATION] VAE training losses saved to: {save_path}")


def plot_ldm_training_losses(
    history: Dict[str, List[float]],
    save_path: str,
    dpi: int = 150,
    figsize: tuple = (16, 10)
) -> None:
    """
    Plot comprehensive LDM training losses with ALL components normalized on same axes.
    
    Creates a figure showing ALL loss components normalized to [0,1] on same plot,
    with TOTAL LOSS highlighted with thicker line.
    
    Args:
        history: Dictionary with training metrics from train_latent_diffusion_unet()
            Expected keys: 'total', 'denoise', 'moment', 'aux', 'lr', 'val_denoise' (optional)
        save_path: Path to save the plot (e.g., 'checkpoints/ldm_losses.png')
        dpi: Resolution for saved figure (default: 150)
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> ldm, hist = train_latent_diffusion_unet(unet, vae, loader, epochs=150)
        >>> plot_ldm_training_losses(hist, 'checkpoints/ldm_training_losses.png')
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract metrics
    total_epochs = len(history.get('total', []))
    if total_epochs == 0:
        print(f"[WARNING] No training metrics to plot for LDM")
        return
    
    epochs = range(1, total_epochs + 1)
    
    # Define all possible LDM loss components with colors and line styles
    # TOTAL LOSS is first and will be highlighted
    all_components = [
        ('total', 'Total Loss', '#000000', '-', 3.5),  # Black, thick
        ('denoise', 'Denoise Loss (v-prediction)', '#FF6B6B', '-', 2.5),  # Red
        ('moment', 'Moment Loss (μ/σ stability)', '#4ECDC4', '--', 2.0),  # Teal
        ('aux', 'Auxiliary Loss (Low-Freq + Seizure)', '#95E1D3', '--', 2.0),  # Light green
        ('low_freq', 'Low-Freq Preservation', '#FCBAD3', ':', 1.8),  # Light pink
        ('seizure', 'Seizure Fidelity', '#FFA07A', ':', 1.8),  # Light salmon
        ('val_denoise', 'Validation Denoise', '#AA96DA', '-.', 2.0)  # Purple
    ]
    
    # Filter active components (non-zero, present in history)
    active_components = []
    for key, label, color, style, width in all_components:
        if key in history and len(history[key]) > 0:
            if np.max(np.abs(history[key])) > 1e-10:
                active_components.append((key, label, color, style, width))
    
    if not active_components:
        print(f"[WARNING] No active loss components to plot for LDM")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot ALL normalized losses on same axes
    legend_entries = []
    for key, label, color, style, width in active_components:
        values = np.array(history[key])
        val_min, val_max = values.min(), values.max()
        val_range = val_max - val_min
        
        # Min-max normalization
        if val_range > 1e-10:
            normalized = (values - val_min) / val_range
        else:
            normalized = np.zeros_like(values)
        
        # Special handling for total loss (highlighted)
        alpha = 1.0 if key == 'total' else 0.75
        zorder = 100 if key == 'total' else 10
        
        # Plot
        line, = ax.plot(epochs, normalized, 
                       color=color, linestyle=style, linewidth=width, 
                       alpha=alpha, zorder=zorder)
        
        # Legend entry with original range
        if val_range > 0.01:
            legend_text = f'{label} [{val_min:.3f} – {val_max:.3f}]'
        else:
            legend_text = f'{label} [{val_min:.2e} – {val_max:.2e}]'
        legend_entries.append((line, legend_text))
    
    # Configure plot
    ax.set_title('LDM Training: All Loss Components (Normalized Scale)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Loss [0 → 1]', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(1, total_epochs)
    
    # Legend with original value ranges
    lines, labels = zip(*legend_entries)
    ax.legend(lines, labels, loc='upper right', fontsize=10, 
             framealpha=0.95, ncol=1, columnspacing=0.8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add annotation for total loss
    if 'total' in history:
        total_values = np.array(history['total'])
        final_total = total_values[-1]
        best_total = total_values.min()
        best_epoch = np.argmin(total_values) + 1
        
        textstr = f'Total Loss:\n  Best: {best_total:.4f} @ Epoch {best_epoch}\n  Final: {final_total:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZATION] LDM training losses saved to: {save_path}")


def plot_combined_training_summary(
    vae_metrics: Dict[str, List[float]],
    ldm_history: Dict[str, List[float]],
    save_path: str,
    dpi: int = 150,
    figsize: tuple = (20, 8)
) -> None:
    """
    Plot combined summary showing VAE and LDM training progress side-by-side.
    
    Creates a 2-column figure:
    - Left: VAE total loss
    - Right: LDM total loss
    
    Useful for manuscript figures showing complete training pipeline.
    
    Args:
        vae_metrics: VAE training metrics from train_vae()
        ldm_history: LDM training history from train_latent_diffusion_unet()
        save_path: Path to save the plot
        dpi: Resolution for saved figure
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot VAE total loss
    vae_epochs = range(1, len(vae_metrics['total_loss']) + 1)
    ax1.plot(vae_epochs, vae_metrics['total_loss'], color='blue', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Total Loss', fontsize=14)
    ax1.set_title('VAE Training', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, len(vae_epochs))
    
    best_vae = np.min(vae_metrics['total_loss'])
    best_vae_epoch = np.argmin(vae_metrics['total_loss']) + 1
    ax1.axhline(y=best_vae, color='blue', linestyle=':', alpha=0.5,
                label=f'Best: {best_vae:.4f} @ Epoch {best_vae_epoch}')
    ax1.legend(loc='best', fontsize=12)
    
    # Plot LDM total loss
    ldm_epochs = range(1, len(ldm_history['total']) + 1)
    ax2.plot(ldm_epochs, ldm_history['total'], color='green', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Total Loss', fontsize=14)
    ax2.set_title('Latent Diffusion Model Training', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1, len(ldm_epochs))
    
    best_ldm = np.min(ldm_history['total'])
    best_ldm_epoch = np.argmin(ldm_history['total']) + 1
    ax2.axhline(y=best_ldm, color='green', linestyle=':', alpha=0.5,
                label=f'Best: {best_ldm:.6f} @ Epoch {best_ldm_epoch}')
    ax2.legend(loc='best', fontsize=12)
    
    # Main title
    fig.suptitle('Complete Training Pipeline: VAE + Latent Diffusion Model', 
                 fontsize=18, fontweight='bold', y=1.00)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Combined training summary saved to: {save_path}")
