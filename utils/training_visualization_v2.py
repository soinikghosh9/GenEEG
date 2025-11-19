"""
Publication-Quality Training Visualization Module

Provides comprehensive visualization functions for training progress of VAE and LDM models.
Optimized for publication with high DPI, clear fonts, and print-friendly colors.

Features:
- Unified loss plots with ALL components normalized and visible
- Total loss highlighted with thick black line for clarity
- Publication-ready: DPI=300, optimized fonts, grayscale-compatible colors
- Automatic handling of missing or zero losses
- Saves plots as high-resolution PNG files

Usage:
    from utils.training_visualization_v2 import plot_vae_training_losses, plot_ldm_training_losses
    
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


# Publication-quality color palette (works in grayscale)
COLORS = {
    'total': '#000000',      # Black - total loss
    'primary': '#E74C3C',    # Red
    'secondary': '#3498DB',  # Blue  
    'tertiary': '#2ECC71',   # Green
    'quaternary': '#F39C12', # Orange
    'quinary': '#9B59B6',    # Purple
    'senary': '#1ABC9C',     # Teal
    'septenary': '#E67E22',  # Dark Orange
    'octonary': '#34495E',   # Dark Gray
    'nonary': '#16A085',     # Dark Teal
    'denary': '#8E44AD',     # Dark Purple
    'undenary': '#C0392B',   # Dark Red
    'duodenary': '#27AE60',  # Dark Green
    'terdenary': '#2980B9',  # Dark Blue
}

LINE_STYLES = ['-', '--', '-.', ':']


def plot_vae_training_losses(
    metrics: Dict[str, List[float]],
    save_path: str,
    dpi: int = 300,
    figsize: tuple = (14, 9)
) -> None:
    """
    Plot comprehensive VAE training losses with ALL components normalized on same axes.
    
    Creates a publication-quality figure showing ALL loss components normalized to [0,1] 
    on same plot, with TOTAL LOSS highlighted with thicker black line.
    
    Args:
        metrics: Dictionary with training metrics from train_vae()
            Expected keys: 'total_loss', 'recon_loss', 'kl_loss', 
                          'contrastive_loss', 'feature_recon_loss', 
                          'high_freq_loss', 'stft_loss', 'sharpness_loss', etc.
        save_path: Path to save the plot (e.g., 'checkpoints/vae_losses.png')
        dpi: Resolution for saved figure (default: 300 for publication)
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> plot_vae_training_losses(
        ...     metrics={'total_loss': [1.2, 1.0, 0.8], 'recon_loss': [0.9, 0.7, 0.6]},
        ...     save_path='results/vae_training.png'
        ... )
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Define all possible VAE loss components with display names
    loss_components = {
        'total_loss': 'Total Loss',
        'recon_loss': 'Reconstruction',
        'kl_loss': 'KL Divergence',
        'feature_recon_loss': 'Feature Reconstruction',
        'sharpness_loss': 'Sharpness',
        'contrastive_loss': 'Contrastive',
        'stft_loss': 'STFT',
        'high_freq_loss': 'High Frequency',
        'low_freq_loss': 'Low Frequency',
        'dwt_loss': 'DWT',
        'connectivity_loss': 'Connectivity',
        'minority_focus_loss': 'Minority Focus',
        'edge_artifact_loss': 'Edge Artifact',
        'freq_band_loss': 'Frequency Band',
    }
    
    # Filter available losses (non-empty, non-zero)
    available_losses = {}
    for key, display_name in loss_components.items():
        if key in metrics and len(metrics[key]) > 0:
            values = np.array(metrics[key])
            if not np.all(values == 0) and not np.all(np.isnan(values)):
                available_losses[key] = display_name
    
    if len(available_losses) == 0:
        print("[WARN] No valid loss components found for VAE plotting")
        return
    
    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })
    
    # Normalize all losses to [0, 1] and collect ranges
    normalized_losses = {}
    value_ranges = {}
    
    for key in available_losses.keys():
        values = np.array(metrics[key])
        values_min = np.min(values)
        values_max = np.max(values)
        
        if values_max - values_min > 1e-10:
            normalized = (values - values_min) / (values_max - values_min)
        else:
            normalized = np.zeros_like(values)
        
        normalized_losses[key] = normalized
        value_ranges[key] = (values_min, values_max)
    
    # Assign colors and line styles
    color_keys = list(COLORS.keys())
    
    # Plot each loss component
    idx = 0
    for key, display_name in available_losses.items():
        epochs = np.arange(1, len(normalized_losses[key]) + 1)
        
        if key == 'total_loss':
            # Total loss: thick black line, most prominent
            ax.plot(
                epochs, normalized_losses[key],
                color=COLORS['total'],
                linewidth=4.0,
                alpha=1.0,
                zorder=100,
                label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
            )
        else:
            # Other components: colored with varying styles
            color_idx = (idx % (len(color_keys) - 1)) + 1  # Skip 'total'
            color = COLORS[color_keys[color_idx]]
            linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
            
            ax.plot(
                epochs, normalized_losses[key],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.8,
                zorder=10,
                label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
            )
            idx += 1
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Loss [0-1]', fontsize=14, fontweight='bold')
    ax.set_title('VAE Training Losses (All Components Normalized)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(1, len(normalized_losses['total_loss']))
    ax.set_ylim(-0.05, 1.05)
    
    # Legend outside plot area
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1
    )
    
    # Add annotation
    ax.text(
        0.02, 0.98,
        'Total Loss = Thick Black Line\nOther components shown with original value ranges in legend',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[VISUALIZATION] VAE training losses plot saved to: {save_path}")


def plot_ldm_training_losses(
    history: Dict[str, List[float]],
    save_path: str,
    dpi: int = 300,
    figsize: tuple = (14, 9)
) -> None:
    """
    Plot comprehensive LDM training losses with ALL components normalized on same axes.
    
    Creates a publication-quality figure showing ALL loss components normalized to [0,1] 
    on same plot, with TOTAL LOSS highlighted with thicker black line.
    
    Args:
        history: Dictionary with training history from train_ldm()
            Expected keys: 'total', 'denoise', 'moment', 'aux', 
                          'low_freq', 'seizure', 'val_denoise', etc.
        save_path: Path to save the plot (e.g., 'checkpoints/ldm_losses.png')
        dpi: Resolution for saved figure (default: 300 for publication)
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    
    Example:
        >>> plot_ldm_training_losses(
        ...     history={'total': [1.5, 1.2, 1.0], 'denoise': [1.0, 0.8, 0.7]},
        ...     save_path='results/ldm_training.png'
        ... )
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Define all possible LDM loss components with display names
    loss_components = {
        'total': 'Total Loss',
        'denoise': 'Denoising',
        'moment': 'Moment Matching',
        'aux': 'Auxiliary',
        'low_freq': 'Low Frequency',
        'seizure': 'Seizure Focus',
        'val_denoise': 'Validation Denoising',
    }
    
    # Filter available losses (non-empty, non-zero)
    available_losses = {}
    for key, display_name in loss_components.items():
        if key in history and len(history[key]) > 0:
            values = np.array(history[key])
            if not np.all(values == 0) and not np.all(np.isnan(values)):
                available_losses[key] = display_name
    
    if len(available_losses) == 0:
        print("[WARN] No valid loss components found for LDM plotting")
        return
    
    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })
    
    # Normalize all losses to [0, 1] and collect ranges
    normalized_losses = {}
    value_ranges = {}
    
    for key in available_losses.keys():
        values = np.array(history[key])
        values_min = np.min(values)
        values_max = np.max(values)
        
        if values_max - values_min > 1e-10:
            normalized = (values - values_min) / (values_max - values_min)
        else:
            normalized = np.zeros_like(values)
        
        normalized_losses[key] = normalized
        value_ranges[key] = (values_min, values_max)
    
    # Assign colors and line styles
    color_keys = list(COLORS.keys())
    
    # Plot each loss component
    idx = 0
    for key, display_name in available_losses.items():
        epochs = np.arange(1, len(normalized_losses[key]) + 1)
        
        if key == 'total':
            # Total loss: thick black line, most prominent
            ax.plot(
                epochs, normalized_losses[key],
                color=COLORS['total'],
                linewidth=4.0,
                alpha=1.0,
                zorder=100,
                label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
            )
        else:
            # Other components: colored with varying styles
            color_idx = (idx % (len(color_keys) - 1)) + 1  # Skip 'total'
            color = COLORS[color_keys[color_idx]]
            linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
            
            ax.plot(
                epochs, normalized_losses[key],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.8,
                zorder=10,
                label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
            )
            idx += 1
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Loss [0-1]', fontsize=14, fontweight='bold')
    ax.set_title('LDM Training Losses (All Components Normalized)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(1, len(normalized_losses['total']))
    ax.set_ylim(-0.05, 1.05)
    
    # Legend outside plot area
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1
    )
    
    # Add annotation
    ax.text(
        0.02, 0.98,
        'Total Loss = Thick Black Line\nOther components shown with original value ranges in legend',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[VISUALIZATION] LDM training losses plot saved to: {save_path}")


def plot_combined_training_summary(
    vae_metrics: Dict[str, List[float]],
    ldm_history: Dict[str, List[float]],
    save_path: str,
    dpi: int = 300,
    figsize: tuple = (18, 9)
) -> None:
    """
    Plot side-by-side comparison of VAE and LDM training progress.
    
    Creates a 2-column publication-quality figure showing VAE and LDM training losses
    side by side for direct comparison.
    
    Args:
        vae_metrics: VAE training metrics dictionary
        ldm_history: LDM training history dictionary
        save_path: Path to save the combined plot
        dpi: Resolution for saved figure (default: 300 for publication)
        figsize: Figure size in inches (width, height)
    
    Returns:
        None (saves plot to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
        'figure.titlesize': 17,
        'lines.linewidth': 2.0,
    })
    
    # Plot VAE total loss on left
    if 'total_loss' in vae_metrics and len(vae_metrics['total_loss']) > 0:
        epochs_vae = np.arange(1, len(vae_metrics['total_loss']) + 1)
        ax1.plot(epochs_vae, vae_metrics['total_loss'], 
                color=COLORS['total'], linewidth=3.0, label='Total Loss')
        
        # Add other VAE components if available
        for key in ['recon_loss', 'kl_loss', 'contrastive_loss']:
            if key in vae_metrics and len(vae_metrics[key]) > 0:
                ax1.plot(epochs_vae, vae_metrics[key], 
                        linewidth=1.5, alpha=0.7, label=key.replace('_', ' ').title())
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss Value', fontweight='bold')
    ax1.set_title('VAE Training Progress', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best')
    
    # Plot LDM total loss on right
    if 'total' in ldm_history and len(ldm_history['total']) > 0:
        epochs_ldm = np.arange(1, len(ldm_history['total']) + 1)
        ax2.plot(epochs_ldm, ldm_history['total'], 
                color=COLORS['total'], linewidth=3.0, label='Total Loss')
        
        # Add other LDM components if available
        for key in ['denoise', 'moment', 'aux']:
            if key in ldm_history and len(ldm_history[key]) > 0:
                ax2.plot(epochs_ldm, ldm_history[key], 
                        linewidth=1.5, alpha=0.7, label=key.replace('_', ' ').title())
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss Value', fontweight='bold')
    ax2.set_title('LDM Training Progress', fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='best')
    
    # Overall title
    fig.suptitle('VAE and LDM Training Comparison', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[VISUALIZATION] Combined training summary saved to: {save_path}")
