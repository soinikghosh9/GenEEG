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


# Aesthetic color palette - soft, dimmed, scientific colors (grayscale-compatible)
# Carefully selected for visual distinction and print compatibility
COLORS = {
    'total': '#2C3E50',      # Dark slate - total loss (prominent but not harsh black)
    
    # Soft warm tones
    'color1': '#E85D75',     # Soft coral red
    'color2': '#F39C6B',     # Warm peach
    'color3': '#FFB74D',     # Soft amber
    'color4': '#FFA726',     # Light orange
    
    # Soft cool tones  
    'color5': '#5DADE2',     # Sky blue
    'color6': '#48C9B0',     # Turquoise
    'color7': '#7FB3D5',     # Powder blue
    'color8': '#76D7C4',     # Mint
    
    # Soft purple/pink tones
    'color9': '#BB8FCE',     # Lavender purple
    'color10': '#D7BDE2',    # Light purple
    'color11': '#F8B4D9',    # Rose pink
    'color12': '#AED6F1',    # Baby blue
    
    # Additional soft tones
    'color13': '#A9DFBF',    # Pale green
    'color14': '#FAD7A0',    # Cream yellow
}

LINE_STYLES = ['-', '--', '-.', ':']

# Create comprehensive color list for many components
COLOR_LIST = [
    '#E85D75', '#5DADE2', '#48C9B0', '#F39C6B', 
    '#BB8FCE', '#FFB74D', '#7FB3D5', '#A9DFBF',
    '#F8B4D9', '#76D7C4', '#FFA726', '#D7BDE2',
    '#AED6F1', '#FAD7A0'
]


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
        'ewc_loss': 'EWC (Continual Learning)',
    }
    
    # Filter available losses (non-empty, non-zero, meaningful values)
    available_losses = {}
    for key, display_name in loss_components.items():
        if key in metrics and len(metrics[key]) > 0:
            values = np.array(metrics[key])
            # Skip if all zeros, all NaN, or negligible values (< 1e-8)
            if not np.all(values == 0) and not np.all(np.isnan(values)):
                max_val = np.max(np.abs(values[np.isfinite(values)]))
                if max_val > 1e-8:  # Only include if has meaningful magnitude
                    available_losses[key] = display_name
    
    if len(available_losses) == 0:
        print("[WARN] No valid loss components found for VAE plotting")
        return
    
    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set publication-quality style with aesthetic enhancements
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',  # Modern, clean look
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9.5,
        'figure.titlesize': 18,
        'lines.linewidth': 2.2,
        'lines.markersize': 5,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.facecolor': '#FAFAFA',  # Very light gray background
        'figure.facecolor': 'white',
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
    
    # Plot TOTAL LOSS FIRST with semi-transparent style (background layer)
    if 'total_loss' in available_losses:
        epochs = np.arange(1, len(normalized_losses['total_loss']) + 1)
        # Plot total loss with transparency so other components remain visible
        ax.plot(
            epochs, normalized_losses['total_loss'],
            color=COLORS['total'],
            linewidth=3.5,
            alpha=0.6,  # Semi-transparent so components underneath are visible
            zorder=5,  # Lower z-order so it's behind other components
            label=f"Total Loss [{value_ranges['total_loss'][0]:.4f} – {value_ranges['total_loss'][1]:.4f}]",
            linestyle='--',
            dashes=(5, 2)  # Distinctive dash pattern
        )
    
    # Now plot all other loss components on top (higher z-order) with full opacity
    idx = 0
    for key, display_name in available_losses.items():
        if key == 'total_loss':
            continue  # Already plotted
            
        epochs = np.arange(1, len(normalized_losses[key]) + 1)
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        ax.plot(
            epochs, normalized_losses[key],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            alpha=0.90,  # High opacity so components are clearly visible
            zorder=10 + idx,  # Higher z-order to be on top of total loss
            label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
        )
        idx += 1
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Normalized Loss [0-1]', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_title('VAE Training Losses (All Components Normalized)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    ax.set_xlim(1, len(normalized_losses['total_loss']))
    ax.set_ylim(-0.05, 1.05)
    
    # Subtle spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Legend outside plot area with soft styling
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=1,
        edgecolor='#BDBDBD',
        facecolor='white',
        framealpha=0.95
    )
    legend.get_frame().set_linewidth(1.0)
    
    # REMOVED: Annotation box removed per user request for cleaner plots
    
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
        'ewc': 'EWC (Continual Learning)',
        'low_freq': 'Low Frequency',
        'seizure': 'Seizure Focus',
        'val_denoise': 'Validation Denoising',
    }
    
    # Filter available losses (non-empty, non-zero, meaningful values)
    available_losses = {}
    for key, display_name in loss_components.items():
        if key in history and len(history[key]) > 0:
            values = np.array(history[key])
            # Skip if all zeros, all NaN, or negligible values (< 1e-8)
            if not np.all(values == 0) and not np.all(np.isnan(values)):
                max_val = np.max(np.abs(values[np.isfinite(values)]))
                if max_val > 1e-8:  # Only include if has meaningful magnitude
                    available_losses[key] = display_name
    
    if len(available_losses) == 0:
        print("[WARN] No valid loss components found for LDM plotting")
        return
    
    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set publication-quality style with aesthetic enhancements
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9.5,
        'figure.titlesize': 18,
        'lines.linewidth': 2.2,
        'lines.markersize': 5,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
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
    
    # Plot TOTAL LOSS FIRST with semi-transparent style (background layer)
    if 'total' in available_losses:
        epochs = np.arange(1, len(normalized_losses['total']) + 1)
        # Plot total loss with transparency so other components remain visible
        ax.plot(
            epochs, normalized_losses['total'],
            color=COLORS['total'],
            linewidth=3.5,
            alpha=0.6,  # Semi-transparent so components underneath are visible
            zorder=5,  # Lower z-order so it's behind other components
            label=f"Total Loss [{value_ranges['total'][0]:.4f} – {value_ranges['total'][1]:.4f}]",
            linestyle='--',
            dashes=(5, 2)  # Distinctive dash pattern
        )
    
    # Now plot all other loss components on top (higher z-order) with full opacity
    idx = 0
    for key, display_name in available_losses.items():
        if key == 'total':
            continue  # Already plotted
            
        epochs = np.arange(1, len(normalized_losses[key]) + 1)
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        ax.plot(
            epochs, normalized_losses[key],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            alpha=0.90,  # High opacity so components are clearly visible
            zorder=10 + idx,  # Higher z-order to be on top of total loss
            label=f'{display_name} [{value_ranges[key][0]:.4f} – {value_ranges[key][1]:.4f}]'
        )
        idx += 1
    
    # Styling with aesthetic enhancements
    ax.set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Normalized Loss [0-1]', fontsize=14, fontweight='600', color='#2C3E50')
    ax.set_title('LDM Training Losses (All Components Normalized)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    ax.set_xlim(1, len(normalized_losses['total']))
    ax.set_ylim(-0.05, 1.05)
    
    # Subtle spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Legend outside plot area with soft styling
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=1,
        edgecolor='#BDBDBD',
        facecolor='white',
        framealpha=0.95
    )
    legend.get_frame().set_linewidth(1.0)
    
    # REMOVED: Annotation box removed per user request for cleaner plots
    
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
    
    # Set aesthetic styling for soft, modern appearance
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
        'figure.titlesize': 17,
        'lines.linewidth': 2.0,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.6,
    })
    
    # Plot VAE total loss on left
    if 'total_loss' in vae_metrics and len(vae_metrics['total_loss']) > 0:
        epochs_vae = np.arange(1, len(vae_metrics['total_loss']) + 1)
        ax1.plot(epochs_vae, vae_metrics['total_loss'], 
                color=COLORS['total'], linewidth=3.5, alpha=0.95, 
                label='Total Loss', zorder=100)
        
        # Add other VAE components if available with soft colors
        component_idx = 0
        for key in ['recon_loss', 'kl_loss', 'contrastive_loss']:
            if key in vae_metrics and len(vae_metrics[key]) > 0:
                color = COLOR_LIST[component_idx % len(COLOR_LIST)]
                ax1.plot(epochs_vae, vae_metrics[key], 
                        color=color, linewidth=2.2, alpha=0.75, 
                        label=key.replace('_', ' ').title(), zorder=10)
                component_idx += 1
    
    ax1.set_xlabel('Epoch', fontweight='600', color='#2C3E50')
    ax1.set_ylabel('Loss Value', fontweight='600', color='#2C3E50')
    ax1.set_title('VAE Training Progress', fontweight='bold', pad=15, color='#2C3E50')
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    ax1.legend(loc='best', framealpha=0.95, edgecolor='#BDBDBD')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Plot LDM total loss on right
    if 'total' in ldm_history and len(ldm_history['total']) > 0:
        epochs_ldm = np.arange(1, len(ldm_history['total']) + 1)
        ax2.plot(epochs_ldm, ldm_history['total'], 
                color=COLORS['total'], linewidth=3.5, alpha=0.95, 
                label='Total Loss', zorder=100)
        
        # Add other LDM components if available with soft colors
        component_idx = 0
        for key in ['denoise', 'moment', 'aux']:
            if key in ldm_history and len(ldm_history[key]) > 0:
                color = COLOR_LIST[component_idx % len(COLOR_LIST)]
                ax2.plot(epochs_ldm, ldm_history[key], 
                        color=color, linewidth=2.2, alpha=0.75, 
                        label=key.replace('_', ' ').title(), zorder=10)
                component_idx += 1
    
    ax2.set_xlabel('Epoch', fontweight='600', color='#2C3E50')
    ax2.set_ylabel('Loss Value', fontweight='600', color='#2C3E50')
    ax2.set_title('LDM Training Progress', fontweight='bold', pad=15, color='#2C3E50')
    ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    ax2.legend(loc='best', framealpha=0.95, edgecolor='#BDBDBD')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Overall title with soft color
    fig.suptitle('VAE and LDM Training Comparison', 
                fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[VISUALIZATION] Combined training summary saved to: {save_path}")
