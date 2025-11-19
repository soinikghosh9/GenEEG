"""
VAE Visualization and Monitoring Tools

Comprehensive visualization suite for monitoring VAE training quality:
1. Reconstruction quality plots
2. Latent space visualization (t-SNE, UMAP)
3. Class separation metrics
4. Loss curves and component analysis
5. Spectral analysis (original vs reconstructed)
6. Channel-wise reconstruction quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import signal
from typing import Dict, List, Tuple, Optional
import os


def to_numpy(data):
    """
    Convert tensor or numpy array to numpy array.
    Handles both torch.Tensor and np.ndarray inputs.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        # Try to convert to numpy (for lists, etc.)
        return np.array(data)


class VAEVisualizer:
    """
    Comprehensive VAE visualization and monitoring toolkit.
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(self, metrics: Dict[str, List[float]], epoch: int, save_name: str = "training_curves.png"):
        """
        Plot training loss curves with all components on normalized scale.
        
        Args:
            metrics: Dictionary with loss histories
            epoch: Current epoch number
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main plot: All losses normalized on same scale
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.set_title(f'VAE Training Curves - Epoch {epoch} (Normalized)', fontsize=16, fontweight='bold')
        
        # Collect all available losses for normalization
        loss_components = []
        if 'total_loss' in metrics and isinstance(metrics['total_loss'], list) and len(metrics['total_loss']) > 0:
            loss_components.append(('Total Loss', metrics['total_loss'], 'blue', '-'))
        if 'recon_loss' in metrics and isinstance(metrics['recon_loss'], list) and len(metrics['recon_loss']) > 0:
            loss_components.append(('Reconstruction (L1)', metrics['recon_loss'], 'orange', '-'))
        if 'kl_loss' in metrics and isinstance(metrics['kl_loss'], list) and len(metrics['kl_loss']) > 0:
            loss_components.append(('KL Divergence', metrics['kl_loss'], 'green', '--'))
        if 'contrastive_loss' in metrics and isinstance(metrics['contrastive_loss'], list) and len(metrics['contrastive_loss']) > 0:
            loss_components.append(('Contrastive', metrics['contrastive_loss'], 'purple', '--'))
        if 'feature_recon_loss' in metrics and isinstance(metrics['feature_recon_loss'], list) and len(metrics['feature_recon_loss']) > 0:
            loss_components.append(('Feature Recon', metrics['feature_recon_loss'], 'red', ':'))
        
        # Normalize each loss to [0, 1] range for comparison
        epochs_range = range(1, len(loss_components[0][1]) + 1) if loss_components else []
        
        for name, values, color, style in loss_components:
            values_np = np.array(values)
            # Min-max normalization
            val_min, val_max = values_np.min(), values_np.max()
            if val_max > val_min:
                normalized = (values_np - val_min) / (val_max - val_min)
            else:
                normalized = np.zeros_like(values_np)
            
            ax_main.plot(epochs_range, normalized, label=f'{name} (range: [{val_min:.2e}, {val_max:.2e}])', 
                        color=color, linestyle=style, linewidth=2, alpha=0.8)
        
        ax_main.set_xlabel('Epoch', fontsize=13)
        ax_main.set_ylabel('Normalized Loss [0, 1]', fontsize=13)
        ax_main.set_ylim(-0.05, 1.05)
        ax_main.legend(loc='best', fontsize=10, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Individual subplots for key losses (original scale)
        # Total loss
        ax1 = fig.add_subplot(gs[1, 0])
        if 'total_loss' in metrics and isinstance(metrics['total_loss'], list) and len(metrics['total_loss']) > 0:
            ax1.plot(metrics['total_loss'], label='Total Loss', color='blue', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Total Loss (Original Scale)', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax2 = fig.add_subplot(gs[1, 1])
        if 'recon_loss' in metrics and isinstance(metrics['recon_loss'], list) and len(metrics['recon_loss']) > 0:
            ax2.plot(metrics['recon_loss'], label='Reconstruction', color='orange', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Loss', fontsize=11)
            ax2.set_title('Reconstruction Loss (Original Scale)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Use constrained_layout instead of tight_layout for GridSpec compatibility
        # or catch the warning if layout is already constrained
        try:
            plt.tight_layout()
        except:
            pass  # GridSpec may already handle layout
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_reconstruction_samples(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                                   labels: torch.Tensor, epoch: int, 
                                   n_samples: int = 4, save_name: str = "reconstruction_samples.png"):
        """
        Plot original vs reconstructed EEG samples.
        
        Args:
            original: Original EEG signals (batch, channels, length)
            reconstructed: Reconstructed signals (batch, channels, length)
            labels: Class labels (batch,)
            epoch: Current epoch
            n_samples: Number of samples to plot
            save_name: Filename to save plot
        """
        # Convert to numpy (handle both tensor and numpy inputs)
        orig_np = to_numpy(original)[:n_samples]
        recon_np = to_numpy(reconstructed)[:n_samples]
        labels_np = to_numpy(labels)[:n_samples]
        
        class_names = {0: 'Normal', 1: 'Preictal', 2: 'Ictal'}
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(16, 3 * n_samples))
        fig.suptitle(f'Reconstruction Quality - Epoch {epoch}', fontsize=16)
        
        for i in range(n_samples):
            # Plot original (first 3 channels)
            for ch in range(min(3, orig_np.shape[1])):
                axes[i, 0].plot(orig_np[i, ch, :], alpha=0.7, label=f'Ch {ch}')
            axes[i, 0].set_title(f'Original - {class_names.get(labels_np[i], "Unknown")}')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].legend(loc='upper right')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot reconstructed
            for ch in range(min(3, recon_np.shape[1])):
                axes[i, 1].plot(recon_np[i, ch, :], alpha=0.7, label=f'Ch {ch}')
            axes[i, 1].set_title(f'Reconstructed - {class_names.get(labels_np[i], "Unknown")}')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].legend(loc='upper right')
            axes[i, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
        
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space_tsne(self, latent_codes: torch.Tensor, labels: torch.Tensor, 
                              epoch: int, save_name: str = "latent_tsne.png"):
        """
        Visualize latent space using t-SNE with class coloring.
        
        Args:
            latent_codes: Latent representations (batch, latent_dim) or (batch, latent_dim, length)
            labels: Class labels (batch,)
            epoch: Current epoch
            save_name: Filename to save plot
        """
        # Flatten if 3D
        if isinstance(latent_codes, torch.Tensor) and latent_codes.ndim == 3:
            latent_codes = latent_codes.mean(dim=2)  # Average over time
        elif isinstance(latent_codes, np.ndarray) and latent_codes.ndim == 3:
            latent_codes = latent_codes.mean(axis=2)
        
        # Convert to numpy
        z_np = to_numpy(latent_codes)
        labels_np = to_numpy(labels)
        
        # Limit samples for t-SNE (computational efficiency)
        max_samples = 2000
        if len(z_np) > max_samples:
            indices = np.random.choice(len(z_np), max_samples, replace=False)
            z_np = z_np[indices]
            labels_np = labels_np[indices]
        
        # Compute t-SNE
        print(f"Computing t-SNE for {len(z_np)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_np) - 1))
        z_tsne = tsne.fit_transform(z_np)
        
        # Compute separation metrics
        silhouette = silhouette_score(z_np, labels_np)
        davies_bouldin = davies_bouldin_score(z_np, labels_np)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        class_names = {0: 'Normal', 1: 'Preictal', 2: 'Ictal'}
        colors = {0: 'blue', 1: 'orange', 2: 'red'}
        
        for class_id in np.unique(labels_np):
            mask = labels_np == class_id
            ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1], 
                      c=colors.get(class_id, 'gray'),
                      label=class_names.get(class_id, f'Class {class_id}'),
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'Latent Space t-SNE - Epoch {epoch}\n'
                    f'Silhouette Score: {silhouette:.3f} | Davies-Bouldin Index: {davies_bouldin:.3f}',
                    fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        
        return silhouette, davies_bouldin
    
    def plot_spectral_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor,
                                labels: torch.Tensor, epoch: int, sfreq: float = 256.0,
                                save_name: str = "spectral_comparison.png"):
        """
        Compare power spectral density of original vs reconstructed signals.
        
        Args:
            original: Original signals (batch, channels, length)
            reconstructed: Reconstructed signals (batch, channels, length)
            labels: Class labels
            epoch: Current epoch
            sfreq: Sampling frequency
            save_name: Filename to save plot
        """
        # Convert to numpy
        orig_np = to_numpy(original)
        recon_np = to_numpy(reconstructed)
        labels_np = to_numpy(labels)
        
        # Select samples from each class
        class_names = {0: 'Normal', 1: 'Preictal', 2: 'Ictal'}
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'Power Spectral Density Comparison - Epoch {epoch}', fontsize=16)
        
        for class_id, ax in enumerate(axes):
            # Get samples from this class
            mask = labels_np == class_id
            if not mask.any():
                continue
            
            # Average across samples and channels
            orig_class = orig_np[mask].mean(axis=(0, 1))  # Average over batch and channels
            recon_class = recon_np[mask].mean(axis=(0, 1))
            
            # Compute PSD
            freqs_orig, psd_orig = signal.welch(orig_class, fs=sfreq, nperseg=min(256, len(orig_class)))
            freqs_recon, psd_recon = signal.welch(recon_class, fs=sfreq, nperseg=min(256, len(recon_class)))
            
            # Plot
            ax.semilogy(freqs_orig, psd_orig, label='Original', linewidth=2, alpha=0.8)
            ax.semilogy(freqs_recon, psd_recon, label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
            ax.set_xlim([0, 70])  # Focus on 0-70 Hz (relevant for EEG)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'{class_names[class_id]} Class')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_channel_wise_error(self, original: torch.Tensor, reconstructed: torch.Tensor,
                               epoch: int, save_name: str = "channel_error.png"):
        """
        Plot reconstruction error per EEG channel.
        
        Args:
            original: Original signals (batch, channels, length)
            reconstructed: Reconstructed signals (batch, channels, length)
            epoch: Current epoch
            save_name: Filename to save plot
        """
        # Compute per-channel MSE
        if isinstance(original, torch.Tensor):
            mse_per_channel = ((original - reconstructed) ** 2).mean(dim=(0, 2)).detach().cpu().numpy()
        else:
            mse_per_channel = ((original - reconstructed) ** 2).mean(axis=(0, 2))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        channels = np.arange(len(mse_per_channel))
        ax.bar(channels, mse_per_channel, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Channel Index', fontsize=12)
        ax.set_ylabel('Mean Squared Error', fontsize=12)
        ax.set_title(f'Reconstruction Error per Channel - Epoch {epoch}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_error = mse_per_channel.mean()
        ax.axhline(y=mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
        ax.legend()
        
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_latent_distribution(self, mu: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor,
                                epoch: int, save_name: str = "latent_distribution.png"):
        """
        Plot latent code distributions (mu and logvar statistics).
        
        Args:
            mu: Latent means (batch, latent_dim, length)
            logvar: Latent log variances (batch, latent_dim, length)
            labels: Class labels
            epoch: Current epoch
            save_name: Filename to save plot
        """
        # Average over spatial dimension
        if isinstance(mu, torch.Tensor):
            if mu.ndim == 3:
                mu = mu.mean(dim=2)
                logvar = logvar.mean(dim=2)
        elif isinstance(mu, np.ndarray):
            if mu.ndim == 3:
                mu = mu.mean(axis=2)
                logvar = logvar.mean(axis=2)
        
        mu_np = to_numpy(mu)
        logvar_np = to_numpy(logvar)
        labels_np = to_numpy(labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Latent Code Statistics - Epoch {epoch}', fontsize=16)
        
        class_names = {0: 'Normal', 1: 'Preictal', 2: 'Ictal'}
        colors = {0: 'blue', 1: 'orange', 2: 'red'}
        
        # Mu distribution per class
        for class_id in np.unique(labels_np):
            mask = labels_np == class_id
            mu_class = mu_np[mask].flatten()
            axes[0, 0].hist(mu_class, bins=50, alpha=0.5, label=class_names.get(class_id, f'Class {class_id}'),
                          color=colors.get(class_id, 'gray'))
        axes[0, 0].set_xlabel('Latent Mean (μ)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Latent Means')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Logvar distribution per class
        for class_id in np.unique(labels_np):
            mask = labels_np == class_id
            logvar_class = logvar_np[mask].flatten()
            axes[0, 1].hist(logvar_class, bins=50, alpha=0.5, label=class_names.get(class_id, f'Class {class_id}'),
                          color=colors.get(class_id, 'gray'))
        axes[0, 1].set_xlabel('Latent Log Variance (log σ²)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Latent Log Variances')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean and std of mu per dimension
        mu_mean_per_dim = mu_np.mean(axis=0)
        mu_std_per_dim = mu_np.std(axis=0)
        dims = np.arange(len(mu_mean_per_dim))
        
        axes[1, 0].errorbar(dims, mu_mean_per_dim, yerr=mu_std_per_dim, fmt='o', alpha=0.6, capsize=3)
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Mean ± Std')
        axes[1, 0].set_title('Latent Mean Statistics per Dimension')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mean and std of logvar per dimension
        logvar_mean_per_dim = logvar_np.mean(axis=0)
        logvar_std_per_dim = logvar_np.std(axis=0)
        
        axes[1, 1].errorbar(dims, logvar_mean_per_dim, yerr=logvar_std_per_dim, fmt='o', alpha=0.6, capsize=3, color='orange')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Mean ± Std')
        axes[1, 1].set_title('Latent LogVar Statistics per Dimension')
        axes[1, 1].grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_latent_metrics(self, latent_codes: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantitative metrics for latent space quality.
        
        Args:
            latent_codes: Latent representations (batch, latent_dim) or (batch, latent_dim, length)
            labels: Class labels (batch,)
        
        Returns:
            Dictionary of metrics
        """
        # Flatten if 3D
        if isinstance(latent_codes, torch.Tensor):
            if latent_codes.ndim == 3:
                latent_codes = latent_codes.mean(dim=2)
        elif isinstance(latent_codes, np.ndarray):
            if latent_codes.ndim == 3:
                latent_codes = latent_codes.mean(axis=2)
        
        z_np = to_numpy(latent_codes)
        labels_np = to_numpy(labels)
        
        # Compute metrics
        silhouette = silhouette_score(z_np, labels_np)
        davies_bouldin = davies_bouldin_score(z_np, labels_np)
        
        # Compute per-class statistics
        class_stats = {}
        for class_id in np.unique(labels_np):
            mask = labels_np == class_id
            z_class = z_np[mask]
            class_stats[f'class_{class_id}_mean_norm'] = np.linalg.norm(z_class.mean(axis=0))
            class_stats[f'class_{class_id}_std_norm'] = np.linalg.norm(z_class.std(axis=0))
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            **class_stats
        }
    
    def create_comprehensive_report(self, epoch: int, metrics: Dict, original: torch.Tensor,
                                   reconstructed: torch.Tensor, mu: torch.Tensor, 
                                   logvar: torch.Tensor, labels: torch.Tensor):
        """
        Create comprehensive visualization report for an epoch.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics dictionary
            original: Original signals
            reconstructed: Reconstructed signals
            mu: Latent means
            logvar: Latent log variances
            labels: Class labels
        """
        print(f"\n[INFO] Generating comprehensive visualization report for epoch {epoch}...")
        
        # Create epoch-specific subdirectory
        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save to epoch directory
        old_save_dir = self.save_dir
        self.save_dir = epoch_dir
        
        try:
            # 1. Training curves
            self.plot_training_curves(metrics, epoch)
            
            # 2. Reconstruction samples
            self.plot_reconstruction_samples(original, reconstructed, labels, epoch, n_samples=6)
            
            # 3. Latent space t-SNE
            sil, db = self.plot_latent_space_tsne(mu, labels, epoch)
            
            # 4. Spectral comparison
            self.plot_spectral_comparison(original, reconstructed, labels, epoch)
            
            # 5. Channel-wise error
            self.plot_channel_wise_error(original, reconstructed, epoch)
            
            # 6. Latent distribution
            self.plot_latent_distribution(mu, logvar, labels, epoch)
            
            # 7. Compute and save latent metrics
            latent_metrics = self.compute_latent_metrics(mu, labels)
            
            print(f"[INFO] Visualization report saved to: {epoch_dir}")
            print(f"[METRICS] Silhouette Score: {sil:.3f}, Davies-Bouldin: {db:.3f}")
            
            return latent_metrics
            
        finally:
            self.save_dir = old_save_dir
