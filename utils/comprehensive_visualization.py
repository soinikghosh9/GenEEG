"""
Comprehensive Visualization Module

This module integrates all visualization tools to provide complete analysis of:
- VAE reconstruction quality and latent space organization
- LDM generation quality and spectral characteristics
- Classification performance (confusion matrix, ROC curves)
- EEG quality assessment (temporal, spectral, channel-wise)

Key Components:
    - ComprehensiveVisualizer: Main class integrating all plotting functions
    - generate_vae_report: Complete VAE analysis
    - generate_ldm_report: LDM generation quality analysis
    - generate_classification_report: Classification performance visualization
    - generate_final_report: Complete pipeline results

Author: GenEEG Team
Date: 2025
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from utils.vae_visualization import VAEVisualizer
from evaluation.metrics import plot_confusion_matrix, plot_roc_auc_curves
from sklearn.metrics import confusion_matrix


class ComprehensiveVisualizer:
    """
    Comprehensive visualization integrating all plotting functions.
    
    Provides end-to-end analysis of the GenEEG pipeline including:
    - VAE training dynamics and reconstruction quality
    - Latent space organization and class separation
    - LDM generation quality and fidelity
    - Classification performance metrics
    - Spectral and temporal EEG quality assessment
    """
    
    def __init__(self, output_dir: str, class_names: List[str] = None):
        """
        Initialize comprehensive visualizer.
        
        Args:
            output_dir: Base directory for saving all plots
            class_names: List of class names for labeling (default: ['Normal', 'Preictal', 'Ictal'])
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized results
        self.vae_dir = self.output_dir / 'vae_analysis'
        self.ldm_dir = self.output_dir / 'ldm_analysis'
        self.classification_dir = self.output_dir / 'classification'
        self.quality_dir = self.output_dir / 'quality_assessment'
        
        for dir_path in [self.vae_dir, self.ldm_dir, self.classification_dir, self.quality_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or ['Normal', 'Preictal', 'Ictal']
        self.logger = logging.getLogger(__name__)
        
        # Initialize VAE visualizer
        self.vae_visualizer = VAEVisualizer(str(self.vae_dir))
    
    def generate_vae_report(
        self,
        vae_model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        metrics: Dict[str, float] = None,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Generate comprehensive VAE analysis report.
        
        Creates visualizations for:
        - Training curves (loss components)
        - Reconstruction quality (sample comparisons)
        - Latent space organization (t-SNE with metrics)
        - Spectral fidelity (PSD comparison)
        - Channel-wise reconstruction errors
        - Latent distribution statistics
        
        Args:
            vae_model: Trained VAE model
            data_loader: DataLoader for evaluation data
            epoch: Current epoch number
            metrics: Dictionary of training metrics
            device: Device for computation
        
        Returns:
            Dictionary containing latent space quality metrics
        """
        self.logger.info(f"Generating VAE comprehensive report for epoch {epoch}...")
        
        vae_model.eval()
        device_obj = torch.device(device)
        
        # Collect data for visualization
        original_samples = []
        reconstructed_samples = []
        latent_codes = []
        mu_samples = []
        logvar_samples = []
        labels_list = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit samples for visualization
                    break
                
                data = data.to(device_obj)
                recon, mu, logvar, z, is_stable = vae_model(data)
                
                if is_stable:
                    original_samples.append(data.cpu().numpy())
                    reconstructed_samples.append(recon.cpu().numpy())
                    latent_codes.append(z.cpu().numpy())
                    mu_samples.append(mu.cpu().numpy())
                    logvar_samples.append(logvar.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
        
        if not original_samples:
            self.logger.warning("No stable VAE outputs to visualize!")
            return {}
        
        # Concatenate all samples
        original = np.concatenate(original_samples, axis=0)
        reconstructed = np.concatenate(reconstructed_samples, axis=0)
        latent = np.concatenate(latent_codes, axis=0)
        mu = np.concatenate(mu_samples, axis=0)
        logvar = np.concatenate(logvar_samples, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        # Generate comprehensive report using VAEVisualizer
        latent_metrics = self.vae_visualizer.create_comprehensive_report(
            epoch=epoch,
            metrics=metrics,
            original_samples=original,
            reconstructed_samples=reconstructed,
            mu=mu,
            logvar=logvar,
            labels=labels
        )
        
        self.logger.info(f"VAE report generated. Latent space metrics: {latent_metrics}")
        return latent_metrics
    
    def generate_ldm_report(
        self,
        ldm_model: torch.nn.Module,
        vae_model: torch.nn.Module,
        num_samples_per_class: int = 50,
        real_samples: np.ndarray = None,
        real_labels: np.ndarray = None,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Generate LDM generation quality report.
        
        Creates visualizations for:
        - Generated EEG samples (time series)
        - Spectral comparison (generated vs real PSD)
        - Per-class generation quality
        - Sample diversity analysis
        
        Args:
            ldm_model: Trained LDM model
            vae_model: Trained VAE model (for decoding latents)
            num_samples_per_class: Number of samples to generate per class
            real_samples: Real EEG samples for comparison
            real_labels: Labels for real samples
            device: Device for computation
        
        Returns:
            Dictionary containing generation quality metrics
        """
        self.logger.info("Generating LDM quality report...")
        
        ldm_model.eval()
        vae_model.eval()
        device_obj = torch.device(device)
        
        num_classes = len(self.class_names)
        all_generated = []
        all_gen_labels = []
        
        # Generate samples for each class
        with torch.no_grad():
            for class_idx in range(num_classes):
                self.logger.info(f"  Generating {num_samples_per_class} samples for class '{self.class_names[class_idx]}'...")
                
                # Generate latent codes using LDM's sampling method
                labels_tensor = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(device_obj)
                
                # Sample from diffusion model (assuming DDPM sampling)
                if hasattr(ldm_model, 'sample'):
                    latent_samples = ldm_model.sample(
                        batch_size=num_samples_per_class,
                        labels=labels_tensor,
                        num_steps=50  # DDPM steps
                    )
                else:
                    # Fallback: sample from standard normal and run reverse diffusion
                    latent_shape = (num_samples_per_class, vae_model.latent_channels, 
                                   vae_model.latent_spatial_dim)
                    latent_samples = torch.randn(latent_shape).to(device_obj)
                
                # Decode latents to EEG space
                generated_eeg = vae_model.decode(latent_samples)
                
                all_generated.append(generated_eeg.cpu().numpy())
                all_gen_labels.append(labels_tensor.cpu().numpy())
        
        generated_samples = np.concatenate(all_generated, axis=0)
        generated_labels = np.concatenate(all_gen_labels, axis=0)
        
        # Plot generated samples
        self._plot_ldm_samples(generated_samples, generated_labels)
        
        # Spectral comparison if real samples provided
        quality_metrics = {}
        if real_samples is not None and real_labels is not None:
            quality_metrics = self._plot_ldm_spectral_comparison(
                generated_samples, generated_labels,
                real_samples, real_labels
            )
        
        self.logger.info(f"LDM report generated. Quality metrics: {quality_metrics}")
        return quality_metrics
    
    def _plot_ldm_samples(self, samples: np.ndarray, labels: np.ndarray):
        """Plot LDM generated EEG samples."""
        num_samples_to_plot = min(6, len(samples))
        num_channels = min(4, samples.shape[1])  # Plot first 4 channels
        
        fig, axes = plt.subplots(num_samples_to_plot, num_channels, 
                                figsize=(16, 2.5 * num_samples_to_plot))
        
        for i in range(num_samples_to_plot):
            for ch in range(num_channels):
                ax = axes[i, ch] if num_samples_to_plot > 1 else axes[ch]
                
                ax.plot(samples[i, ch, :], linewidth=0.5, color='darkblue')
                ax.set_title(f"Class: {self.class_names[labels[i]]} | Ch{ch+1}", fontsize=9)
                ax.set_xlabel("Sample", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.ldm_dir / 'generated_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"  LDM samples plot saved to: {save_path}")
    
    def _plot_ldm_spectral_comparison(
        self, 
        generated: np.ndarray, 
        gen_labels: np.ndarray,
        real: np.ndarray, 
        real_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compare spectral characteristics of generated vs real EEG."""
        from scipy import signal
        
        fs = 256  # Sampling frequency (adjust based on your data)
        num_classes = len(self.class_names)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4))
        if num_classes == 1:
            axes = [axes]
        
        quality_metrics = {}
        
        for class_idx in range(num_classes):
            # Get samples for this class
            gen_class = generated[gen_labels == class_idx]
            real_class = real[real_labels == class_idx]
            
            if len(gen_class) == 0 or len(real_class) == 0:
                continue
            
            # Compute average PSD across channels
            gen_psd_list = []
            real_psd_list = []
            
            for ch in range(gen_class.shape[1]):
                # Generated PSD
                freqs_gen, psd_gen = signal.welch(gen_class[:, ch, :], fs=fs, nperseg=128)
                gen_psd_list.append(psd_gen.mean(axis=0))
                
                # Real PSD
                freqs_real, psd_real = signal.welch(real_class[:, ch, :], fs=fs, nperseg=128)
                real_psd_list.append(psd_real.mean(axis=0))
            
            gen_psd_avg = np.mean(gen_psd_list, axis=0)
            real_psd_avg = np.mean(real_psd_list, axis=0)
            
            # Plot
            ax = axes[class_idx]
            ax.semilogy(freqs_gen, gen_psd_avg, label='Generated', linewidth=2, alpha=0.8)
            ax.semilogy(freqs_real, real_psd_avg, label='Real', linewidth=2, alpha=0.8, linestyle='--')
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('PSD (µV²/Hz)', fontsize=11)
            ax.set_title(f"{self.class_names[class_idx]}", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 50])  # Focus on 0-50 Hz
            
            # Compute spectral distance (KL divergence approximation)
            psd_gen_norm = gen_psd_avg / (gen_psd_avg.sum() + 1e-10)
            psd_real_norm = real_psd_avg / (real_psd_avg.sum() + 1e-10)
            kl_div = np.sum(psd_real_norm * np.log((psd_real_norm + 1e-10) / (psd_gen_norm + 1e-10)))
            quality_metrics[f'spectral_kl_{self.class_names[class_idx]}'] = kl_div
        
        plt.tight_layout()
        save_path = self.ldm_dir / 'spectral_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"  Spectral comparison plot saved to: {save_path}")
        
        return quality_metrics
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        scenario_name: str = "Test Set"
    ) -> Dict[str, float]:
        """
        Generate classification performance report.
        
        Creates:
        - Confusion matrix (normalized and raw counts)
        - ROC curves (per-class, micro-avg, macro-avg)
        - Performance metrics summary
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (N, num_classes)
            scenario_name: Name for this evaluation scenario
        
        Returns:
            Dictionary containing all performance metrics
        """
        self.logger.info(f"Generating classification report for: {scenario_name}")
        
        num_classes = len(self.class_names)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        # Plot confusion matrix (normalized)
        cm_path_norm = self.classification_dir / f'confusion_matrix_{scenario_name.replace(" ", "_")}_normalized.png'
        plot_confusion_matrix(
            cm, self.class_names,
            title=f'Confusion Matrix - {scenario_name} (Normalized)',
            save_path=str(cm_path_norm),
            normalize=True
        )
        
        # Plot confusion matrix (raw counts)
        cm_path_raw = self.classification_dir / f'confusion_matrix_{scenario_name.replace(" ", "_")}_raw.png'
        plot_confusion_matrix(
            cm, self.class_names,
            title=f'Confusion Matrix - {scenario_name} (Raw Counts)',
            save_path=str(cm_path_raw),
            normalize=False
        )
        
        # Plot ROC curves
        roc_path = self.classification_dir / f'roc_curves_{scenario_name.replace(" ", "_")}.png'
        auc_scores = plot_roc_auc_curves(
            y_true, y_pred_proba,
            num_classes=num_classes,
            class_names_list=self.class_names,
            title_prefix=f"{scenario_name} ROC",
            save_path=str(roc_path)
        )
        
        # Compute overall accuracy
        accuracy = (y_true == y_pred).mean()
        
        # Combine metrics
        performance_metrics = {
            'accuracy': accuracy,
            **auc_scores
        }
        
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Macro-avg AUC: {auc_scores.get('auc_macro_avg', 'N/A')}")
        
        return performance_metrics
    
    def generate_final_report(
        self,
        vae_metrics: Dict[str, float] = None,
        ldm_metrics: Dict[str, float] = None,
        classification_metrics: Dict[str, float] = None,
        additional_info: Dict = None
    ):
        """
        Generate comprehensive final report combining all analyses.
        
        Creates a summary document with:
        - Overall pipeline performance
        - Key metrics from each stage
        - Recommendations and observations
        
        Args:
            vae_metrics: VAE latent space quality metrics
            ldm_metrics: LDM generation quality metrics
            classification_metrics: Classification performance metrics
            additional_info: Additional metadata (training time, hyperparameters, etc.)
        """
        self.logger.info("Generating final comprehensive report...")
        
        report_path = self.output_dir / 'final_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" GenEEG Pipeline - Comprehensive Results Report\n")
            f.write("=" * 80 + "\n\n")
            
            # VAE Section
            if vae_metrics:
                f.write("VAE Latent Space Quality:\n")
                f.write("-" * 40 + "\n")
                for key, value in vae_metrics.items():
                    f.write(f"  {key:30s}: {value:.4f}\n")
                f.write("\n")
            
            # LDM Section
            if ldm_metrics:
                f.write("LDM Generation Quality:\n")
                f.write("-" * 40 + "\n")
                for key, value in ldm_metrics.items():
                    f.write(f"  {key:30s}: {value:.4f}\n")
                f.write("\n")
            
            # Classification Section
            if classification_metrics:
                f.write("Classification Performance:\n")
                f.write("-" * 40 + "\n")
                for key, value in classification_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key:30s}: {value:.4f}\n")
                    else:
                        f.write(f"  {key:30s}: {value}\n")
                f.write("\n")
            
            # Additional Information
            if additional_info:
                f.write("Additional Information:\n")
                f.write("-" * 40 + "\n")
                for key, value in additional_info.items():
                    f.write(f"  {key:30s}: {value}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"All visualizations saved to: {self.output_dir}\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Final report saved to: {report_path}")
        print(f"\n{'='*80}")
        print(f" Comprehensive Report Generated")
        print(f"{'='*80}")
        print(f"Location: {self.output_dir}")
        print(f"  - VAE Analysis: {self.vae_dir}")
        print(f"  - LDM Analysis: {self.ldm_dir}")
        print(f"  - Classification: {self.classification_dir}")
        print(f"  - Quality Assessment: {self.quality_dir}")
        print(f"{'='*80}\n")


def create_visualizer(output_dir: str, class_names: List[str] = None) -> ComprehensiveVisualizer:
    """
    Factory function to create a ComprehensiveVisualizer instance.
    
    Args:
        output_dir: Directory for saving all visualizations
        class_names: List of class names
    
    Returns:
        Initialized ComprehensiveVisualizer
    """
    return ComprehensiveVisualizer(output_dir, class_names)
