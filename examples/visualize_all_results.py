"""
Example: Comprehensive Results Visualization

This script demonstrates how to use all plotting functions to visualize
complete pipeline results including:
- VAE reconstruction quality and latent space organization
- LDM generation quality and spectral fidelity
- Classification performance (confusion matrix, ROC curves)
- EEG quality assessment across all stages

Usage:
    python examples/visualize_all_results.py --output_dir ./results --checkpoint ./checkpoints/best_model.pth
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.comprehensive_visualization import ComprehensiveVisualizer
from models import DecoupledVAE, LatentDiffusionUNetEEG, CNNBiLSTM
from data import get_dataset
from configs.dataset_config import DatasetConfig
from configs.model_config import ModelConfig
from sklearn.metrics import accuracy_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_models(checkpoint_path: str, device: str = 'cuda'):
    """
    Load trained models from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load models on
    
    Returns:
        Tuple of (vae, ldm, classifier)
    """
    logger.info(f"Loading models from: {checkpoint_path}")
    
    # Initialize models
    vae = DecoupledVAE(
        in_channels=ModelConfig.VAE_IN_CHANNELS,
        base_channels=128,
        latent_channels=ModelConfig.VAE_LATENT_DIM,
        ch_mults=(1, 2, 4, 8),
        blocks_per_stage=2,
        use_feature_cond=True,
        feature_cond_dim=12
    ).to(device)
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=ModelConfig.VAE_LATENT_DIM,
        base_unet_channels=256,
        time_emb_dim=256,
        num_classes=3
    ).to(device)
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=ModelConfig.VAE_IN_CHANNELS,
        seq_length=DatasetConfig.SEGMENT_SAMPLES
    ).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'vae_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['vae_state_dict'])
        if 'ldm_state_dict' in checkpoint:
            ldm.load_state_dict(checkpoint['ldm_state_dict'])
        if 'classifier_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        logger.info(f"✓ Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Using randomly initialized models.")
    
    return vae, ldm, classifier


def prepare_test_data(dataset_name: str = 'siena', num_samples: int = 500):
    """
    Prepare test data for visualization.
    
    Args:
        dataset_name: Name of dataset to load
        num_samples: Maximum number of samples to use
    
    Returns:
        Tuple of (samples, labels)
    """
    logger.info(f"Loading {dataset_name} dataset...")
    
    patient_data = get_dataset(dataset_name)
    
    # Aggregate all patient data
    all_samples = []
    all_labels = []
    
    for patient_id, (segments, labels) in patient_data.items():
        all_samples.append(segments)
        all_labels.append(labels)
    
    samples = np.concatenate(all_samples, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Limit to num_samples
    if len(samples) > num_samples:
        indices = np.random.choice(len(samples), num_samples, replace=False)
        samples = samples[indices]
        labels = labels[indices]
    
    logger.info(f"✓ Loaded {len(samples)} samples")
    logger.info(f"  Class distribution: {np.bincount(labels)}")
    
    return samples, labels


def main(args):
    """
    Main function to generate comprehensive visualization.
    """
    print("="*80)
    print(" GenEEG Comprehensive Results Visualization")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comprehensive visualizer
    visualizer = ComprehensiveVisualizer(
        output_dir=str(output_dir),
        class_names=['Normal', 'Preictal', 'Ictal']
    )
    
    # Load models
    vae, ldm, classifier = load_models(args.checkpoint, str(device))
    vae.eval()
    ldm.eval()
    classifier.eval()
    
    # Prepare test data
    samples, labels = prepare_test_data(args.dataset, args.num_samples)
    
    # Convert to tensor
    samples_tensor = torch.from_numpy(samples).float().to(device)
    labels_tensor = torch.from_numpy(labels).long()
    
    # ======================
    # 1. VAE Analysis
    # ======================
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: VAE Reconstruction Quality Analysis")
    logger.info("="*80)
    
    # Create a simple DataLoader wrapper
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(samples_tensor, labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    vae_metrics = visualizer.generate_vae_report(
        vae_model=vae,
        data_loader=test_loader,
        epoch=999,  # Final epoch
        metrics={
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'contrastive_loss': 0.0,
            'feature_recon_loss': 0.0
        },
        device=str(device)
    )
    
    logger.info(f"✓ VAE analysis complete")
    logger.info(f"  Latent space metrics: {vae_metrics}")
    
    # ======================
    # 2. LDM Generation Quality
    # ======================
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: LDM Generation Quality Analysis")
    logger.info("="*80)
    
    ldm_metrics = visualizer.generate_ldm_report(
        ldm_model=ldm,
        vae_model=vae,
        num_samples_per_class=50,
        real_samples=samples,
        real_labels=labels,
        device=str(device)
    )
    
    logger.info(f"✓ LDM analysis complete")
    logger.info(f"  Generation quality metrics: {ldm_metrics}")
    
    # ======================
    # 3. Classification Performance
    # ======================
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: Classification Performance Analysis")
    logger.info("="*80)
    
    # Get classifier predictions
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = classifier(batch_x)
            probs = torch.softmax(logits, dim=1)
            
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_pred_proba = np.concatenate(all_probs, axis=0)
    y_true = labels
    
    classification_metrics = visualizer.generate_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        scenario_name="Final_Test_Set"
    )
    
    logger.info(f"✓ Classification analysis complete")
    logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"  Macro-avg AUC: {classification_metrics.get('auc_macro_avg', 'N/A')}")
    
    # ======================
    # 4. Final Comprehensive Report
    # ======================
    logger.info("\n" + "="*80)
    logger.info("STAGE 4: Generating Final Comprehensive Report")
    logger.info("="*80)
    
    additional_info = {
        'Dataset': args.dataset,
        'Num_Samples': len(samples),
        'Device': str(device),
        'Checkpoint': args.checkpoint,
        'VAE_Params': sum(p.numel() for p in vae.parameters()),
        'LDM_Params': sum(p.numel() for p in ldm.parameters()),
        'Classifier_Params': sum(p.numel() for p in classifier.parameters())
    }
    
    visualizer.generate_final_report(
        vae_metrics=vae_metrics,
        ldm_metrics=ldm_metrics,
        classification_metrics=classification_metrics,
        additional_info=additional_info
    )
    
    logger.info(f"✓ Final report generated")
    
    print("\n" + "="*80)
    print(" VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - VAE Analysis: {output_dir / 'vae_analysis'}")
    print(f"  - LDM Analysis: {output_dir / 'ldm_analysis'}")
    print(f"  - Classification: {output_dir / 'classification'}")
    print(f"  - Final Report: {output_dir / 'final_report.txt'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive results visualization")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/comprehensive_analysis',
        help='Directory to save all visualizations'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='siena',
        choices=['siena'],
        help='Dataset to use for visualization'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help='Maximum number of samples to use'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (default: use GPU if available)'
    )
    
    args = parser.parse_args()
    
    main(args)
