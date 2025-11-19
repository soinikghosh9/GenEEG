"""Main entry point for GenEEG experiments.

This script provides a command-line interface for running various experiments
with the GenEEG framework.

Usage:
    python main.py --dataset chbmit --experiment full_cl
    python main.py --dataset siena --experiment ewc_only
    python main.py --help
"""

import os
# Optimized for Intel i7-14700 (20 cores) + RTX 4060 Ti (16GB VRAM) + 32GB RAM
# NumExpr configuration (for pandas/tables operations)
os.environ['NUMEXPR_MAX_THREADS'] = '20'  # Use all cores for max threads
os.environ['NUMEXPR_NUM_THREADS'] = '12'  # Actual compute threads (leave headroom for PyTorch)

# PyTorch CPU threading (works with GPU operations)
os.environ['OMP_NUM_THREADS'] = '8'       # OpenMP threads (conservative for stability)
os.environ['MKL_NUM_THREADS'] = '8'       # Intel MKL threads (optimal for i7)

# CUDA optimizations for RTX 4060 Ti
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async launches (faster)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Reduce fragmentation

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs import DatasetConfig, ModelConfig, TrainingConfig
from data import get_dataset


def setup_experiment(args):
    """Configure experiment based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Set dataset
    DatasetConfig.DATASET_NAME = args.dataset
    
    # Set experiment type
    if args.experiment == 'full_cl':
        # Full continual learning with EWC + ER
        TrainingConfig.USE_EWC = True
        TrainingConfig.USE_EXPERIENCE_REPLAY = True
        TrainingConfig.RUN_EWC_ONLY_ABLATION = False
        TrainingConfig.RUN_ER_ONLY_ABLATION = False
        TrainingConfig.RUN_POOLED_BASELINE = False
        TrainingConfig.RUN_NAIVE_SEQUENTIAL = False
    
    elif args.experiment == 'ewc_only':
        # EWC only (no Experience Replay)
        TrainingConfig.RUN_EWC_ONLY_ABLATION = True
        TrainingConfig.RUN_ER_ONLY_ABLATION = False
        TrainingConfig.RUN_POOLED_BASELINE = False
        TrainingConfig.RUN_NAIVE_SEQUENTIAL = False
    
    elif args.experiment == 'er_only':
        # Experience Replay only (no EWC)
        TrainingConfig.RUN_EWC_ONLY_ABLATION = False
        TrainingConfig.RUN_ER_ONLY_ABLATION = True
        TrainingConfig.RUN_POOLED_BASELINE = False
        TrainingConfig.RUN_NAIVE_SEQUENTIAL = False
    
    elif args.experiment == 'pooled_baseline':
        # Train on all patients simultaneously
        TrainingConfig.RUN_POOLED_BASELINE = True
        TrainingConfig.RUN_EWC_ONLY_ABLATION = False
        TrainingConfig.RUN_ER_ONLY_ABLATION = False
        TrainingConfig.RUN_NAIVE_SEQUENTIAL = False
    
    elif args.experiment == 'naive_sequential':
        # Sequential training without CL techniques
        TrainingConfig.RUN_NAIVE_SEQUENTIAL = True
        TrainingConfig.RUN_POOLED_BASELINE = False
        TrainingConfig.RUN_EWC_ONLY_ABLATION = False
        TrainingConfig.RUN_ER_ONLY_ABLATION = False
    
    # Set batch size if provided
    if args.batch_size:
        TrainingConfig.VAE_BATCH_SIZE = args.batch_size
        TrainingConfig.LDM_BATCH_SIZE = args.batch_size
    
    # Set device
    if args.device:
        TrainingConfig.DEVICE = args.device
    
    # Enable/disable reporting
    TrainingConfig.REPORT_PERCLASS_METRICS = not args.no_perclass
    TrainingConfig.REPORT_PERPATIENT_VARIANCE = not args.no_perpatient
    
    # Validate dataset path
    if not DatasetConfig.validate_paths():
        print(f"\n[ERROR] Dataset not found!")
        print(f"   Current path: {DatasetConfig.get_root_dir()}")
        print(f"\nPlease update the path in configs/dataset_config.py:")
        if args.dataset == 'chbmit':
            print(f"   CHBMIT_ROOT_DIR = r'<your_path_here>'")
        else:
            print(f"   SIENA_ROOT_DIR = r'<your_path_here>'")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"GenEEG Experiment Configuration")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Experiment: {args.experiment}")
    print(f"Device: {TrainingConfig.DEVICE}")
    print(f"Batch Size: {TrainingConfig.VAE_BATCH_SIZE}")
    print(f"EWC Enabled: {TrainingConfig.get_ewc_enabled()}")
    print(f"Experience Replay Enabled: {TrainingConfig.get_er_enabled()}")
    print(f"{'='*80}\n")


def load_data_only(args):
    """Load and validate dataset only (no training).
    
    Args:
        args: Parsed command-line arguments
    """
    print(f"\nLoading {args.dataset.upper()} dataset...")
    
    try:
        patient_data = get_dataset(args.dataset)
        
        print(f"\n[SUCCESS] Successfully loaded {len(patient_data)} patients:")
        print(f"\n{'Patient':<12} {'Segments':<10} {'Normal':<10} {'Preictal':<10} {'Ictal':<10}")
        print(f"{'-'*60}")
        
        total_segments = 0
        total_normal = 0
        total_preictal = 0
        total_ictal = 0
        
        for patient_id, (segments, labels) in patient_data.items():
            n_segments = len(segments)
            n_normal = (labels == 0).sum()
            n_preictal = (labels == 1).sum()
            n_ictal = (labels == 2).sum()
            
            print(f"{patient_id:<12} {n_segments:<10} {n_normal:<10} {n_preictal:<10} {n_ictal:<10}")
            
            total_segments += n_segments
            total_normal += n_normal
            total_preictal += n_preictal
            total_ictal += n_ictal
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':<12} {total_segments:<10} {total_normal:<10} {total_preictal:<10} {total_ictal:<10}")
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"  Total Patients: {len(patient_data)}")
        print(f"  Total Segments: {total_segments}")
        print(f"  Segment Shape: {list(patient_data.values())[0][0].shape}")
        print(f"  Class Distribution:")
        print(f"    Normal:   {total_normal:>6} ({100*total_normal/total_segments:.1f}%)")
        print(f"    Preictal: {total_preictal:>6} ({100*total_preictal/total_segments:.1f}%)")
        print(f"    Ictal:    {total_ictal:>6} ({100*total_ictal/total_segments:.1f}%)")
        
        return patient_data
        
    except Exception as e:
        print(f"\n[ERROR] Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_experiment(args):
    """Run the configured experiment.
    
    Args:
        args: Parsed command-line arguments
    """
    # NOTE: This requires the training module to be extracted from GenEEG.py
    # For now, we'll just load the data and show configuration
    
    print("\n[WARNING] Training modules not yet extracted from GenEEG.py")
    print("   This will be implemented in the next step.")
    print("\nCurrent functionality:")
    print("  [DONE] Dataset loading")
    print("  [DONE] Configuration system")
    print("  [TODO] Model architectures (pending extraction)")
    print("  [TODO] Training pipeline (pending extraction)")
    print("  [TODO] Evaluation metrics (pending extraction)")
    
    # Load data to verify everything works
    patient_data = load_data_only(args)
    
    print(f"\n{'='*80}")
    print(f"Next Steps:")
    print(f"{'='*80}")
    print(f"1. Extract model architectures from GenEEG.py to models/")
    print(f"2. Extract training functions to training/")
    print(f"3. Extract evaluation metrics to evaluation/")
    print(f"4. Re-run this script to execute full pipeline")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GenEEG: Generative EEG for Seizure Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and validate CHB-MIT dataset
  python main.py --dataset chbmit --load-only
  
  # Run full CL experiment on Siena
  python main.py --dataset siena --experiment full_cl
  
  # Run EWC-only ablation on CHB-MIT
  python main.py --dataset chbmit --experiment ewc_only
  
  # Run pooled baseline with custom batch size
  python main.py --dataset siena --experiment pooled_baseline --batch-size 64
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['siena', 'chbmit'],
        default='siena',
        help='Dataset to use (default: siena)'
    )
    
    # Experiment type
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['full_cl', 'ewc_only', 'er_only', 'pooled_baseline', 'naive_sequential'],
        default='full_cl',
        help='Experiment type (default: full_cl)'
    )
    
    # Training options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: cuda if available)'
    )
    
    # Reporting options
    parser.add_argument(
        '--no-perclass',
        action='store_true',
        help='Disable per-class metrics reporting'
    )
    
    parser.add_argument(
        '--no-perpatient',
        action='store_true',
        help='Disable per-patient variance reporting'
    )
    
    # Mode options
    parser.add_argument(
        '--load-only',
        action='store_true',
        help='Only load and validate dataset (no training)'
    )
    
    parser.add_argument(
        '--test-loader',
        action='store_true',
        help='Run dataset loader tests'
    )
    
    args = parser.parse_args()
    
    # Set up experiment configuration
    setup_experiment(args)
    
    # Run appropriate mode
    if args.load_only:
        load_data_only(args)
    elif args.test_loader:
        # Run test script
        if args.dataset == 'chbmit':
            from tests import test_chbmit_loader
            test_chbmit_loader.main()
        else:
            print("Siena loader test not yet implemented")
    else:
        run_experiment(args)


if __name__ == '__main__':
    # CRITICAL: Required for multiprocessing on Windows with num_workers > 0
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
