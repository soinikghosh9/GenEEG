"""
Quick Test Script - Ultra-Fast Workflow Test
Tests the full pipeline with minimal epochs for rapid validation
Key configuration:
- VAE: 2 epochs (initial) / 2 epochs (fine-tuning) 
- LDM: 2 epochs (initial) / 2 epochs (fine-tuning)
- Reduced synthetic samples (100 per class instead of 500)
- Immediate per-fold reporting
- Progress tracking with time estimates
Note: This is for workflow testing only. For quality results, use run_siena_experiment.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from configs.dataset_config import DatasetConfig
from configs.training_config import TrainingConfig, SEED_VALUE
from utils.logger import setup_training_logging, get_logger


def main():
    """Quick test with 3 patients and optimized training"""
    # Set random seeds
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)

    # Setup logging
    setup_training_logging(log_level='INFO', output_dir='./test_output')
    logger = get_logger(__name__)

    print("="*80)
    print(" GenEEG: ULTRA-FAST WORKFLOW TEST")
    print("="*80)
    print(" Training Configuration:")
    print("   - VAE epochs: 2 (initial) / 2 (fine-tune)")
    print("   - LDM epochs: 2 (initial) / 2 (fine-tune)")
    print("   - Reduced synthetic samples: 100 per class")
    print("   - 3 patients for rapid testing")
    print("   - Expected time: ~15-30 minutes per fold")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        from pipeline.cl_lopo_pipeline import main_cl_lopo_validation
        
        logger.info("\n"+"="*80)
        logger.info("Starting ULTRA-FAST CL-LOPO Pipeline Test")
        logger.info("  Max Patients: 3 (2 train + 1 test per fold)")
        logger.info("  VAE Epochs: 2 (initial) / 2 (fine-tune)")
        logger.info("  LDM Epochs: 2 (initial) / 2 (fine-tune)")
        logger.info("  Synthetic Samples: 100 per class")
        logger.info("  Expected time: ~15-30 minutes per fold")
        logger.info("="*80+"\n")
        
        # Run with 3 patients for quick validation
        results = main_cl_lopo_validation(
            data_root_dir=DatasetConfig.SIENA_ROOT_DIR,
            output_dir='./test_output/quick_test',
            device=str(device),
            dataset_name='siena',
            max_patients=3  # QUICK TEST: 3 patients
        )
        
        logger.info("\n"+"="*80)
        logger.info(" QUICK TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        if results and len(results) > 0:
            print("\n✓ Pipeline executed successfully!")
            print(f"✓ Processed {len(results)} fold(s) with 3 patients")
            print("✓ Check ./test_output/quick_test/ for:")
            print("    - Per-fold summaries (LOPO_fold_N_summary.json)")
            print("    - Progress reports (progress_after_fold_N.json)")
            print("    - Quality assessments")
            print("    - Classifier comparisons")
            print("    - Final aggregated results")
        else:
            print("\n⚠ Warning: No results returned")
            
    except Exception as e:
        logger.error(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
