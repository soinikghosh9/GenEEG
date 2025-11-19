"""
Optimized Script to Run Siena Experiment
Includes performance optimizations while maintaining quality:
- Reduced epochs with maintained convergence
- Efficient synthetic sample generation
- Progressive reporting and checkpointing
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
    """Main execution function with optimizations"""
    # Set random seeds
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)
    
    # PRODUCTION MODE: Use optimized epochs for faster training
    # Note: Epochs are now optimized in pipeline (auto-reduced by 4x)
    # This provides good convergence while reducing training time by ~75%
    print(f"[CONFIG] Optimized Production Mode:")
    print(f"  - VAE: {TrainingConfig.VAE_EPOCHS} → ~{TrainingConfig.VAE_EPOCHS // 4} epochs (first patient)")
    print(f"  - LDM: {TrainingConfig.LDM_EPOCHS} → ~{TrainingConfig.LDM_EPOCHS // 4} epochs (first patient)")
    print(f"  - Fine-tuning: Reduced by 2x")
    print(f"  - Synthetic samples: 100 per class (fast generation)")
    print(f"  - Progressive reporting enabled")

    # Setup logging
    setup_training_logging(log_level='INFO', output_dir='./outputs')
    logger = get_logger(__name__)

    print("="*80)
    print(" GenEEG: Siena Dataset Experiment (Optimized)")
    print("="*80)

    # Skip redundant data loading - pipeline will load it
    logger.info("Starting Optimized CL-LOPO Pipeline...")
    logger.info("(Dataset will be loaded by pipeline)")
    
    # Step 1: Setup Device
    logger.info("\nStep 1: Checking device...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info("[OK] Device configured successfully")

    # Step 2: Run Optimized CL-LOPO Pipeline
    logger.info("\nStep 2: Running Optimized CL-LOPO Pipeline...")
    logger.info("  Expected time: ~1.5-2 hours per fold (was ~8 hours)")
    logger.info("  Total estimated time: ~20-30 hours for 14 patients (was ~100+ hours)")

    try:
        from pipeline.cl_lopo_pipeline import main_cl_lopo_validation
        
        logger.info("Starting full CL-LOPO validation...")
        logger.info(f"  Dataset: Siena (14 patients)")
        logger.info(f"  Device: {device}")
        logger.info(f"  Output: ./outputs/cl_lopo_siena")
        
        # Run the pipeline with all optimizations
        results = main_cl_lopo_validation(
            data_root_dir=DatasetConfig.SIENA_ROOT_DIR,
            output_dir='./outputs/cl_lopo_siena',
            device=str(device),
            dataset_name='siena',
            max_patients=None  # Process all patients
        )
        
        logger.info("[OK] CL-LOPO validation completed")
        
        # Print results summary
        print("\n" + "="*80)
        print(" FINAL RESULTS")
        print("="*80)
        
        if results and len(results) > 0:
            print(f"\nTotal LOPO Folds: {len(results)}")
            
            # Aggregate accuracies across all folds
            all_accuracies = []
            for fold_result in results:
                # Extract accuracy from classifier performance
                if 'classifier_performance' in fold_result:
                    for scenario_name, scenario_results in fold_result['classifier_performance'].items():
                        for clf_name, clf_results in scenario_results.items():
                            acc = clf_results.get('accuracy', None)
                            if acc is not None:
                                all_accuracies.append(acc)
            
            if all_accuracies:
                # numpy already imported at module level
                mean_acc = np.mean(all_accuracies)
                std_acc = np.std(all_accuracies)
                print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"Best Accuracy: {np.max(all_accuracies):.4f}")
                print(f"Worst Accuracy: {np.min(all_accuracies):.4f}")
        
        print("\n" + "="*80)
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"[ERROR] Error in CL-LOPO pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print(" Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to: ./outputs/cl_lopo_siena/")
    print(f"Logs saved to: ./outputs/experiment_log.txt")
    print("="*80 + "\n")

if __name__ == '__main__':
    # CRITICAL: This is required for multiprocessing on Windows
    # when using num_workers > 0 in DataLoader
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
