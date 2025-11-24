"""
Ultra-Fast Test Script for GenEEG
Runs the CL-LOPO pipeline with minimal parameters to verify end-to-end functionality in minutes.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 1. MONKEY-PATCH CONFIGS BEFORE IMPORTING PIPELINE
from configs import training_config
from configs import dataset_config

print("="*80)
print(" MONKEY-PATCHING CONFIGURATION FOR ULTRA-FAST TEST")
print("="*80)

# Patch TrainingConfig
training_config.TrainingConfig.VAE_EPOCHS_DEFAULT = 2
training_config.TrainingConfig.LDM_EPOCHS_DEFAULT = 2
training_config.TrainingConfig.VAE_EPOCHS_PRODUCTION = 2
training_config.TrainingConfig.LDM_EPOCHS_PRODUCTION = 2
training_config.TrainingConfig.VAE_EPOCHS = 2
training_config.TrainingConfig.LDM_EPOCHS = 2
training_config.TrainingConfig.CLASSIFIER_EPOCHS = 2
training_config.TrainingConfig.FINETUNE_EPOCHS = 1

# Reduce batch sizes for speed/memory
training_config.TrainingConfig.VAE_BATCH_SIZE = 16
training_config.TrainingConfig.LDM_BATCH_SIZE = 16
training_config.TrainingConfig.CLASSIFIER_BATCH_SIZE = 16
training_config.TrainingConfig.FINETUNE_BATCH_SIZE = 16

# Reduce generation/validation load
training_config.TrainingConfig.SYNTHETIC_SAMPLES_PER_CLASS = 10  # Only generate 10 samples
training_config.TrainingConfig.LDM_DIFFUSION_TIMESTEPS = 50      # Faster diffusion
training_config.TrainingConfig.EXPERIENCE_REPLAY_BUFFER_SIZE_PER_PATIENT = 50

from pipeline.cl_lopo_pipeline import main_cl_lopo_validation
import pipeline.cl_lopo_pipeline as pipeline_module

# Patch module-level exports in training_config
training_config.VAE_EPOCHS = 2
training_config.LDM_EPOCHS = 2
training_config.CLASSIFIER_EPOCHS = 2
training_config.FINETUNE_EPOCHS = 1
training_config.VAE_BATCH_SIZE = 16
training_config.LDM_BATCH_SIZE = 16
training_config.LDM_DIFFUSION_TIMESTEPS = 50
training_config.SYNTHETIC_SAMPLES_PER_CLASS = 10
training_config.VAE_EWC_LAMBDA = 2000
training_config.LDM_EWC_LAMBDA = 100
training_config.VAE_KL_ANNEALING_RESTART = False

# Force patch pipeline module
pipeline_module.VAE_EPOCHS = 2
pipeline_module.LDM_EPOCHS = 2
pipeline_module.LDM_DIFFUSION_TIMESTEPS = 50
pipeline_module.SYNTHETIC_SAMPLES_PER_CLASS = 10
pipeline_module.VAE_EWC_LAMBDA = 2000
pipeline_module.LDM_EWC_LAMBDA = 100
pipeline_module.VAE_KL_ANNEALING_RESTART = False
pipeline_module.CLASSIFIER_EPOCHS = 2

def main():
    print("\nStarting Ultra-Fast Test...")
    
    # Run on just 2 patients (1 train, 1 test)
    results = main_cl_lopo_validation(
        data_root_dir=dataset_config.DatasetConfig.SIENA_ROOT_DIR,
        output_dir='./test_output/ultra_fast',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dataset_name='siena',
        max_patients=2
    )
    
    print("\nTest Complete!")
    return 0

if __name__ == "__main__":
    main()
