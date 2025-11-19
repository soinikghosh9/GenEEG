# GenEEG: Patient-Adaptive Epileptic Seizure Detection with VAE-LDM and CL-LOPO

**GenEEG** is a continual learning framework for automated seizure detection that combines neurophysiologically conditioned variational autoencoders (VAE) with latent diffusion models (LDM) to generate synthetic EEG data and adapt to individual patients in class-imbalanced settings.

<img width="1472" height="1172" alt="geneeg drawio (1)" src="https://github.com/user-attachments/assets/332292a8-961a-4417-9f1e-1a35f4f23de5" />

## Overview

Automated seizure detection faces critical challenges:
- Limited availability of clinical EEG data
- Severe class imbalance between seizure and non-seizure recordings
- High inter-patient variability
- Catastrophic forgetting in sequential multi-patient learning

GenEEG addresses these challenges through a hybrid continual learning approach that achieves **macro F1-scores of 0.84 (Siena) and 0.82 (CHB-MIT)**, representing a **15 percentage point improvement** over traditional oversampling baselines while maintaining ictal sensitivity above 75%.

## Key Features

### 1. Dual-Conditioned VAE-LDM Architecture
- **Clinical State Conditioning**: Binary labels (normal, preictal, ictal)
- **Neurophysiological Feature Conditioning**: 12 quantitative features including:
  - Delta/theta/alpha/beta/gamma band powers
  - Hjorth parameters (activity, mobility, complexity)
  - Sample entropy, line length, zero crossing rate
  - Kurtosis and skewness
- Precise control over synthetic EEG generation for improved classification

### 2. Continual Learning LOPO (CL-LOPO) Protocol
- Leave-one-patient-out validation with continual learning
- Fold-specific normalization to prevent test-set leakage
- Sequential patient adaptation instead of pooled training
- Memory efficient: **4.8 GB vs 12.4 GB** for pooled approaches

### 3. Hybrid Anti-Forgetting Strategy
- **Elastic Weight Consolidation (EWC)**: Protects important network parameters
- **Experience Replay (ER)**: Retains representative samples from previous patients
- Prevents catastrophic forgetting while adapting to new patients

### 4. Multi-Dataset Support
- **Siena Scalp EEG Database**: Adult population (14 patients)
- **CHB-MIT Scalp EEG Database**: Pediatric population (24 patients)
- Cross-dataset validation with 16-channel 10-20 montage

## Performance Highlights

| Dataset | Macro F1 | Ictal Sensitivity | Specificity | Improvement over Baseline |
|---------|----------|-------------------|-------------|---------------------------|
| Siena   | 0.84     | 76.3%            | 91.2%       | +15 pp (0.84 vs 0.69)    |
| CHB-MIT | 0.82     | 75.8%            | 89.7%       | +15 pp (0.82 vs 0.67)    |

**Key Results:**
- Consistent performance across diverse populations (adult and pediatric)
- Neurophysiological feature distributions in synthetic data closely match real recordings
- Effective mitigation of catastrophic forgetting in sequential learning

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 32GB+ RAM for full experiments

### Setup

1. **Clone or download the repository:**
```bash
cd GenEEG_Project
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. **Download datasets:**
   - Siena Scalp EEG Database: [PhysioNet](https://physionet.org/content/siena-scalp-eeg/3.0.0/)
   - CHB-MIT Scalp EEG Database: [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)

2. **Configure dataset paths** in `configs/dataset_config.py`:
```python
SIENA_ROOT_DIR = r"D:\Datasets\siena-scalp-eeg-database-3.0.0"
CHBMIT_ROOT_DIR = r"D:\Datasets\CHBMIT"
```

## Quick Start

### Running Full Experiments

**Siena Dataset:**
```bash
python run_siena_experiment.py
```

**CHB-MIT Dataset:**
```bash
python run_chbmit_experiment.py
```

These scripts run the complete CL-LOPO pipeline with optimized settings for production use.

### Quick Testing

For rapid prototyping and testing (reduced epochs):
```bash
python test_quick.py
```

### Using Main Entry Point

The `main.py` script provides flexible experiment configuration:

```bash
# Full continual learning (EWC + ER)
python main.py --dataset siena --experiment full_cl

# EWC ablation (no Experience Replay)
python main.py --dataset chbmit --experiment ewc_only

# ER ablation (no EWC)
python main.py --dataset siena --experiment er_only

# Pooled baseline (all patients together)
python main.py --dataset chbmit --experiment pooled_baseline

# Naive sequential (no anti-forgetting)
python main.py --dataset siena --experiment naive_sequential
```

## Running Tests

### All Tests
Run the complete test suite:
```bash
python tests/run_all_tests.py
```

### Individual Component Tests

**VAE Architecture:**
```bash
python tests/test_vae.py
```

**Latent Diffusion Model:**
```bash
python tests/test_ldm.py
```

**Building Blocks (Attention, Residual Blocks):**
```bash
python tests/test_building_blocks.py
```

**Classifier (EEGNet):**
```bash
python tests/test_classifier.py
```

**Data Loaders:**
```bash
python tests/test_chbmit_loader.py
```

**Skip Connections:**
```bash
python tests/test_skip_connections.py
```

## Project Structure

```
GenEEG_Project/
├── configs/              # Configuration files
│   ├── dataset_config.py # Dataset paths and preprocessing settings
│   ├── model_config.py   # Model architecture hyperparameters
│   └── training_config.py # Training hyperparameters and loss weights
├── data/                 # Data loading and preprocessing
│   ├── chbmit_loader.py  # CHB-MIT dataset loader
│   ├── siena_loader.py   # Siena dataset loader
│   └── preprocessing.py  # Common preprocessing utilities
├── models/               # Neural network architectures
│   ├── vae.py           # Variational autoencoder
│   ├── ldm.py           # Latent diffusion model
│   ├── eegnet.py        # EEGNet classifier
│   ├── classifier.py    # Classifier wrapper
│   └── building_blocks.py # Reusable components
├── training/             # Training procedures
│   ├── vae_trainer.py   # VAE training logic
│   ├── ldm_trainer.py   # LDM training logic
│   ├── classifier_trainer.py # Classifier training
│   ├── vae_losses.py    # Custom VAE loss functions
│   └── ewc_utils.py     # Elastic Weight Consolidation
├── pipeline/             # Continual learning pipeline
│   └── cl_lopo_pipeline.py # CL-LOPO implementation
├── evaluation/           # Metrics and visualization
│   ├── metrics.py       # Performance metrics
│   └── classifier_plots.py # Result visualization
├── utils/                # Utility functions
│   ├── feature_extraction.py # Neurophysiological features
│   ├── generation.py    # Synthetic data generation
│   ├── quality_filter.py # Sample quality filtering
│   └── logger.py        # Logging utilities
├── tests/                # Unit tests
├── outputs/              # Experiment results
│   ├── cl_lopo_siena/   # Siena results
│   └── cl_lopo_chbmit/  # CHB-MIT results
├── main.py              # Main entry point
├── run_siena_experiment.py    # Siena experiment script
├── run_chbmit_experiment.py   # CHB-MIT experiment script
└── test_quick.py        # Quick testing script
```

## Configuration and Tuning

### Dataset Configuration (`configs/dataset_config.py`)

**Key Parameters:**
```python
# Dataset selection
DATASET_NAME = 'siena'  # or 'chbmit'

# Preprocessing
TARGET_SFREQ = 256.0              # Sampling frequency
SEGMENT_SAMPLES = 3072            # 12 seconds at 256 Hz
BANDPASS_LOW = 0.5               # Hz
BANDPASS_HIGH = 70.0             # Hz
PREICTAL_DURATION_SEC = 20 * 60  # 20 minutes

# Channel selection
COMMON_EEG_CHANNELS = 16         # Number of channels
COMMON_CHANNEL_NAMES = [...]     # 10-20 system channels
```

**Tuning Recommendations:**
- **Segment Duration**: Increase `SEGMENT_SAMPLES` for longer temporal context (e.g., 5120 for 20s)
- **Preictal Window**: Adjust `PREICTAL_DURATION_SEC` based on clinical requirements (5-30 minutes)
- **Frequency Range**: Modify `BANDPASS_HIGH` based on noise characteristics (40-100 Hz)

### Model Configuration (`configs/model_config.py`)

**VAE Architecture:**
```python
VAE_LATENT_DIM = 128              # Latent space dimensionality
VAE_NUM_ENCODER_BLOCKS = 4        # Encoder depth
VAE_NUM_DECODER_BLOCKS = 4        # Decoder depth
VAE_BASE_CHANNELS = 64            # Initial channel count
VAE_CHANNEL_MULTIPLIERS = [1,2,4,8] # Channel scaling
```

**LDM Architecture:**
```python
LDM_DIM = 256                     # Hidden dimension
LDM_NUM_HEADS = 8                 # Attention heads
LDM_NUM_LAYERS = 6                # Transformer layers
LDM_DIFFUSION_TIMESTEPS = 1000    # Diffusion steps
```

**Classifier:**
```python
CLASSIFIER_TYPE = 'eegnet'        # or custom architecture
CLASSIFIER_DROPOUT = 0.5          # Regularization
```

**Tuning Recommendations:**
- **Latent Dimension**: Increase for complex datasets (128-512)
- **Model Depth**: Balance between capacity and overfitting (4-8 blocks)
- **Attention Heads**: Scale with `LDM_DIM` (typically `DIM / 32`)
- **Diffusion Steps**: Reduce for faster inference (250-500), increase for quality (1000-2000)

### Training Configuration (`configs/training_config.py`)

**Epochs (Production vs Testing):**
```python
# Quick testing (test_quick.py)
VAE_EPOCHS_DEFAULT = 20
LDM_EPOCHS_DEFAULT = 20

# Full experiments (run_siena/chbmit_experiment.py)
VAE_EPOCHS_PRODUCTION = 100
LDM_EPOCHS_PRODUCTION = 100
```

**VAE Training:**
```python
VAE_BATCH_SIZE = 64               # Adjust based on VRAM
VAE_LEARNING_RATE = 1e-4          # AdamW learning rate
VAE_L1_RECON_WEIGHT = 10.0        # Reconstruction loss weight
VAE_KL_WEIGHT = 5e-5              # KL divergence weight
VAE_KL_ANNEALING = True           # Enable KL warmup
VAE_KL_WARMUP_EPOCHS = 3          # Warmup duration
VAE_KL_ANNEAL_EPOCHS = 7          # Annealing duration
```

**LDM Training:**
```python
LDM_BATCH_SIZE = 64
LDM_LEARNING_RATE = 2e-4
LDM_MIN_SNR_GAMMA = 5.0           # SNR weighting
LDM_MOMENT_LOSS_WEIGHT = 1.0      # Mean/std preservation
```

**Continual Learning:**
```python
USE_EWC = True                    # Elastic Weight Consolidation
EWC_LAMBDA = 5000.0               # Fisher information weight
USE_EXPERIENCE_REPLAY = True      # Experience replay
ER_BUFFER_SIZE_PER_CLASS = 50     # Samples per class to retain
```

**Classifier Training:**
```python
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LEARNING_RATE = 1e-3
CLASSIFIER_EPOCHS = 50
CLASSIFIER_SYNTHETIC_SAMPLES = 100 # Per class
```

**Tuning Recommendations:**

1. **For Limited VRAM (<8GB):**
   - Reduce batch sizes: `VAE_BATCH_SIZE = 32`, `LDM_BATCH_SIZE = 32`
   - Reduce model size: `VAE_BASE_CHANNELS = 32`
   - Enable gradient checkpointing in code

2. **For Faster Training:**
   - Use production epochs only for final runs
   - Reduce synthetic samples: `CLASSIFIER_SYNTHETIC_SAMPLES = 50`
   - Decrease early stopping patience

3. **For Better Quality:**
   - Increase epochs: `VAE_EPOCHS = 200`, `LDM_EPOCHS = 200`
   - Adjust loss weights based on reconstruction quality
   - Increase `EWC_LAMBDA` for more forgetting resistance

4. **For New Datasets:**
   - Start with default settings
   - Monitor VAE reconstruction loss (target: <0.5)
   - Adjust `VAE_L1_RECON_WEIGHT` if reconstruction is poor
   - Tune `VAE_KL_WEIGHT` if latent space collapses
   - Validate neurophysiological features match real data

## Hardware Optimization

The codebase is optimized for:
- **CPU**: Intel i7-14700 (20 cores)
- **GPU**: NVIDIA RTX 4060 Ti (16GB VRAM)
- **RAM**: 32GB

**Environment Variables** (set in `main.py`):
```python
os.environ['NUMEXPR_MAX_THREADS'] = '20'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
```

Adjust these based on your system specifications.

## Output Organization

Results are saved in `outputs/` with the following structure:

```
outputs/
└── cl_lopo_{dataset}/
    └── run_{timestamp}_CL_LOPO/
        └── LOPO_fold_{i}_{test_patient}/
            ├── CL_step_{j}_{train_patient}/
            │   ├── vae_model.pt          # Trained VAE
            │   ├── ldm_model.pt          # Trained LDM
            │   ├── classifier_model.pt   # Trained classifier
            │   ├── metrics.json          # Performance metrics
            │   ├── training_curves.png   # Loss curves
            │   └── vae_analysis/        # VAE visualizations
            └── final_results.json        # Aggregated results
```

## Visualization

After training, visualize results:

```bash
python examples/visualize_all_results.py
```

This generates:
- Confusion matrices
- ROC curves
- Per-patient performance breakdowns
- Synthetic vs. real EEG comparisons
- Neurophysiological feature distributions

## Citation

If you use GenEEG in your research, please cite:

```bibtex
@article{geneeg2024,
  title={GenEEG: Epileptic EEG Detection through Patient-Adaptive Latent Diffusion and Continual Learning},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is provided for research purposes. Please refer to the LICENSE file for details.

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
- Reduce batch sizes in `training_config.py`
- Use gradient accumulation
- Enable mixed precision training

**2. Poor VAE Reconstruction:**
- Check `VAE_L1_RECON_WEIGHT` (increase if needed)
- Verify input normalization
- Monitor KL divergence (should be 2-10)

**3. NaN Losses:**
- Verify data preprocessing (no inf/nan values)
- Reduce learning rates
- Check gradient clipping is enabled

**4. Low Classification Performance:**
- Ensure sufficient synthetic samples
- Verify class balance in augmented data
- Check neurophysiological feature extraction

**5. Catastrophic Forgetting:**
- Increase `EWC_LAMBDA` (try 10000-50000)
- Increase experience replay buffer size
- Monitor per-patient performance over CL steps

## Contact and Support

For questions, issues, or contributions, please contact:

**Soinik Ghosh**  
Email: soinikghosh.rs.bme23@itbhu.ac.in

## Acknowledgments

This work utilizes:
- Siena Scalp EEG Database (PhysioNet)
- CHB-MIT Scalp EEG Database (PhysioNet)
- PyTorch deep learning framework
- MNE-Python for EEG processing

---

## Copyright

© 2024-2025 Soinik Ghosh. All rights reserved.

This software is provided for academic and research purposes only. Any commercial use, redistribution, or modification requires explicit written permission from the author. For any quaries contact soinikghosh.rs.bme23@itbhu.ac.in.
