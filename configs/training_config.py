"""Training configuration for GenEEG.

Hardware Optimized for:
- CPU: Intel i7-14700 (20 cores: 8P + 12E)
- RAM: 32GB DDR4/DDR5
- GPU: NVIDIA RTX 4060 Ti (16GB VRAM, Ada Lovelace)
"""

# Global seed value for reproducibility
SEED_VALUE = 42


class TrainingConfig:
    """Configuration for training procedures."""
    
    # ========================================
    # EPOCH CONFIGURATION (different for testing vs full training)
    # ========================================
    # These will be overridden by run scripts:
    # - test_quick.py: Uses these default values (2 epochs for ultra-fast testing)
    # - run_siena_experiment.py / run_chb_experiment.py: Uses 100 epochs for production
    VAE_EPOCHS_DEFAULT = 2  # Ultra-fast testing mode (was 20)
    LDM_EPOCHS_DEFAULT = 2  # Ultra-fast testing mode (was 20)
    VAE_EPOCHS_PRODUCTION = 100  # Full training (Siena/CHB-MIT)
    LDM_EPOCHS_PRODUCTION = 100  # Full training (Siena/CHB-MIT)
    
    # ========================================
    # VAE TRAINING (Optimized for RTX 4060 Ti 16GB)
    # ========================================
    VAE_BATCH_SIZE = 64  # Increased from 32 - 16GB VRAM can handle larger batches
    VAE_EPOCHS = VAE_EPOCHS_DEFAULT  # Will be overridden by run scripts
    # FIX 1: Increased from 5e-6 to 1e-4 (standard AdamW LR) for proper learning
    VAE_LEARNING_RATE = 1e-4  # Was 5e-6 (TOO LOW - model couldn't learn)
    VAE_WEIGHT_DECAY = 1e-5
    VAE_OPTIMIZER = 'AdamW'
    VAE_SCHEDULER = 'CosineAnnealingLR'
    VAE_EARLY_STOPPING_PATIENCE = 15
    
    # ========================================
    # VAE LOSS WEIGHTS (OPTIMIZED - STABLE 4-LOSS SETUP)
    # ========================================
    # STABLE: Only 4 proven losses - L1, KL, Feature Recon, Sharpness
    # FIX 4: Increased L1 weight from 1.0 to 10.0 for stronger reconstruction signal
    VAE_L1_RECON_WEIGHT = 10.0  # Was 1.0 (TOO WEAK) - Main reconstruction loss - ESSENTIAL
    VAE_DWT_LOSS_WEIGHT = 0.0  # DISABLED
    VAE_CONNECTIVITY_LOSS_WEIGHT = 0.0  # DISABLED
    VAE_MULTI_SCALE_STFT_WEIGHT = 0.0  # DISABLED - was causing instability
    VAE_FEATURE_RECON_LOSS_WEIGHT = 0.1  # ENABLED - neurophysiological features
    VAE_MINORITY_FOCUS_LOSS_WEIGHT = 0.0  # DISABLED
    # FIX 3: Increased sharpness weight from 0.5 to 2.0 after improving loss function
    VAE_SHARPNESS_LOSS_WEIGHT = 2.0  # Was 0.5 - ENABLED - preserves spikes (improved loss)
    VAE_LOW_FREQ_PRESERVATION_WEIGHT = 0.0  # DISABLED
    VAE_HIGH_FREQ_PRESERVATION_WEIGHT = 0.0  # DISABLED - was degrading reconstruction
    VAE_SPECTRAL_ENVELOPE_WEIGHT = 0.0  # DISABLED - new loss removed
    VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT = 0.0  # DISABLED
    VAE_CONTRASTIVE_WEIGHT = 0.0  # DISABLED
    VAE_FREQ_BAND_LOSS_WEIGHT = 0.0  # DISABLED
    
    # ========================================
    # VAE KL DIVERGENCE CONFIGURATION
    # ========================================
    VAE_KL_WEIGHT = 5e-5  # Final target KL weight
    VAE_USE_KL_ANNEALING = True  # Enable KL annealing
    # FIX 2: Reduced KL annealing duration - was warmup=5+anneal=15 (too long for 20 epochs)
    VAE_KL_WARMUP_EPOCHS = 3  # Was 5 - Quick warmup to let decoder stabilize
    VAE_KL_ANNEAL_EPOCHS = 7  # Was 15 - Ramp from 0 to target over epochs 3-10 (50% of training)
    VAE_KLD_CLAMP_MAX = 5.0  # Hard cap on KLD loss per sample
    VAE_FREE_BITS_THRESHOLD = 2.0  # KLD slack before penalizing
    
    # ========================================
    # GRADIENT AND STABILITY CONFIGURATION
    # ========================================
    GRADIENT_CLIP_VAL = 1.0  # Increased from 0.3 - proper init should prevent explosion
    VAE_LOSS_FFT_SIZE = 1024
    FREQ_BANDS_FOR_LOSS = {
        "delta": (0.5, 4.0),
        "alpha_beta": (8.0, 30.0),
        "gamma_low": (30.0, 50.0),
        "gamma_high": (50.0, 100.0)
    }

    
    # ========================================
    # LDM TRAINING (Optimized for RTX 4060 Ti 16GB)
    # ========================================
    LDM_BATCH_SIZE = 64  # Increased from 32 - 16GB VRAM allows larger batches
    LDM_EPOCHS = LDM_EPOCHS_DEFAULT  # Will be overridden by run scripts
    LDM_LEARNING_RATE = 2e-4
    LDM_MIN_LEARNING_RATE = 1e-6
    LDM_WEIGHT_DECAY = 1e-5
    LDM_OPTIMIZER = 'AdamW'
    LDM_SCHEDULER = 'CosineAnnealingLR'
    LDM_EARLY_STOPPING_PATIENCE = 20
    LDM_DIFFUSION_TIMESTEPS = 1000
    LDM_LOSS_TARGET = 'v'  # 'v' for v-prediction (better than eps)
    LDM_MIN_SNR_GAMMA = 5.0  # Increased from 2.0 - better sample quality at high noise
    LDM_P2_K = 1.0  # P2 weighting parameter k
    LDM_P2_GAMMA = 0.7  # Reduced from 1.0 - less aggressive weighting
    LDM_MOMENT_LOSS_WEIGHT = 1.0  # FIXED: Increased from 0.05 - critical for mean/std stability
    LDM_LOW_FREQ_PRESERVATION_WEIGHT = 0.05  # FIXED: Back to original value for low-freq preservation
    LDM_SEIZURE_LOSS_WEIGHT = 0.05  # FIXED: Back to original value for seizure-specific features
    LDM_AUX_EVERY = 4  # FIXED: Back to 4 - more frequent auxiliary loss application
    LDM_AUX_BATCH_FRAC = 0.5  # Fraction of batch for auxiliary losses
    LDM_AUX_HUBER_DELTA = 1.0  # Huber loss delta for auxiliary losses
    LDM_AUX_CAP = 10.0  # Cap for auxiliary losses
    LDM_AUX_WARMUP_STEPS = 10000  # FIXED: Back to 10000 - proper warmup duration
    LDM_GRAD_CLIP = 1.0  # Gradient clipping value
    LDM_SCALING_FACTOR = 1.0  # Latent scaling factor (set dynamically from data)
    LDM_EMA_DECAY = 0.9995  # EMA decay rate
    LDM_LR_WARMUP_STEPS = 5000  # FIXED: Back to 5000 - proper warmup
    LDM_USE_LOGNORMAL_T = True  # Use log-normal timestep sampling (better distribution)
    
    # ========================================
    # CLASSIFIER TRAINING (Optimized for RTX 4060 Ti 16GB)
    # ========================================
    CLASSIFIER_BATCH_SIZE = 128  # Increased from 64 - classifier is lighter than VAE/LDM
    CLASSIFIER_EPOCHS = 50
    CLASSIFIER_LEARNING_RATE = 1e-4
    CLASSIFIER_WEIGHT_DECAY = 1e-5
    CLASSIFIER_OPTIMIZER = 'AdamW'
    CLASSIFIER_EARLY_STOPPING_PATIENCE = 10
    
    # ========================================
    # FINETUNING CONFIGURATION (for continual learning)
    # ========================================
    FINETUNE_EPOCHS = 20  # Reduced epochs when fine-tuning on new patients
    FINETUNE_LR_FACTOR = 2.0  # Reduce learning rate by this factor during finetuning
    FINETUNE_BATCH_SIZE = 32
    FINETUNE_LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
    
    # ========================================
    # CONTINUAL LEARNING CONFIGURATION
    # ========================================
    USE_EWC = True  # Elastic Weight Consolidation
    USE_EXPERIENCE_REPLAY = True  # Experience Replay
    
    EWC_LAMBDA = 5000  # EWC regularization strength
    EXPERIENCE_REPLAY_RATIO = 0.3  # Proportion of replay samples
    EXPERIENCE_REPLAY_BUFFER_SIZE_PER_PATIENT = 500  # Samples to store per patient
    
    # ========================================
    # VALIDATION CONFIGURATION
    # ========================================
    VALIDATION_SCHEME = 'lopo'  # 'lopo' (Leave-One-Patient-Out) or 'kfold'
    KFOLD_NUM_FOLDS = 5  # If using k-fold instead of LOPO
    
    # ========================================
    # DATA AUGMENTATION
    # ========================================
    SYNTHETIC_SAMPLES_PER_CLASS = 1000
    AUGMENTATION_RATIO = 0.5  # Ratio of synthetic to real samples
    USE_SYNTHETIC_AUGMENTATION = True
    
    # ========================================
    # DEVICE CONFIGURATION
    # ========================================
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 8  # DataLoader workers (optimized for i7-14700)
    PIN_MEMORY = True
    
    # ========================================
    # ABLATION EXPERIMENTS
    # ========================================
    RUN_EWC_ONLY_ABLATION = False  # Disable Experience Replay
    RUN_ER_ONLY_ABLATION = False  # Disable EWC
    RUN_POOLED_BASELINE = False  # Train on all patients simultaneously
    RUN_NAIVE_SEQUENTIAL = False  # No CL techniques (for catastrophic forgetting demo)
    
    # ========================================
    # REPORTING CONFIGURATION
    # ========================================
    REPORT_PERCLASS_METRICS = True  # Report per-class F1/Precision/Recall
    REPORT_PERPATIENT_VARIANCE = True  # Report mean ± SD across patients
    COMPUTE_CONFIDENCE_INTERVALS = True  # 95% CI
    SAVE_CONFUSION_MATRICES = True
    
    # ========================================
    # REPRODUCIBILITY
    # ========================================
    RANDOM_SEED = 42
    DETERMINISTIC = True  # Use deterministic algorithms (slower but reproducible)
    
    @classmethod
    def get_ewc_enabled(cls) -> bool:
        """Check if EWC should be enabled based on ablation settings."""
        if cls.RUN_ER_ONLY_ABLATION or cls.RUN_POOLED_BASELINE or cls.RUN_NAIVE_SEQUENTIAL:
            return False
        return cls.USE_EWC
    
    @classmethod
    def get_er_enabled(cls) -> bool:
        """Check if Experience Replay should be enabled based on ablation settings."""
        if cls.RUN_EWC_ONLY_ABLATION or cls.RUN_POOLED_BASELINE or cls.RUN_NAIVE_SEQUENTIAL:
            return False
        return cls.USE_EXPERIENCE_REPLAY
    
    @classmethod
    def get_experiment_name(cls) -> str:
        """Get descriptive name for current experiment configuration."""
        if cls.RUN_POOLED_BASELINE:
            return "pooled_baseline"
        elif cls.RUN_NAIVE_SEQUENTIAL:
            return "naive_sequential"
        elif cls.RUN_EWC_ONLY_ABLATION:
            return "ewc_only"
        elif cls.RUN_ER_ONLY_ABLATION:
            return "er_only"
        elif cls.USE_EWC and cls.USE_EXPERIENCE_REPLAY:
            return "full_cl_ewc_er"
        else:
            return "standard"


# Export constants for easy access
VAE_L1_RECON_WEIGHT = TrainingConfig.VAE_L1_RECON_WEIGHT
VAE_DWT_LOSS_WEIGHT = TrainingConfig.VAE_DWT_LOSS_WEIGHT
VAE_CONNECTIVITY_LOSS_WEIGHT = TrainingConfig.VAE_CONNECTIVITY_LOSS_WEIGHT
VAE_MULTI_SCALE_STFT_WEIGHT = TrainingConfig.VAE_MULTI_SCALE_STFT_WEIGHT
VAE_FEATURE_RECON_LOSS_WEIGHT = TrainingConfig.VAE_FEATURE_RECON_LOSS_WEIGHT
VAE_MINORITY_FOCUS_LOSS_WEIGHT = TrainingConfig.VAE_MINORITY_FOCUS_LOSS_WEIGHT
VAE_SHARPNESS_LOSS_WEIGHT = TrainingConfig.VAE_SHARPNESS_LOSS_WEIGHT
VAE_LOW_FREQ_PRESERVATION_WEIGHT = TrainingConfig.VAE_LOW_FREQ_PRESERVATION_WEIGHT
VAE_HIGH_FREQ_PRESERVATION_WEIGHT = TrainingConfig.VAE_HIGH_FREQ_PRESERVATION_WEIGHT
VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT = TrainingConfig.VAE_EDGE_ARTIFACT_SUPPRESSION_WEIGHT
VAE_CONTRASTIVE_WEIGHT = TrainingConfig.VAE_CONTRASTIVE_WEIGHT
VAE_KL_WEIGHT = TrainingConfig.VAE_KL_WEIGHT
VAE_USE_KL_ANNEALING = TrainingConfig.VAE_USE_KL_ANNEALING
VAE_KL_WARMUP_EPOCHS = TrainingConfig.VAE_KL_WARMUP_EPOCHS
VAE_KL_ANNEAL_EPOCHS = TrainingConfig.VAE_KL_ANNEAL_EPOCHS
VAE_KLD_CLAMP_MAX = TrainingConfig.VAE_KLD_CLAMP_MAX
VAE_FREE_BITS_THRESHOLD = TrainingConfig.VAE_FREE_BITS_THRESHOLD
GRADIENT_CLIP_VAL = TrainingConfig.GRADIENT_CLIP_VAL

# LDM constants
LDM_DIFFUSION_TIMESTEPS = TrainingConfig.LDM_DIFFUSION_TIMESTEPS
LDM_LOSS_TARGET = TrainingConfig.LDM_LOSS_TARGET
LDM_MIN_SNR_GAMMA = TrainingConfig.LDM_MIN_SNR_GAMMA
LDM_P2_K = TrainingConfig.LDM_P2_K
LDM_P2_GAMMA = TrainingConfig.LDM_P2_GAMMA
LDM_MOMENT_LOSS_WEIGHT = TrainingConfig.LDM_MOMENT_LOSS_WEIGHT
LDM_LOW_FREQ_PRESERVATION_WEIGHT = TrainingConfig.LDM_LOW_FREQ_PRESERVATION_WEIGHT
LDM_SEIZURE_LOSS_WEIGHT = TrainingConfig.LDM_SEIZURE_LOSS_WEIGHT
LDM_AUX_EVERY = TrainingConfig.LDM_AUX_EVERY
LDM_AUX_BATCH_FRAC = TrainingConfig.LDM_AUX_BATCH_FRAC
LDM_AUX_HUBER_DELTA = TrainingConfig.LDM_AUX_HUBER_DELTA
LDM_AUX_CAP = TrainingConfig.LDM_AUX_CAP
LDM_AUX_WARMUP_STEPS = TrainingConfig.LDM_AUX_WARMUP_STEPS
LDM_GRAD_CLIP = TrainingConfig.LDM_GRAD_CLIP
LDM_SCALING_FACTOR = TrainingConfig.LDM_SCALING_FACTOR
LDM_EMA_DECAY = TrainingConfig.LDM_EMA_DECAY
LDM_LR_WARMUP_STEPS = TrainingConfig.LDM_LR_WARMUP_STEPS
LDM_USE_LOGNORMAL_T = TrainingConfig.LDM_USE_LOGNORMAL_T
