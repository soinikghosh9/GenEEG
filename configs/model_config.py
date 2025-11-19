"""Model architecture configuration for GenEEG."""


class ModelConfig:
    """Configuration for all model architectures."""
    
    # ========================================
    # VAE CONFIGURATION
    # ========================================
    # Architecture
    VAE_IN_CHANNELS = 16  # Number of EEG channels (16 common channels)
    VAE_BASE_CHANNELS = 128
    VAE_LATENT_CHANNELS = 64
    VAE_CH_MULTS = (1, 2, 4, 8)
    VAE_BLOCKS_PER_STAGE = 2
    VAE_USE_FEATURE_COND = True
    VAE_FEATURE_COND_DIM = 12
    
    # Legacy lowercase (for backward compatibility)
    vae_in_channels = VAE_IN_CHANNELS
    vae_base_channels = VAE_BASE_CHANNELS
    vae_latent_channels = VAE_LATENT_CHANNELS
    vae_ch_mults = VAE_CH_MULTS
    vae_blocks_per_stage = VAE_BLOCKS_PER_STAGE
    vae_use_feature_cond = VAE_USE_FEATURE_COND
    vae_feature_cond_dim = VAE_FEATURE_COND_DIM
    
    # Legacy parameters (for backward compatibility)
    VAE_LATENT_DIM = 128
    VAE_LATENT_LENGTH = 192  # Latent temporal length after encoding
    VAE_ENCODER_CHANNELS = [16, 32, 64, 128]  # Adjusted to 16 input channels
    VAE_DECODER_CHANNELS = [128, 64, 32, 16]  # Adjusted to 16 output channels
    VAE_USE_SPECTRAL_NORM = True
    VAE_USE_RESIDUAL = True
    
    # VAE Loss Weights (11 components)
    VAE_L1_WEIGHT = 9.0
    VAE_STFT_WEIGHT = 10.0
    VAE_DWT_WEIGHT = 3.0
    VAE_SHARPNESS_WEIGHT = 4.0
    VAE_LOW_FREQ_WEIGHT = 5.0
    VAE_HIGH_FREQ_WEIGHT = 6.0
    VAE_KL_WEIGHT = 0.01
    VAE_PERCEPTUAL_WEIGHT = 2.0
    VAE_SEIZURE_FIDELITY_WEIGHT = 7.0
    VAE_SPECTRAL_CONVERGENCE_WEIGHT = 3.0
    VAE_PHASE_CONSISTENCY_WEIGHT = 2.0
    
    # ========================================
    # LATENT DIFFUSION MODEL CONFIGURATION
    # ========================================
    # Architecture
    LDM_LATENT_CHANNELS = 64  # Must match VAE latent channels
    LDM_BASE_CHANNELS = 256
    LDM_TIME_EMB_DIM = 256
    LDM_NUM_CLASSES = 3
    LDM_CHANNEL_MULTS = (1, 2, 4, 8)
    LDM_NUM_RES_BLOCKS = 2
    LDM_USE_CROSS_ATTENTION = True
    LDM_USE_ADAGN = True
    LDM_CONTEXT_DIM = 256
    LDM_FEATURE_COND_DIM = 12
    LDM_USE_FEATURE_COND = True
    
    # Legacy lowercase (for backward compatibility)
    ldm_latent_channels = LDM_LATENT_CHANNELS
    ldm_base_unet_channels = LDM_BASE_CHANNELS
    ldm_time_emb_dim = LDM_TIME_EMB_DIM
    ldm_num_classes = LDM_NUM_CLASSES
    ldm_channel_mults = LDM_CHANNEL_MULTS
    ldm_num_res_blocks = LDM_NUM_RES_BLOCKS
    ldm_use_cross_attention = LDM_USE_CROSS_ATTENTION
    ldm_use_adagn = LDM_USE_ADAGN
    ldm_context_dim = LDM_CONTEXT_DIM
    ldm_feature_cond_dim = LDM_FEATURE_COND_DIM
    ldm_use_feature_cond = LDM_USE_FEATURE_COND
    
    # Diffusion Schedule
    LDM_TIMESTEPS = 1000
    LDM_BETA_SCHEDULE = 'cosine'  # 'cosine' or 'linear'
    LDM_PREDICTION_TYPE = 'epsilon'  # 'epsilon', 'sample', or 'v_prediction'
    
    # Classifier-Free Guidance
    LDM_CFG_SCALE = 5.0  # Guidance scale for sampling
    LDM_CFG_DROPOUT = 0.1  # Probability of unconditional training
    
    # Legacy parameters (for backward compatibility)
    LDM_UNET_CHANNELS = 128
    LDM_UNET_CHANNEL_MULT = [1, 2, 4, 8]
    LDM_UNET_NUM_RES_BLOCKS = 2
    LDM_UNET_ATTENTION_RESOLUTIONS = [4, 8]
    LDM_UNET_DROPOUT = 0.1
    
    # Conditioning
    LDM_NUM_CLASSES = 3
    LDM_CLASS_EMBED_DIM = 256
    LDM_NEURO_FEATURE_DIM = 12  # Number of neurophysiological features
    LDM_NEURO_EMBED_DIM = 256
    LDM_USE_DUAL_CONDITIONING = True
    
    # Auxiliary Losses
    LDM_USE_SEIZURE_FIDELITY_LOSS = True
    LDM_SEIZURE_LOSS_WEIGHT = 0.5
    LDM_USE_RAW_DOMAIN_LOSS = True
    LDM_RAW_LOSS_WEIGHT = 0.3
    LDM_RAW_LOSS_FREQUENCY = 4  # Apply every N steps
    
    # ========================================
    # CLASSIFIER CONFIGURATION (CNN-BiLSTM)
    # ========================================
    CLASSIFIER_CNN_FILTERS = [32, 64, 128]
    CLASSIFIER_CNN_KERNEL_SIZE = 3
    CLASSIFIER_CNN_POOL_SIZE = 2
    CLASSIFIER_LSTM_HIDDEN = 128
    CLASSIFIER_LSTM_LAYERS = 2
    CLASSIFIER_DROPOUT = 0.3
    CLASSIFIER_NUM_CLASSES = 3
    
    # ========================================
    # K-MEANS PROTOTYPE CONFIGURATION
    # ========================================
    KMEANS_NUM_PROTOTYPES = 10  # Per class
    KMEANS_USE_WEIGHTED_AVERAGE = True
    
    # ========================================
    # ABLATION STUDY FLAGS
    # ========================================
    RUN_VAE_LOSS_ABLATION = False  # Test removing individual loss components
    RUN_MINIMAL_VAE = False  # Use only L1 + STFT + KL (3 components)
    RUN_LOSS_WEIGHT_SENSITIVITY = False  # Test ±20% weight variations
