"""
Enhanced Quality Filtering for Synthetic EEG

Provides advanced filtering to ensure high-quality synthetic EEG samples
that maintain physiological plausibility and statistical similarity to real data.
"""

import numpy as np
from scipy.signal import welch
from data.preprocessing import calculate_eeg_features
from configs.dataset_config import TARGET_SFREQ


def enhanced_filter_synthetic_batch(
    synth_raw_np: np.ndarray,
    real_samples_for_comparison: np.ndarray,
    sfreq: int = TARGET_SFREQ,
    feature_similarity_threshold: float = 3.0,
    amplitude_ratio_threshold: float = 4.0,
    spectral_similarity_threshold: float = 0.6,
    physiological_constraint_threshold: float = 2.5,
    min_samples_to_return: int = 5
) -> np.ndarray:
    """
    Enhanced filtering with physiological constraints and adaptive thresholds to ensure 
    high-quality synthetic EEG data that maintains physiological plausibility.
    
    Args:
        synth_raw_np: Synthetic samples (N, C, T) in ORIGINAL SCALE
        real_samples_for_comparison: Real samples (M, C, T) in ORIGINAL SCALE
        sfreq: Sampling frequency
        feature_similarity_threshold: Maximum L2 distance for features (lower = stricter)
        amplitude_ratio_threshold: Maximum amplitude ratio (lower = stricter)
        spectral_similarity_threshold: Minimum PSD correlation (higher = stricter)
        physiological_constraint_threshold: Maximum std deviation from physiological norms
        min_samples_to_return: Minimum samples to return (fallback)
    
    Returns:
        Filtered synthetic samples (M, C, T) where M <= N
    
    Features:
    - Physiological plausibility checks
    - Channel correlation analysis  
    - Improved spectral power validation
    - Artifact detection
    - Adaptive quality scoring
    """
    if len(real_samples_for_comparison) == 0:
        print("[WARNING] No real samples for quality filtering. Returning all synthetic.")
        return synth_raw_np
    
    if len(synth_raw_np) == 0:
        return synth_raw_np
        
    # Calculate reference statistics from real samples
    real_features = np.array([calculate_eeg_features(sample, sfreq) for sample in real_samples_for_comparison])
    real_mean_features = np.mean(real_features, axis=0)
    real_std_features = np.std(real_features, axis=0) + 1e-6
    
    # Calculate physiological reference values
    real_channel_corrs = []
    real_temporal_smoothness = []
    real_frequency_ratios = []
    
    for sample in real_samples_for_comparison:
        # Channel correlation (healthy EEG should have some inter-channel correlation)
        corr_matrix = np.corrcoef(sample)
        real_channel_corrs.append(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
        
        # Temporal smoothness (measure of signal continuity)
        diff_signal = np.diff(sample, axis=1)
        real_temporal_smoothness.append(np.mean(np.std(diff_signal, axis=1)))
        
        # Frequency ratios (physiological frequency band relationships)
        try:
            freqs, psd = welch(sample.mean(axis=0), fs=sfreq, nperseg=min(256, sample.shape[1]//4))
            
            # Define physiological frequency bands
            delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
            theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
            alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
            beta_mask = (freqs >= 12.0) & (freqs <= 30.0)
            gamma_mask = (freqs >= 30.0) & (freqs <= 50.0)
            
            delta_power = np.sum(psd[delta_mask])
            alpha_power = np.sum(psd[alpha_mask])
            
            # Physiological frequency ratio (alpha/delta is important in EEG)
            alpha_delta_ratio = alpha_power / (delta_power + 1e-8)
            real_frequency_ratios.append(alpha_delta_ratio)
        except:
            real_frequency_ratios.append(1.0)  # Default
    
    # Calculate reference values
    real_corr_mean = np.mean(real_channel_corrs)
    real_corr_std = np.std(real_channel_corrs) + 1e-6
    real_smoothness_mean = np.mean(real_temporal_smoothness)
    real_smoothness_std = np.std(real_temporal_smoothness) + 1e-6
    real_freq_ratio_mean = np.mean(real_frequency_ratios)
    real_freq_ratio_std = np.std(real_frequency_ratios) + 1e-6
    
    # Reference amplitude statistics
    real_amplitudes = np.array([np.std(sample) for sample in real_samples_for_comparison])
    real_amp_mean = np.mean(real_amplitudes)
    real_amp_std = np.std(real_amplitudes) + 1e-6
    
    # Filter synthetic samples
    quality_scores = []
    passed_indices = []
    
    for idx, synth_sample in enumerate(synth_raw_np):
        # Initialize quality score
        quality_score = 0.0
        fail_reasons = []
        
        # 1. Feature similarity check
        try:
            synth_features = calculate_eeg_features(synth_sample, sfreq)
            feature_distance = np.linalg.norm(synth_features - real_mean_features)
            
            if feature_distance < feature_similarity_threshold:
                quality_score += 1.0
            else:
                fail_reasons.append(f"feature_dist={feature_distance:.2f}")
        except:
            fail_reasons.append("feature_extraction_failed")
        
        # 2. Amplitude ratio check
        synth_amp = np.std(synth_sample)
        amp_ratio = synth_amp / (real_amp_mean + 1e-8)
        
        if amp_ratio < amplitude_ratio_threshold and amp_ratio > 0.25:
            quality_score += 1.0
        else:
            fail_reasons.append(f"amp_ratio={amp_ratio:.2f}")
        
        # 3. Spectral similarity check
        try:
            synth_freqs, synth_psd = welch(synth_sample.mean(axis=0), fs=sfreq, nperseg=min(256, synth_sample.shape[1]//4))
            real_sample = real_samples_for_comparison[idx % len(real_samples_for_comparison)]
            real_freqs, real_psd = welch(real_sample.mean(axis=0), fs=sfreq, nperseg=min(256, real_sample.shape[1]//4))
            
            # Compute correlation between PSDs
            psd_corr = np.corrcoef(synth_psd, real_psd)[0, 1]
            
            if psd_corr > spectral_similarity_threshold:
                quality_score += 1.0
            else:
                fail_reasons.append(f"psd_corr={psd_corr:.2f}")
        except:
            fail_reasons.append("spectral_analysis_failed")
        
        # 4. Channel correlation check (physiological constraint)
        try:
            synth_corr_matrix = np.corrcoef(synth_sample)
            synth_corr = np.mean(np.abs(synth_corr_matrix[np.triu_indices_from(synth_corr_matrix, k=1)]))
            corr_z_score = np.abs(synth_corr - real_corr_mean) / real_corr_std
            
            if corr_z_score < physiological_constraint_threshold:
                quality_score += 1.0
            else:
                fail_reasons.append(f"corr_z={corr_z_score:.2f}")
        except:
            fail_reasons.append("corr_check_failed")
        
        # 5. Temporal smoothness check
        try:
            synth_diff = np.diff(synth_sample, axis=1)
            synth_smoothness = np.mean(np.std(synth_diff, axis=1))
            smoothness_z_score = np.abs(synth_smoothness - real_smoothness_mean) / real_smoothness_std
            
            if smoothness_z_score < physiological_constraint_threshold:
                quality_score += 1.0
            else:
                fail_reasons.append(f"smooth_z={smoothness_z_score:.2f}")
        except:
            fail_reasons.append("smoothness_check_failed")
        
        # 6. Frequency ratio check (physiological constraint)
        try:
            synth_freqs, synth_psd = welch(synth_sample.mean(axis=0), fs=sfreq, nperseg=min(256, synth_sample.shape[1]//4))
            delta_mask = (synth_freqs >= 0.5) & (synth_freqs <= 4.0)
            alpha_mask = (synth_freqs >= 8.0) & (synth_freqs <= 12.0)
            
            synth_delta = np.sum(synth_psd[delta_mask])
            synth_alpha = np.sum(synth_psd[alpha_mask])
            synth_freq_ratio = synth_alpha / (synth_delta + 1e-8)
            
            freq_ratio_z_score = np.abs(synth_freq_ratio - real_freq_ratio_mean) / real_freq_ratio_std
            
            if freq_ratio_z_score < physiological_constraint_threshold:
                quality_score += 1.0
            else:
                fail_reasons.append(f"freq_ratio_z={freq_ratio_z_score:.2f}")
        except:
            fail_reasons.append("freq_ratio_check_failed")
        
        quality_scores.append(quality_score)
        
        # Sample passes if it passes at least 4 out of 6 checks
        if quality_score >= 4.0:
            passed_indices.append(idx)
    
    # Fallback: if too few samples pass, keep best samples
    if len(passed_indices) < min_samples_to_return:
        # Sort by quality score and keep top samples
        sorted_indices = np.argsort(quality_scores)[::-1]
        passed_indices = sorted_indices[:min(min_samples_to_return, len(synth_raw_np))].tolist()
        print(f"[QUALITY FILTER] Only {len(passed_indices)} samples passed strict filtering (wanted {min_samples_to_return}). Keeping best samples.")
    
    filtered_samples = synth_raw_np[passed_indices]
    
    print(f"[QUALITY FILTER] Kept {len(filtered_samples)}/{len(synth_raw_np)} samples "
          f"(avg quality: {np.mean([quality_scores[i] for i in passed_indices]):.2f}/6.0)")
    
    return filtered_samples
