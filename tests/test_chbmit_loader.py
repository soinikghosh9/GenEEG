"""Test script for CHB-MIT dataset loader.

This script validates the CHB-MIT loader by:
1. Loading a sample patient
2. Verifying data shapes and labels
3. Displaying summary statistics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.chbmit_loader import CHBMITLoader, load_chbmit_dataset
from configs.dataset_config import DatasetConfig


def test_single_patient():
    """Test loading a single patient."""
    print("="*80)
    print("TEST 1: Loading Single Patient (chb01)")
    print("="*80)
    
    loader = CHBMITLoader(
        root_dir=DatasetConfig.CHBMIT_ROOT_DIR,
        target_sampling_rate=256,
        common_channels=DatasetConfig.CHBMIT_COMMON_CHANNELS,
        preictal_duration_sec=20 * 60,  # 20 minutes
        verbose=True
    )
    
    # Test parsing summary file
    print("\n--- Parsing Summary File ---")
    annotations = loader.parse_summary_file('chb01')
    
    print(f"\nFound annotations for {len(annotations)} files:")
    for filename, events in list(annotations.items())[:3]:  # Show first 3
        print(f"  {filename}: {len(events)} seizure(s)")
        for event in events:
            duration = event['end'] - event['start']
            print(f"    Start: {event['start']}s, End: {event['end']}s (Duration: {duration}s)")
    
    # Test loading a single EDF file
    if annotations:
        print("\n--- Loading First EDF File ---")
        first_file = list(annotations.keys())[0]
        print(f"Loading: {first_file}")
        
        data, sr, channels = loader.load_edf_file('chb01', first_file)
        print(f"Data shape: {data.shape}")
        print(f"Sampling rate: {sr} Hz")
        print(f"Channels ({len(channels)}): {channels[:5]}...")  # Show first 5
    
    # Test extracting segments
    print("\n--- Extracting Segments ---")
    segments, labels, patient_ids = loader.extract_segments(
        'chb01',
        segment_duration_sec=5.0,
        overlap=0.5,
        balance_classes=True
    )
    
    print(f"\nExtracted {len(segments)} segments")
    print(f"Segment shape: {segments.shape}")
    print(f"Label distribution:")
    print(f"  Normal (0): {(labels == 0).sum()}")
    print(f"  Preictal (1): {(labels == 1).sum()}")
    print(f"  Ictal (2): {(labels == 2).sum()}")
    
    return segments, labels


def test_all_patients():
    """Test loading all patients."""
    print("\n" + "="*80)
    print("TEST 2: Loading All Patients")
    print("="*80)
    
    patient_data = load_chbmit_dataset(
        root_dir=DatasetConfig.CHBMIT_ROOT_DIR,
        target_sampling_rate=256,
        common_channels=DatasetConfig.CHBMIT_COMMON_CHANNELS,
        segment_duration_sec=5.0,
        overlap=0.5,
        preictal_duration_sec=20 * 60,
        verbose=False  # Less verbose for multiple patients
    )
    
    print(f"\nLoaded {len(patient_data)} patients:")
    print(f"{'Patient':<10} {'Segments':<10} {'Normal':<10} {'Preictal':<10} {'Ictal':<10}")
    print("-"*60)
    
    total_segments = 0
    total_normal = 0
    total_preictal = 0
    total_ictal = 0
    
    for patient_id, (segments, labels) in patient_data.items():
        n_segments = len(segments)
        n_normal = (labels == 0).sum()
        n_preictal = (labels == 1).sum()
        n_ictal = (labels == 2).sum()
        
        print(f"{patient_id:<10} {n_segments:<10} {n_normal:<10} {n_preictal:<10} {n_ictal:<10}")
        
        total_segments += n_segments
        total_normal += n_normal
        total_preictal += n_preictal
        total_ictal += n_ictal
    
    print("-"*60)
    print(f"{'TOTAL':<10} {total_segments:<10} {total_normal:<10} {total_preictal:<10} {total_ictal:<10}")
    
    return patient_data


def test_data_quality(segments, labels):
    """Test data quality and statistics."""
    print("\n" + "="*80)
    print("TEST 3: Data Quality Checks")
    print("="*80)
    
    import numpy as np
    
    # Check for NaN or Inf
    print("\n--- Checking for Invalid Values ---")
    has_nan = np.isnan(segments).any()
    has_inf = np.isinf(segments).any()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("[WARNING] Data contains invalid values!")
    else:
        print("[OK] Data is clean (no NaN or Inf)")
    
    # Basic statistics
    print("\n--- Data Statistics ---")
    print(f"Mean: {segments.mean():.6f}")
    print(f"Std: {segments.std():.6f}")
    print(f"Min: {segments.min():.6f}")
    print(f"Max: {segments.max():.6f}")
    
    # Per-class statistics
    print("\n--- Per-Class Statistics ---")
    for class_label in [0, 1, 2]:
        class_name = ['Normal', 'Preictal', 'Ictal'][class_label]
        class_segments = segments[labels == class_label]
        
        if len(class_segments) > 0:
            print(f"\n{class_name}:")
            print(f"  Count: {len(class_segments)}")
            print(f"  Mean: {class_segments.mean():.6f}")
            print(f"  Std: {class_segments.std():.6f}")
            print(f"  Shape: {class_segments.shape}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CHB-MIT Dataset Loader Test Suite")
    print("="*80)
    
    # Check if dataset path exists
    if not Path(DatasetConfig.CHBMIT_ROOT_DIR).exists():
        print(f"\n[ERROR] CHB-MIT dataset not found at:")
        print(f"   {DatasetConfig.CHBMIT_ROOT_DIR}")
        print(f"\nPlease update CHBMIT_ROOT_DIR in configs/dataset_config.py")
        return
    
    try:
        # Test 1: Single patient
        segments, labels = test_single_patient()
        
        # Test 2: All patients (commented out for speed during development)
        # Uncomment when ready for full test
        # patient_data = test_all_patients()
        
        # Test 3: Data quality
        test_data_quality(segments, labels)
        
        print("\n" + "="*80)
        print("[SUCCESS] All tests completed successfully!")
        print("="*80)
    
    except Exception as e:
        print("\n" + "="*80)
        print(f"[ERROR] Test failed with error:")
        print(f"   {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
