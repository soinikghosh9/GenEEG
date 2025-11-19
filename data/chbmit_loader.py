"""CHB-MIT Scalp EEG Database loader with seizure annotation support.

This module handles loading and preprocessing of the CHB-MIT dataset:
- Parses patient summary files for seizure annotations
- Loads EDF files with MNE
- Extracts seizure events with precise timestamps
- Preprocesses data to match Siena pipeline for fair comparison
"""

import os
import re
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import mne
from mne.io import read_raw_edf

# Suppress MNE filter warnings from EDF metadata
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", message=".*highpass.*")
warnings.filterwarnings("ignore", message=".*lowpass.*")


class CHBMITLoader:
    """Loader for CHB-MIT Scalp EEG Database.
    
    Dataset structure:
        chb-mit-scalp-eeg-database-1.0.0/
        ├── chb01/
        │   ├── chb01-summary.txt  (seizure annotations)
        │   ├── chb01_01.edf
        │   ├── chb01_02.edf
        │   └── ...
        ├── chb02/
        └── ...
    
    Each summary file contains:
        File Name: chb01_03.edf
        Start Time: 11:42:54
        End Time: 12:42:54
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    """
    
    def __init__(
        self,
        root_dir: str,
        target_sampling_rate: int = 256,
        common_channels: Optional[List[str]] = None,
        preictal_duration_sec: int = 1200,  # 20 minutes
        bandpass_low: float = 0.5,
        bandpass_high: float = 70.0,
        notch_freq: float = 60.0,  # US standard
        verbose: bool = True
    ):
        """Initialize CHB-MIT loader.
        
        Args:
            root_dir: Path to CHB-MIT dataset root
            target_sampling_rate: Target sampling rate (Hz)
            common_channels: List of channel names to use (None = all channels)
            preictal_duration_sec: Duration before seizure to mark as preictal
            bandpass_low: Bandpass filter lower cutoff (Hz)
            bandpass_high: Bandpass filter upper cutoff (Hz)
            notch_freq: Notch filter frequency (Hz)
            verbose: Print progress messages
        """
        self.root_dir = Path(root_dir)
        self.target_sr = target_sampling_rate
        self.common_channels = common_channels
        self.preictal_duration = preictal_duration_sec
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.verbose = verbose
        
        # Validate dataset exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"CHB-MIT dataset not found at {root_dir}")
        
        # Discover all patients
        self.patients = self._discover_patients()
        if self.verbose:
            print(f"Found {len(self.patients)} patients in CHB-MIT dataset")
    
    def _discover_patients(self) -> List[str]:
        """Discover all patient directories.
        
        Handles two possible structures:
        1. Direct patient folders: root_dir/chb01/, root_dir/chb02/
        2. Nested structure: root_dir/chb01/chb01/, root_dir/chb02/chb02/
        """
        patients = []
        
        # First check if root_dir itself is a patient directory
        if (self.root_dir.name.startswith('chb') and 
            any(self.root_dir.glob('*-summary.txt'))):
            # Root dir is a patient collection, look inside
            for item in sorted(self.root_dir.iterdir()):
                if item.is_dir() and item.name.startswith('chb'):
                    # Check for summary file in subdirectory
                    summary_file = item / f"{item.name}-summary.txt"
                    if summary_file.exists():
                        patients.append(item.name)
                        if self.verbose:
                            print(f"  Found patient: {item.name}")
        else:
            # Standard structure - look for chbXX directories
            for item in sorted(self.root_dir.iterdir()):
                if item.is_dir() and item.name.startswith('chb'):
                    summary_file = item / f"{item.name}-summary.txt"
                    if summary_file.exists():
                        patients.append(item.name)
                        if self.verbose:
                            print(f"  Found patient: {item.name}")
        
        return patients
    
    def parse_summary_file(self, patient_id: str) -> Dict[str, List[Dict]]:
        """Parse patient summary file to extract seizure annotations.
        
        Args:
            patient_id: Patient ID (e.g., 'chb01')
        
        Returns:
            Dictionary mapping filename to list of seizure events:
            {
                'chb01_03.edf': [
                    {'start': 2996, 'end': 3036},
                    {'start': 3600, 'end': 3650}
                ],
                ...
            }
        """
        summary_path = self.root_dir / patient_id / f"{patient_id}-summary.txt"
        
        if not summary_path.exists():
            if self.verbose:
                print(f"Warning: Summary file not found for {patient_id}")
                print(f"  Looking for: {summary_path}")
            return {}
        
        if self.verbose:
            print(f"Reading summary file: {summary_path.name}")
        
        annotations = {}
        current_file = None
        
        with open(summary_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect file entry
            if line.startswith('File Name:'):
                current_file = line.split(':', 1)[1].strip()
                annotations[current_file] = []
                i += 1
                continue
            
            # Detect number of seizures
            if line.startswith('Number of Seizures in File:'):
                num_seizures_str = line.split(':', 1)[1].strip()
                num_seizures = int(num_seizures_str)
                
                # Parse each seizure in this file
                for _ in range(num_seizures):
                    i += 1
                    # Seizure Start Time
                    if i < len(lines) and 'Seizure Start Time:' in lines[i]:
                        start_line = lines[i].strip()
                        # Match number before "seconds"
                        match = re.search(r'(\d+)\s*seconds?', start_line)
                        if match:
                            start_sec = int(match.group(1))
                        else:
                            # Try alternative format without "seconds"
                            start_sec = int(start_line.split(':', 1)[1].strip().split()[0])
                        
                        i += 1
                        # Seizure End Time
                        if i < len(lines) and 'Seizure End Time:' in lines[i]:
                            end_line = lines[i].strip()
                            match = re.search(r'(\d+)\s*seconds?', end_line)
                            if match:
                                end_sec = int(match.group(1))
                            else:
                                end_sec = int(end_line.split(':', 1)[1].strip().split()[0])
                            
                            annotations[current_file].append({
                                'start': start_sec,
                                'end': end_sec
                            })
            
            i += 1
        
        if self.verbose and annotations:
            total_seizures = sum(len(v) for v in annotations.values())
            files_with_seizures = sum(1 for v in annotations.values() if len(v) > 0)
            print(f"  Found {files_with_seizures} files with {total_seizures} total seizures")
        
        return annotations
    
    def load_edf_file(
        self,
        patient_id: str,
        filename: str,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, int, List[str]]:
        """Load and preprocess a single EDF file.
        
        Args:
            patient_id: Patient ID (e.g., 'chb01')
            filename: EDF filename (e.g., 'chb01_03.edf')
            preprocess: Apply preprocessing (bandpass, notch, resampling)
        
        Returns:
            Tuple of (data, sampling_rate, channel_names)
            data shape: (n_channels, n_samples)
        """
        edf_path = self.root_dir / patient_id / filename
        
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")
        
        # Load with MNE (suppress warnings from EDF metadata)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Get original sampling rate
        original_sr = int(raw.info['sfreq'])
        
        if preprocess:
            # Apply bandpass filter
            raw.filter(
                l_freq=self.bandpass_low,
                h_freq=self.bandpass_high,
                picks='eeg',
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'},
                verbose=False
            )
            
            # Apply notch filter for power line noise
            raw.notch_filter(
                freqs=self.notch_freq,
                picks='eeg',
                method='iir',
                verbose=False
            )
            
            # Resample if needed
            if original_sr != self.target_sr:
                raw.resample(self.target_sr, npad='auto', verbose=False)
            
            # Common average reference
            raw.set_eeg_reference('average', projection=False, verbose=False)
        
        # Extract data
        data = raw.get_data()  # (n_channels, n_samples)
        channel_names = raw.ch_names
        sampling_rate = int(raw.info['sfreq'])
        
        # Select common channels if specified
        if self.common_channels is not None:
            channel_indices = []
            selected_names = []
            
            for ch in self.common_channels:
                # Try exact match first
                if ch in channel_names:
                    idx = channel_names.index(ch)
                    channel_indices.append(idx)
                    selected_names.append(ch)
                else:
                    # Try case-insensitive match
                    for i, name in enumerate(channel_names):
                        if name.upper() == ch.upper():
                            channel_indices.append(i)
                            selected_names.append(name)
                            break
            
            if len(channel_indices) == 0:
                raise ValueError(
                    f"None of the common channels {self.common_channels} "
                    f"found in {filename}. Available: {channel_names}"
                )
            
            data = data[channel_indices, :]
            channel_names = selected_names
        
        return data, sampling_rate, channel_names
    
    def extract_segments(
        self,
        patient_id: str,
        segment_duration_sec: float = 5.0,
        overlap: float = 0.5,
        balance_classes: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract labeled segments from all files for a patient.
        
        Args:
            patient_id: Patient ID (e.g., 'chb01')
            segment_duration_sec: Duration of each segment
            overlap: Overlap ratio between segments (0-1)
            balance_classes: Balance normal/preictal/ictal classes
        
        Returns:
            Tuple of (segments, labels, patient_ids)
            segments shape: (n_segments, n_channels, n_samples)
            labels shape: (n_segments,) - 0=Normal, 1=Preictal, 2=Ictal
            patient_ids shape: (n_segments,) - patient identifier
        """
        # Parse seizure annotations
        annotations = self.parse_summary_file(patient_id)
        
        all_segments = []
        all_labels = []
        
        # Process each EDF file
        patient_dir = self.root_dir / patient_id
        edf_files = sorted(patient_dir.glob('*.edf'))
        
        for edf_file in edf_files:
            filename = edf_file.name
            
            try:
                # Load data
                data, sr, channels = self.load_edf_file(patient_id, filename)
                
                # Get seizure events for this file
                seizure_events = annotations.get(filename, [])
                
                # Create label mask for entire recording
                duration_sec = data.shape[1] / sr
                label_mask = np.zeros(int(duration_sec * sr))  # All normal initially
                
                # Mark ictal periods
                for event in seizure_events:
                    start_sample = int(event['start'] * sr)
                    end_sample = int(event['end'] * sr)
                    label_mask[start_sample:end_sample] = 2  # Ictal
                    
                    # Mark preictal period (before seizure)
                    preictal_start = max(0, start_sample - int(self.preictal_duration * sr))
                    label_mask[preictal_start:start_sample] = 1  # Preictal
                
                # Create segments
                segment_samples = int(segment_duration_sec * sr)
                step_samples = int(segment_samples * (1 - overlap))
                
                for start_idx in range(0, data.shape[1] - segment_samples, step_samples):
                    end_idx = start_idx + segment_samples
                    
                    # Extract segment
                    segment = data[:, start_idx:end_idx]
                    
                    # Determine label (majority vote in segment)
                    segment_labels = label_mask[start_idx:end_idx]
                    label_counts = np.bincount(segment_labels.astype(int), minlength=3)
                    segment_label = np.argmax(label_counts)
                    
                    all_segments.append(segment)
                    all_labels.append(segment_label)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {filename}: {e}")
                continue
        
        if len(all_segments) == 0:
            raise ValueError(f"No valid segments extracted for patient {patient_id}")
        
        # Stack segments
        segments = np.stack(all_segments, axis=0)
        labels = np.array(all_labels)
        patient_ids = np.full(len(labels), patient_id)
        
        # Balance classes if requested
        if balance_classes:
            segments, labels, patient_ids = self._balance_classes(segments, labels, patient_ids)
        
        if self.verbose:
            print(f"Patient {patient_id}: {len(segments)} segments")
            print(f"  Normal: {np.sum(labels == 0)}")
            print(f"  Preictal: {np.sum(labels == 1)}")
            print(f"  Ictal: {np.sum(labels == 2)}")
        
        return segments, labels, patient_ids
    
    def _balance_classes(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
        patient_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Balance classes by upsampling minority classes.
        
        Args:
            segments: Segment data
            labels: Segment labels
            patient_ids: Patient IDs
        
        Returns:
            Balanced (segments, labels, patient_ids)
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        
        balanced_segments = []
        balanced_labels = []
        balanced_patient_ids = []
        
        for label in unique_labels:
            label_mask = labels == label
            label_segments = segments[label_mask]
            label_patient_ids = patient_ids[label_mask]
            
            # Upsample if needed
            if len(label_segments) < max_count:
                indices = np.random.choice(
                    len(label_segments),
                    size=max_count,
                    replace=True
                )
                label_segments = label_segments[indices]
                label_patient_ids = label_patient_ids[indices]
            
            balanced_segments.append(label_segments)
            balanced_labels.extend([label] * len(label_segments))
            balanced_patient_ids.extend(label_patient_ids)
        
        return (
            np.concatenate(balanced_segments, axis=0),
            np.array(balanced_labels),
            np.array(balanced_patient_ids)
        )
    
    def load_all_patients(
        self,
        segment_duration_sec: float = 5.0,
        overlap: float = 0.5,
        balance_classes: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load data for all patients.
        
        Args:
            segment_duration_sec: Duration of each segment
            overlap: Overlap ratio between segments
            balance_classes: Balance classes within each patient
        
        Returns:
            Dictionary mapping patient_id to (segments, labels):
            {
                'chb01': (segments_array, labels_array),
                'chb02': (segments_array, labels_array),
                ...
            }
        """
        patient_data = {}
        
        for patient_id in self.patients:
            if self.verbose:
                print(f"\nLoading patient {patient_id}...")
            
            try:
                segments, labels, _ = self.extract_segments(
                    patient_id,
                    segment_duration_sec=segment_duration_sec,
                    overlap=overlap,
                    balance_classes=balance_classes
                )
                patient_data[patient_id] = (segments, labels)
            
            except Exception as e:
                if self.verbose:
                    print(f"Error loading patient {patient_id}: {e}")
                continue
        
        return patient_data


def load_chbmit_dataset(
    root_dir: str,
    target_sampling_rate: int = 256,
    common_channels: Optional[List[str]] = None,
    segment_duration_sec: float = 5.0,
    overlap: float = 0.5,
    preictal_duration_sec: int = 1200,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convenience function to load entire CHB-MIT dataset.
    
    Args:
        root_dir: Path to CHB-MIT dataset root
        target_sampling_rate: Target sampling rate (Hz)
        common_channels: List of channel names to use
        segment_duration_sec: Duration of each segment
        overlap: Overlap ratio between segments
        preictal_duration_sec: Duration before seizure to mark as preictal
        verbose: Print progress messages
    
    Returns:
        Dictionary mapping patient_id to (segments, labels)
    """
    loader = CHBMITLoader(
        root_dir=root_dir,
        target_sampling_rate=target_sampling_rate,
        common_channels=common_channels,
        preictal_duration_sec=preictal_duration_sec,
        verbose=verbose
    )
    
    return loader.load_all_patients(
        segment_duration_sec=segment_duration_sec,
        overlap=overlap,
        balance_classes=True
    )
