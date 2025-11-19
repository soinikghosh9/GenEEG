"""Siena Scalp EEG Database loader.

This module handles loading and preprocessing of the Siena dataset:
- Parses patient summary files for seizure annotations
- Loads EDF files with MNE
- Applies preprocessing (filtering, resampling, channel selection)
- Extracts segments with labels (Normal, Preictal, Ictal)
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter
import numpy as np
import mne
from mne.io import read_raw_edf

# Suppress MNE filter warnings from EDF metadata
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", message=".*highpass.*")
warnings.filterwarnings("ignore", message=".*lowpass.*")


def time_to_seconds(time_str: str) -> int:
    """Convert time string to seconds.
    
    Args:
        time_str: Time string in format 'HH:MM:SS' or 'HH.MM.SS'
    
    Returns:
        Time in seconds
    """
    # Handle both ':' and '.' as separators
    time_str = time_str.replace('.', ':')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def normalize_filename(filename: str) -> str:
    """Normalize filename for consistent matching.
    
    Args:
        filename: Original filename
    
    Returns:
        Normalized filename (lowercase, stripped)
    """
    return filename.lower().strip()


class SienaLoader:
    """Loader for Siena Scalp EEG Database.
    
    Dataset structure:
        Siena_dataset/
        ├── PN00/
        │   ├── PN00-1.summary  (seizure annotations)
        │   ├── PN00-1.edf
        │   ├── PN00-2.edf
        │   └── ...
        ├── PN03/
        └── ...
    
    Summary file format:
        File name: PN00-1.edf
        Registration start time: 11:42:54
        Seizure start time: 12:15:30
        Seizure end time: 12:16:15
    """
    
    def __init__(
        self,
        root_dir: str,
        original_sampling_rate: int = 512,
        target_sampling_rate: int = 256,
        num_channels: int = 16,  # Changed from 18 to 16 for consistency
        preictal_duration_sec: int = 1200,  # 20 minutes
        bandpass_low: float = 0.5,
        bandpass_high: float = 70.0,
        notch_freq: float = 50.0,  # European standard
        verbose: bool = True
    ):
        """Initialize Siena loader.
        
        Args:
            root_dir: Path to Siena dataset root
            original_sampling_rate: Original sampling rate (Hz)
            target_sampling_rate: Target sampling rate (Hz)
            num_channels: Number of channels to select
            preictal_duration_sec: Duration before seizure to mark as preictal
            bandpass_low: Bandpass filter lower cutoff (Hz)
            bandpass_high: Bandpass filter upper cutoff (Hz)
            notch_freq: Notch filter frequency (Hz)
            verbose: Print progress messages
        """
        self.root_dir = Path(root_dir)
        self.original_sr = original_sampling_rate
        self.target_sr = target_sampling_rate
        self.num_channels = num_channels
        self.preictal_duration = preictal_duration_sec
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.verbose = verbose
        
        # Preferred channel order (10-20 system)
        self.preferred_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
            'Fz', 'Cz', 'Pz'
        ]
        
        # Class labels
        self.label_map = {
            "normal": 0,
            "preictal": 1,
            "ictal": 2
        }
        
        # Validate dataset exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Siena dataset not found at {root_dir}")
        
        # Discover all patients
        self.patients = self._discover_patients()
        if self.verbose:
            print(f"Found {len(self.patients)} patients in Siena dataset")
    
    def _discover_patients(self) -> List[str]:
        """Discover all patient directories."""
        patients = []
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir() and item.name.upper().startswith('PN'):
                patients.append(item.name)
        return patients
    
    def parse_summary_file(self, patient_id: str) -> Dict[str, Dict]:
        """Parse patient summary file to extract seizure annotations.
        
        Args:
            patient_id: Patient ID (e.g., 'PN00')
        
        Returns:
            Dictionary mapping filename to seizure info:
            {
                'pn00-1.edf': {
                    'seizures': [(start_sec, end_sec), ...]
                },
                ...
            }
        """
        patient_dir = self.root_dir / patient_id
        
        # Find summary file - try multiple patterns
        # Pattern 1: *.summary (old format)
        summary_files = list(patient_dir.glob('*.summary'))
        
        # Pattern 2: Seizures-list-*.txt (Siena v3.0.0 format)
        if not summary_files:
            summary_files = list(patient_dir.glob('Seizures-list-*.txt'))
        
        # Pattern 3: Any .txt file (fallback)
        if not summary_files:
            summary_files = list(patient_dir.glob('*.txt'))
        
        if not summary_files:
            if self.verbose:
                print(f"Warning: No summary file found for {patient_id}")
            return {}
        
        seizure_data = {}
        current_file = None
        registration_start_seconds = None
        seizure_start_rel = None
        
        for summary_file in summary_files:
            if self.verbose:
                print(f"Reading summary file: {summary_file.name}")
            
            with open(summary_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    
                    # Parse file name
                    if line.startswith("File name:"):
                        current_file = normalize_filename(
                            line.split("File name:")[1].strip()
                        )
                        seizure_data.setdefault(current_file, {"seizures": []})
                        registration_start_seconds = None
                        seizure_start_rel = None
                    
                    # Parse registration start time
                    elif line.startswith("Registration start time:") and current_file:
                        time_str = line.split(":", 1)[1].strip()
                        registration_start_seconds = time_to_seconds(time_str)
                    
                    # Parse seizure start time
                    elif line.startswith(("Seizure start time:", "Start time:")) and current_file:
                        if registration_start_seconds is not None:
                            time_str = line.split(":", 1)[1].strip()
                            # Remove extra spaces
                            time_str = time_str.replace(" ", "")
                            seizure_start_abs = time_to_seconds(time_str)
                            seizure_start_rel = seizure_start_abs - registration_start_seconds
                    
                    # Parse seizure end time
                    elif line.startswith(("Seizure end time:", "End time:")) and current_file:
                        if seizure_start_rel is not None and registration_start_seconds is not None:
                            time_str = line.split(":", 1)[1].strip()
                            # Remove extra spaces
                            time_str = time_str.replace(" ", "")
                            seizure_end_abs = time_to_seconds(time_str)
                            seizure_end_rel = seizure_end_abs - registration_start_seconds
                            
                            seizure_data[current_file]["seizures"].append(
                                (seizure_start_rel, seizure_end_rel)
                            )
                            seizure_start_rel = None
        
        if self.verbose and seizure_data:
            total_seizures = sum(len(v["seizures"]) for v in seizure_data.values())
            print(f"  Found {len(seizure_data)} files with {total_seizures} total seizures")
        
        return seizure_data
    
    def select_common_channels(self, raw: mne.io.Raw) -> Optional[mne.io.Raw]:
        """Select common EEG channels from raw data.
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Raw object with selected channels, or None if insufficient channels
        """
        available_eeg_channels = [
            ch for ch in raw.ch_names 
            if 'EEG' in ch.upper() or any(
                p_ch in ch.upper() 
                for p_ch in ['FP', 'F', 'C', 'P', 'O', 'T']
            )
        ]
        
        selected_ch_names = []
        
        # Try to select preferred channels first
        for pch in self.preferred_channels:
            found = next(
                (ach for ach in available_eeg_channels 
                 if pch.upper() == ach.upper().replace("EEG ", "")),
                None
            )
            if found:
                selected_ch_names.append(found)
            if len(selected_ch_names) >= self.num_channels:
                break
        
        # Fill remaining with any available EEG channels
        if len(selected_ch_names) < self.num_channels:
            remaining_eeg = [
                ch for ch in available_eeg_channels 
                if ch not in selected_ch_names
            ]
            selected_ch_names.extend(
                remaining_eeg[:self.num_channels - len(selected_ch_names)]
            )
        
        # Check if we have enough channels
        if len(selected_ch_names) < self.num_channels:
            if self.verbose:
                print(f"Warning: Only {len(selected_ch_names)} channels available, need {self.num_channels}")
            return None
        
        # Pick selected channels
        raw.pick(selected_ch_names[:self.num_channels])
        return raw
    
    def preprocess_eeg(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing to raw EEG data.
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Preprocessed Raw object
        """
        # Bandpass filter
        raw.filter(
            l_freq=self.bandpass_low,
            h_freq=self.bandpass_high,
            method='iir',
            iir_params={'order': 4, 'ftype': 'butter', 'output': 'sos'},
            verbose=False
        )
        
        # Notch filter
        if self.notch_freq:
            raw.notch_filter(
                freqs=self.notch_freq,
                method='fir',
                fir_design='firwin',
                verbose=False
            )
        
        # Resample if needed
        if abs(raw.info['sfreq'] - self.target_sr) > 0.1:
            raw.resample(self.target_sr, npad='auto', verbose=False)
        
        return raw
    
    def determine_segment_label(
        self,
        segment_start_sec: float,
        seizures: List[Tuple[float, float]]
    ) -> int:
        """Determine label for a segment based on seizure annotations.
        
        Args:
            segment_start_sec: Start time of segment in seconds
            seizures: List of (start, end) tuples for seizures
        
        Returns:
            Label: 0=Normal, 1=Preictal, 2=Ictal
        """
        for sz_start, sz_end in seizures:
            # Ictal: during seizure
            if sz_start <= segment_start_sec < sz_end:
                return self.label_map["ictal"]
            
            # Preictal: before seizure
            preictal_start = sz_start - self.preictal_duration
            if preictal_start <= segment_start_sec < sz_start:
                return self.label_map["preictal"]
        
        # Normal: everything else
        return self.label_map["normal"]
    
    def extract_segments(
        self,
        patient_id: str,
        segment_duration_sec: float = 5.0,
        overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract labeled segments from all files for a patient.
        
        Args:
            patient_id: Patient ID (e.g., 'PN00')
            segment_duration_sec: Duration of each segment
            overlap: Overlap ratio between segments (0-1)
        
        Returns:
            Tuple of (segments, labels)
            segments shape: (n_segments, n_channels, n_samples)
            labels shape: (n_segments,) - 0=Normal, 1=Preictal, 2=Ictal
        """
        # Parse seizure annotations
        summary_data = self.parse_summary_file(patient_id)
        
        patient_segments = []
        patient_labels = []
        
        # Process all EDF files
        patient_dir = self.root_dir / patient_id
        edf_files = sorted(patient_dir.glob('*.edf'))
        
        segment_samples = int(segment_duration_sec * self.target_sr)
        step_samples = int(segment_samples * (1 - overlap))
        
        for edf_file in edf_files:
            filename_normalized = normalize_filename(edf_file.name)
            seizure_info = summary_data.get(filename_normalized, {}).get("seizures", [])
            
            try:
                # Load EDF file (suppress filter warnings from EDF metadata)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    raw = read_raw_edf(str(edf_file), preload=True, verbose=False)
                
                # Preprocess
                raw = self.preprocess_eeg(raw)
                
                # Select channels
                raw = self.select_common_channels(raw)
                if raw is None or raw.info['nchan'] != self.num_channels:
                    continue
                
                # Get data
                data = raw.get_data()  # (n_channels, n_samples)
                
                # Extract segments
                for i in range(0, data.shape[1] - segment_samples + 1, step_samples):
                    segment = data[:, i:i + segment_samples]
                    
                    if segment.shape[1] != segment_samples:
                        continue
                    
                    # Determine label
                    segment_start_sec = i / self.target_sr
                    label = self.determine_segment_label(segment_start_sec, seizure_info)
                    
                    patient_segments.append(segment.astype(np.float32))
                    patient_labels.append(label)
            
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {edf_file.name}: {e}")
                continue
        
        if len(patient_segments) == 0:
            raise ValueError(f"No valid segments extracted for patient {patient_id}")
        
        segments = np.stack(patient_segments, axis=0)
        labels = np.array(patient_labels)
        
        if self.verbose:
            print(f"Patient {patient_id}: {len(segments)} segments")
            print(f"  Normal: {(labels == 0).sum()}")
            print(f"  Preictal: {(labels == 1).sum()}")
            print(f"  Ictal: {(labels == 2).sum()}")
        
        return segments, labels
    
    def load_all_patients(
        self,
        segment_duration_sec: float = 5.0,
        overlap: float = 0.5
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load data for all patients.
        
        Args:
            segment_duration_sec: Duration of each segment
            overlap: Overlap ratio between segments
        
        Returns:
            Dictionary mapping patient_id to (segments, labels):
            {
                'PN00': (segments_array, labels_array),
                'PN03': (segments_array, labels_array),
                ...
            }
        """
        patient_data = {}
        
        for patient_id in self.patients:
            if self.verbose:
                print(f"\nLoading patient {patient_id}...")
            
            try:
                segments, labels = self.extract_segments(
                    patient_id,
                    segment_duration_sec=segment_duration_sec,
                    overlap=overlap
                )
                patient_data[patient_id] = (segments, labels)
            
            except Exception as e:
                if self.verbose:
                    print(f"Error loading patient {patient_id}: {e}")
                continue
        
        return patient_data


def load_siena_dataset(
    root_dir: str,
    original_sampling_rate: int = 512,
    target_sampling_rate: int = 256,
    num_channels: int = 18,
    segment_duration_sec: float = 5.0,
    overlap: float = 0.5,
    preictal_duration_sec: int = 1200,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convenience function to load entire Siena dataset.
    
    Args:
        root_dir: Path to Siena dataset root
        original_sampling_rate: Original sampling rate (Hz)
        target_sampling_rate: Target sampling rate (Hz)
        num_channels: Number of channels to select
        segment_duration_sec: Duration of each segment
        overlap: Overlap ratio between segments
        preictal_duration_sec: Duration before seizure to mark as preictal
        verbose: Print progress messages
    
    Returns:
        Dictionary mapping patient_id to (segments, labels)
    """
    loader = SienaLoader(
        root_dir=root_dir,
        original_sampling_rate=original_sampling_rate,
        target_sampling_rate=target_sampling_rate,
        num_channels=num_channels,
        preictal_duration_sec=preictal_duration_sec,
        verbose=verbose
    )
    
    return loader.load_all_patients(
        segment_duration_sec=segment_duration_sec,
        overlap=overlap
    )
