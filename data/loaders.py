"""
EEG Data Loaders for Siena Scalp EEG and CHB-MIT Datasets

Provides functionality to load, preprocess, and segment EEG recordings
from clinical seizure datasets.

This module contains:
- SienaLoader: Loader for Siena Scalp EEG Database (adult patients)
- CHBMITLoader: Loader for CHB-MIT Scalp EEG Database (pediatric patients)
"""

import os
import re
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


class SienaLoader:
    """
    Data loader for Siena Scalp EEG Database (adult patients).
    
    The Siena dataset contains EEG recordings from adult epilepsy patients
    with annotated seizure events.
    
    Args:
        root_dir: Root directory containing Siena dataset
        target_sfreq: Target sampling frequency (default: 256 Hz)
        segment_duration: Duration of each segment in seconds (default: 2.0)
        channels: List of channel names to use (default: None, uses all common channels)
        num_channels: Number of channels to select (default: 16)
    """
    def __init__(
        self,
        root_dir: str,
        target_sfreq: int = 256,
        segment_duration: float = 2.0,
        channels: Optional[List[str]] = None,
        num_channels: int = 16  # Added parameter for pipeline compatibility
    ):
        self.root_dir = Path(root_dir)
        self.target_sfreq = target_sfreq
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * target_sfreq)
        self.num_channels = num_channels
        
        # Common EEG channel names (default 16 channels)
        if channels is None:
            self.channels = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6'
            ][:num_channels]  # Limit to num_channels
        else:
            self.channels = channels[:num_channels] if len(channels) > num_channels else channels
    
    def load_patient_data(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all EEG data for a specific patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'PN00', 'PN01')
        
        Returns:
            segments: Array of EEG segments, shape (N, C, T)
            labels: Array of labels, shape (N,) where 0=interictal, 1=preictal, 2=ictal
        """
        patient_dir = self.root_dir / patient_id
        
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        
        all_segments = []
        all_labels = []
        
        # Look for EDF files
        edf_files = sorted(patient_dir.glob('*.edf'))
        
        for edf_file in edf_files:
            try:
                # Load with MNE
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                
                # Select channels
                available_channels = [ch for ch in self.channels if ch in raw.ch_names]
                if not available_channels:
                    print(f"Warning: No matching channels in {edf_file.name}")
                    continue
                
                raw.pick_channels(available_channels)
                
                # Resample
                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq)
                
                # Get data
                data = raw.get_data()  # Shape: (C, T)
                
                # Determine label from filename or annotations
                label = self._determine_label(edf_file, raw)
                
                # Segment the data
                segments = self._segment_recording(data)
                labels = np.full(len(segments), label, dtype=np.int64)
                
                all_segments.extend(segments)
                all_labels.extend(labels)
                
            except Exception as e:
                print(f"Error loading {edf_file.name}: {e}")
                continue
        
        if not all_segments:
            raise ValueError(f"No valid data loaded for patient {patient_id}")
        
        segments_array = np.array(all_segments, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        return segments_array, labels_array
    
    def _determine_label(self, edf_file: Path, raw: mne.io.Raw) -> int:
        """
        Determine seizure state label from filename or annotations.
        
        Returns:
            0: Interictal (no seizure)
            1: Preictal (pre-seizure)
            2: Ictal (seizure)
        """
        filename = edf_file.name.lower()
        
        # Check filename for keywords
        if 'ictal' in filename or 'seizure' in filename or '_sz' in filename:
            return 2  # Ictal
        elif 'preictal' in filename or 'pre' in filename:
            return 1  # Preictal
        else:
            return 0  # Interictal (default)
    
    def _segment_recording(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Segment continuous recording into fixed-length windows.
        
        Args:
            data: EEG data of shape (C, T)
        
        Returns:
            List of segments, each of shape (C, segment_samples)
        """
        n_channels, n_samples = data.shape
        segments = []
        
        # Slide window across recording
        for start in range(0, n_samples - self.segment_samples + 1, self.segment_samples):
            segment = data[:, start:start + self.segment_samples]
            segments.append(segment)
        
        return segments
    
    def load_all_patients(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data for all patients in the dataset.
        
        Returns:
            Dictionary mapping patient_id to (segments, labels) tuple
        """
        patient_data = {}
        
        # Find all patient directories (e.g., PN00, PN01, ...)
        patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('PN')])
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                segments, labels = self.load_patient_data(patient_id)
                patient_data[patient_id] = (segments, labels)
                print(f"✓ Loaded {patient_id}: {len(segments)} segments")
            except Exception as e:
                print(f"✗ Failed to load {patient_id}: {e}")
        
        return patient_data


class CHBMITLoader:
    """
    Data loader for CHB-MIT Scalp EEG Database (pediatric patients).
    
    The CHB-MIT dataset contains long-term EEG recordings from pediatric
    epilepsy patients at Boston Children's Hospital.
    
    Args:
        root_dir: Root directory containing CHB-MIT dataset
        target_sampling_rate: Target sampling frequency (default: 256 Hz)
        segment_duration: Duration of each segment in seconds (default: 2.0)
        common_channels: List of channel names to use (default: None, uses common channels)
        verbose: Print progress messages (default: True)
    """
    def __init__(
        self,
        root_dir: str,
        target_sampling_rate: int = 256,  # Match pipeline parameter name
        segment_duration: float = 2.0,
        common_channels: Optional[List[str]] = None,  # Match pipeline parameter name
        verbose: bool = True  # Added for pipeline compatibility
    ):
        self.root_dir = Path(root_dir)
        self.target_sfreq = target_sampling_rate  # Store as target_sfreq internally
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * target_sampling_rate)
        self.verbose = verbose
        
        # Common CHB-MIT channels (23 channels in original)
        # We'll standardize to 16 for compatibility
        if common_channels is None:
            self.channels = [
                'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2'
            ]
        else:
            self.channels = common_channels
    
    def load_patient_data(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all EEG data for a specific patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'chb01', 'chb02')
        
        Returns:
            segments: Array of EEG segments, shape (N, C, T)
            labels: Array of labels, shape (N,)
        """
        patient_dir = self.root_dir / patient_id
        
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        
        all_segments = []
        all_labels = []
        
        # Look for EDF files
        edf_files = sorted(patient_dir.glob('*.edf'))
        
        # Load seizure annotations if available
        seizure_info = self._load_seizure_info(patient_dir)
        
        for edf_file in edf_files:
            try:
                # Load with MNE
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                
                # Select available channels
                available_channels = [ch for ch in self.channels if ch in raw.ch_names]
                if not available_channels:
                    # Try to find similar channels
                    available_channels = [ch for ch in raw.ch_names if any(target in ch for target in self.channels[:8])]
                    if not available_channels:
                        print(f"Warning: No matching channels in {edf_file.name}")
                        continue
                
                raw.pick_channels(available_channels[:16])  # Limit to 16 channels
                
                # Resample
                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq)
                
                # Get data
                data = raw.get_data()  # Shape: (C, T)
                
                # Pad if fewer than 16 channels
                if data.shape[0] < 16:
                    padding = np.zeros((16 - data.shape[0], data.shape[1]), dtype=data.dtype)
                    data = np.vstack([data, padding])
                
                # Determine labels (with seizure annotations)
                file_seizures = seizure_info.get(edf_file.name, [])
                segments, labels = self._segment_with_annotations(data, file_seizures, raw.info['sfreq'])
                
                all_segments.extend(segments)
                all_labels.extend(labels)
                
            except Exception as e:
                print(f"Error loading {edf_file.name}: {e}")
                continue
        
        if not all_segments:
            raise ValueError(f"No valid data loaded for patient {patient_id}")
        
        segments_array = np.array(all_segments, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        return segments_array, labels_array
    
    def _load_seizure_info(self, patient_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
        """
        Load seizure timing information from summary file.
        
        Returns:
            Dictionary mapping filename to list of (start_time, end_time) tuples in seconds
        """
        summary_file = patient_dir / f"{patient_dir.name}-summary.txt"
        seizure_info = {}
        
        if not summary_file.exists():
            return seizure_info
        
        try:
            with open(summary_file, 'r') as f:
                current_file = None
                for line in f:
                    line = line.strip()
                    if line.endswith('.edf'):
                        current_file = line
                        seizure_info[current_file] = []
                    elif 'Seizure Start Time' in line:
                        start_time = float(line.split(':')[-1].strip().split()[0])
                    elif 'Seizure End Time' in line:
                        end_time = float(line.split(':')[-1].strip().split()[0])
                        if current_file:
                            seizure_info[current_file].append((start_time, end_time))
        except Exception as e:
            print(f"Warning: Could not parse seizure info: {e}")
        
        return seizure_info
    
    def _segment_with_annotations(
        self,
        data: np.ndarray,
        seizures: List[Tuple[float, float]],
        sfreq: float
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Segment recording and label based on seizure annotations.
        
        Args:
            data: EEG data of shape (C, T)
            seizures: List of (start, end) times in seconds
            sfreq: Sampling frequency
        
        Returns:
            segments: List of segments
            labels: List of labels (0=interictal, 2=ictal)
        """
        n_channels, n_samples = data.shape
        segments = []
        labels = []
        
        for start in range(0, n_samples - self.segment_samples + 1, self.segment_samples):
            segment = data[:, start:start + self.segment_samples]
            
            # Determine if segment overlaps with seizure
            start_time = start / sfreq
            end_time = (start + self.segment_samples) / sfreq
            
            is_seizure = False
            for sz_start, sz_end in seizures:
                if (start_time < sz_end) and (end_time > sz_start):
                    is_seizure = True
                    break
            
            segments.append(segment)
            labels.append(2 if is_seizure else 0)  # 2=ictal, 0=interictal
        
        return segments, labels
    
    def load_all_patients(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data for all patients in the dataset.
        
        Returns:
            Dictionary mapping patient_id to (segments, labels) tuple
        """
        patient_data = {}
        
        # Find all patient directories (e.g., chb01, chb02, ...)
        patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('chb')])
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                segments, labels = self.load_patient_data(patient_id)
                patient_data[patient_id] = (segments, labels)
                print(f"✓ Loaded {patient_id}: {len(segments)} segments")
            except Exception as e:
                print(f"✗ Failed to load {patient_id}: {e}")
        
        return patient_data
