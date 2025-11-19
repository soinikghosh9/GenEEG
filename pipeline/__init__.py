"""
Pipeline Package for GenEEG

This package contains high-level training and validation pipelines that orchestrate
multiple components (VAE, LDM, Classifier, EWC) for end-to-end workflows.

Key Pipelines:
    - cl_lopo_pipeline: Continual Learning Leave-One-Patient-Out validation framework
    - sequential_training: Sequential training on multiple patients with EWC+ER
    
These pipelines integrate all GenEEG components to provide complete, reproducible
workflows for model training and validation.

Author: GenEEG Team
Date: 2025
"""

from .cl_lopo_pipeline import main_cl_lopo_validation, sequential_patient_training

__all__ = [
    'main_cl_lopo_validation',
    'sequential_patient_training',
]
