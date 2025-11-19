"""
EEGNet Model Implementation

Compact CNN specifically designed for EEG signals.
Used as a baseline classifier in the GenEEG project.
"""

import torch
import torch.nn as nn
from configs.dataset_config import DatasetConfig


class EEGNet(nn.Module):
    """
    EEGNet: A compact convolutional neural network for EEG-based BCIs.
    
    Reference:
    Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based 
    brain–computer interfaces." Journal of Neural Engineering, 2018.
    
    Args:
        num_classes: Number of output classes
        Chans: Number of EEG channels (default from DatasetConfig)
        Samples: Number of time samples per segment (default from DatasetConfig)
        F1: Number of temporal filters (default: 8)
        D: Depth multiplier for depthwise convolution (default: 2)
        F2: Number of pointwise filters (default: 16)
        kernel_length: Length of temporal convolution kernel (default: 64)
        dropoutRate: Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes, Chans=None, Samples=None, 
                 F1=8, D=2, F2=16, kernel_length=64, dropoutRate=0.5):
        super(EEGNet, self).__init__()
        
        # Use dataset config defaults if not specified
        if Chans is None:
            Chans = DatasetConfig.COMMON_EEG_CHANNELS
        if Samples is None:
            Samples = DatasetConfig.SEGMENT_SAMPLES
        
        # Ensure kernel_length is reasonable for Samples
        if kernel_length > Samples // 2:  # Ad-hoc adjustment if kernel is too large
            kernel_length = Samples // 4 
            print(f"[EEGNet Warning] kernel_length was too large for Samples, adjusted to {kernel_length}")

        self.Chans = Chans
        self.Samples = Samples
        self.num_classes = num_classes

        # Block 1 (Temporal Convolution)
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise Convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu2 = nn.ELU()
        
        # Adjusted pooling based on typical EEGNet structures, ensure output size is positive
        pool2_kernel_size = 4 
        if self.Samples // pool2_kernel_size < 1:
            pool2_kernel_size = 2 
        if self.Samples // pool2_kernel_size < 1:
            pool2_kernel_size = 1
        self.pool2 = nn.AvgPool2d((1, pool2_kernel_size))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Separable Convolution
        # Adjusted kernel based on typical EEGNet structures
        sep_kernel_length = 16 
        current_samples_after_pool2 = self.Samples // pool2_kernel_size
        if sep_kernel_length > current_samples_after_pool2 // 2:
            sep_kernel_length = current_samples_after_pool2 // 4
        
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, sep_kernel_length), 
                              padding=(0, sep_kernel_length // 2), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu3 = nn.ELU()
        
        pool3_kernel_size = 8
        current_samples_after_sepconv = current_samples_after_pool2
        if current_samples_after_sepconv // pool3_kernel_size < 1:
            pool3_kernel_size = 4
        if current_samples_after_sepconv // pool3_kernel_size < 1:
            pool3_kernel_size = 2
        if current_samples_after_sepconv // pool3_kernel_size < 1:
            pool3_kernel_size = 1
        self.pool3 = nn.AvgPool2d((1, pool3_kernel_size))
        self.dropout3 = nn.Dropout(dropoutRate)
        
        # Classifier: Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)  # (B, InChannels, H, W) for Conv2D
            x_calc = self.bn1(self.conv1(dummy_input))
            x_calc = self.dropout2(self.pool2(self.elu2(self.bn2(self.conv2(x_calc)))))
            x_calc = self.dropout3(self.pool3(self.elu3(self.bn3(self.conv3(x_calc)))))
            self.flattened_size = x_calc.shape[1] * x_calc.shape[2] * x_calc.shape[3]
            if self.flattened_size == 0:
                raise ValueError(
                    f"EEGNet flattened size is 0. Check pooling/conv parameters. "
                    f"Chans={Chans}, Samples={Samples}, F1={F1}, D={D}, F2={F2}, "
                    f"kernel_length={kernel_length}, sep_kernel_length={sep_kernel_length}, "
                    f"pool2_size={pool2_kernel_size}, pool3_size={pool3_kernel_size}"
                )

        self.fc1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
            
        Returns:
            Raw logits of shape (batch, num_classes)
        """
        # Input x: (batch, chans, samples) -> expected by EEGNet
        x = x.unsqueeze(1)  # Reshape for Conv2D: (batch, 1, chans, samples)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x  # Raw logits
