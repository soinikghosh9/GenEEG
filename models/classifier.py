"""
CNN-BiLSTM Classifier for EEG Seizure Detection

This module implements a hybrid CNN-BiLSTM architecture specifically designed
for EEG classification tasks:
- CNN layers for spatial and temporal feature extraction
- BiLSTM layers for modeling long-range temporal dependencies
- Designed for multi-class seizure state classification

Architecture:
1. Temporal CNN: Extract features across time
2. Spatial CNN: Extract features across EEG channels
3. BiLSTM: Model temporal dependencies
4. FC layers: Final classification
"""

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM hybrid model for EEG classification.
    
    Combines CNN feature extraction with BiLSTM temporal modeling for
    robust seizure state classification.
    
    Args:
        num_classes: Number of output classes (default: 3)
        num_channels: Number of EEG input channels (default: 16)
        seq_length: Length of input sequence in samples (default: 512)
        cnn_filters: List of CNN filter sizes (default: [32, 64, 128])
        lstm_hidden: Hidden size for LSTM (default: 128)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.5)
    """
    def __init__(self, 
                 num_classes=3, 
                 num_channels=16, 
                 seq_length=512, 
                 cnn_filters=[32, 64, 128], 
                 lstm_hidden=128, 
                 lstm_layers=2, 
                 dropout=0.5):
        super(CNNBiLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.seq_length = seq_length
        
        # CNN Feature Extraction Layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # First conv layer: input channels -> first filter size
        in_channels = 1  # We'll reshape input to (batch, 1, channels, samples)
        
        for i, out_channels in enumerate(cnn_filters):
            # Temporal convolution (across time dimension)
            conv_layer = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(1, 7),  # Temporal kernel
                padding=(0, 3),
                bias=False
            )
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            self.pool_layers.append(nn.MaxPool2d((1, 2)))  # Pool only in time dimension
            in_channels = out_channels
        
        # Spatial convolution layer (across channels)
        self.spatial_conv = nn.Conv2d(
            cnn_filters[-1], cnn_filters[-1], 
            kernel_size=(num_channels, 1),
            bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(cnn_filters[-1])
        
        # Calculate the temporal dimension after convolutions
        temp_length = seq_length
        for _ in cnn_filters:
            temp_length = temp_length // 2  # Each pooling layer halves the length
        
        self.cnn_output_size = cnn_filters[-1] * temp_length
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Final classification layers
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)  # *2 for bidirectional
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, x):
        """
        Forward pass for CNN-BiLSTM.
        
        Args:
            x: Input tensor of shape (batch_size, channels, samples)
        
        Returns:
            Raw logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for 2D convolution: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = self.elu(x)
            x = pool(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.elu(x)
        
        # Reshape for LSTM: (batch, time_steps, features)
        # x shape: (batch, filters, 1, time_steps) -> (batch, time_steps, filters)
        x = x.squeeze(2).transpose(1, 2)  # (batch, time_steps, filters)
        
        # BiLSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output from both directions
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        # Take the last time step
        x = lstm_out[:, -1, :]  # (batch, hidden_size * 2)
        
        # Classification head
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x  # Raw logits
    
    def extract_features(self, x):
        """
        Extract features before final classification layer.
        
        Useful for continual learning, visualization, and analysis.
        
        Args:
            x: Input tensor of shape (batch_size, channels, samples)
        
        Returns:
            Features of shape (batch_size, lstm_hidden)
        """
        batch_size = x.size(0)
        
        # Reshape for 2D convolution: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = self.elu(x)
            x = pool(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.elu(x)
        
        # Reshape for LSTM
        x = x.squeeze(2).transpose(1, 2)
        
        # BiLSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        # Feature extraction (before classification)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        return x  # Features before final classification
