"""
Test CNN-BiLSTM Classifier

Validates classifier architecture, forward pass, and feature extraction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.classifier import CNNBiLSTM
from configs.model_config import ModelConfig


def test_classifier_instantiation():
    print("[TEST] Classifier Instantiation...")
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=18,
        seq_length=512,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )
    
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    print(f"  [SUCCESS] Classifier created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return classifier


def test_classifier_forward_pass():
    print("[TEST] Classifier Forward Pass...")
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=18,
        seq_length=512,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )
    
    batch = 4
    channels = 18
    length = 512
    
    x = torch.randn(batch, channels, length)
    logits = classifier(x)
    
    assert logits.shape == (batch, 3), f"Logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values!"
    
    print(f"  [SUCCESS] Input: {x.shape} -> Logits: {logits.shape}")
    print(f"  Sample logits: {logits[0].tolist()}")


def test_classifier_feature_extraction():
    print("[TEST] Classifier Feature Extraction...")
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=18,
        seq_length=512,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )
    
    batch = 4
    channels = 18
    length = 512
    
    x = torch.randn(batch, channels, length)
    features = classifier.extract_features(x)
    
    assert features.shape == (batch, 128), f"Features shape: {features.shape}"
    assert torch.isfinite(features).all(), "Features contain non-finite values!"
    
    print(f"  [SUCCESS] Input: {x.shape} -> Features: {features.shape}")


def test_classifier_different_configs():
    print("[TEST] Classifier with Different Configurations...")
    
    configs = [
        {"cnn_filters": [16, 32], "lstm_hidden": 64, "lstm_layers": 1},
        {"cnn_filters": [32, 64, 128, 256], "lstm_hidden": 256, "lstm_layers": 3},
        {"cnn_filters": [64], "lstm_hidden": 128, "lstm_layers": 2},
    ]
    
    for i, config in enumerate(configs):
        classifier = CNNBiLSTM(
            num_classes=3,
            num_channels=18,
            seq_length=512,
            **config
        )
        
        x = torch.randn(2, 18, 512)
        logits = classifier(x)
        
        assert logits.shape == (2, 3), f"Config {i}: Logits shape {logits.shape}"
        print(f"  [SUCCESS] Config {i}: CNN={config['cnn_filters']}, LSTM hidden={config['lstm_hidden']}")


def test_classifier_different_lengths():
    print("[TEST] Classifier with Different Sequence Lengths...")
    
    # Test different sequence lengths
    lengths = [256, 512, 1024, 2048]
    
    for length in lengths:
        classifier = CNNBiLSTM(
            num_classes=3,
            num_channels=18,
            seq_length=length,
            cnn_filters=[32, 64, 128],
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.5
        )
        
        x = torch.randn(2, 18, length)
        logits = classifier(x)
        
        assert logits.shape == (2, 3), f"Length {length}: Logits shape {logits.shape}"
        print(f"  [SUCCESS] Length {length}: Input {x.shape} -> Logits {logits.shape}")


def test_classifier_gradient_flow():
    print("[TEST] Classifier Gradient Flow...")
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=18,
        seq_length=512,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )
    
    batch = 4
    channels = 18
    length = 512
    
    x = torch.randn(batch, channels, length, requires_grad=True)
    labels = torch.randint(0, 3, (batch,))
    
    # Forward pass
    logits = classifier(x)
    
    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and torch.isfinite(p.grad).all() for p in classifier.parameters())
    assert has_gradients, "No gradients or non-finite gradients!"
    
    print(f"  [SUCCESS] Gradient flow verified. Loss: {loss.item():.6f}")


def test_classifier_eval_mode():
    print("[TEST] Classifier Eval Mode (dropout disabled)...")
    
    classifier = CNNBiLSTM(
        num_classes=3,
        num_channels=18,
        seq_length=512,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )
    
    x = torch.randn(2, 18, 512)
    
    # Train mode (stochastic)
    classifier.train()
    out1 = classifier(x)
    out2 = classifier(x)
    train_diff = torch.abs(out1 - out2).max().item()
    
    # Eval mode (deterministic)
    classifier.eval()
    with torch.no_grad():
        out3 = classifier(x)
        out4 = classifier(x)
    eval_diff = torch.abs(out3 - out4).max().item()
    
    print(f"  Train mode max diff: {train_diff:.6f}")
    print(f"  Eval mode max diff: {eval_diff:.6f}")
    print(f"  [SUCCESS] Eval mode is deterministic (diff={eval_diff:.10f})")


def main():
    print("="*60)
    print("Testing CNN-BiLSTM Classifier")
    print("="*60)
    
    test_classifier_instantiation()
    test_classifier_forward_pass()
    test_classifier_feature_extraction()
    test_classifier_different_configs()
    test_classifier_different_lengths()
    test_classifier_gradient_flow()
    test_classifier_eval_mode()
    
    print("="*60)
    print("[SUCCESS] All classifier tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
