"""
Classifier Training Module for GenEEG

This module provides training and evaluation functions for the CNN-BiLSTM classifier
used in seizure detection. Includes weighted loss for class imbalance, validation
split, early stopping, and comprehensive evaluation metrics.

Key Components:
    - train_pytorch_classifier: Training loop with validation and early stopping
    - evaluate_pytorch_classifier: Comprehensive evaluation on test set
    - get_class_weights: Compute inverse frequency weights for class balancing

The classifier training is designed for multi-class seizure detection (Ictal/Interictal/
Preictal) with proper handling of class imbalance and fold-specific normalization.

Author: GenEEG Team
Date: 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import GenericDataset, create_optimized_dataloader
from configs.training_config import SEED_VALUE


def get_class_weights(y_labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse frequency weights for class balancing.
    
    Args:
        y_labels: Array of integer class labels
        device: Device to place weights tensor on
    
    Returns:
        Tensor of class weights (num_classes,) on specified device
    
    Example:
        >>> y = np.array([0, 0, 0, 1, 1, 2])  # 3 samples class 0, 2 samples class 1, 1 sample class 2
        >>> weights = get_class_weights(y, device)
        >>> # weights ≈ [1.0, 1.5, 3.0] (inversely proportional to frequency)
    """
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    num_classes = len(unique_classes)
    total_samples = len(y_labels)
    
    # Inverse frequency weighting: weight_i = total / (num_classes * count_i)
    weights = total_samples / (num_classes * class_counts)
    
    # Ensure weights align with class indices (handle missing classes)
    weight_tensor = torch.ones(num_classes, dtype=torch.float32, device=device)
    for i, cls in enumerate(unique_classes):
        weight_tensor[cls] = weights[i]
    
    return weight_tensor


def train_pytorch_classifier(
    model: nn.Module,
    X_data: np.ndarray,
    y_data: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name_suffix: str,
    scenario_name: str,
    output_dir: str,
    class_names_list_global: list[str],
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    validation_split: float = 0.2,
) -> tuple[nn.Module, dict]:
    """
    Generic training loop for PyTorch classifiers with validation and early stopping.
    
    This function properly handles class-weighted loss to address class imbalance,
    implements early stopping based on validation loss, and tracks comprehensive
    training metrics including accuracy, loss, and AUC.
    
    Args:
        model: PyTorch classifier model (e.g., CNNBiLSTM)
        X_data: Input features (N_samples, ...) as numpy array
        y_data: Target labels (N_samples,) as numpy array
        epochs: Maximum number of training epochs
        lr: Learning rate for AdamW optimizer
        device: Device to train on (cuda/cpu)
        model_name_suffix: Model identifier (e.g., "CNNBiLSTM")
        scenario_name: Scenario description (e.g., "Real_Data", "Fold_01")
        output_dir: Directory to save checkpoints and plots
        class_names_list_global: List of class names for plotting
        batch_size: Training batch size (default: 32)
        early_stopping_patience: Epochs to wait before early stopping (default: 10)
        validation_split: Fraction of data for validation (default: 0.2)
                          Set to None or 0 to disable validation
    
    Returns:
        Tuple of (trained_model, history_dict)
        
        history_dict contains:
            - train_loss: List of training losses per epoch
            - train_acc: List of training accuracies per epoch
            - val_loss: List of validation losses per epoch (if validation enabled)
            - val_acc: List of validation accuracies per epoch (if validation enabled)
            - val_auc_weighted_epoch: List of validation weighted AUC per epoch
            - final_metrics_train: Final metrics on training set using best model
            - final_metrics_val: Final metrics on validation set using best model
    
    Example:
        >>> from models import CNNBiLSTM
        >>> model = CNNBiLSTM(input_channels=22, num_classes=3)
        >>> model, history = train_pytorch_classifier(
        ...     model, X_train, y_train, epochs=100, lr=1e-3, device=device,
        ...     model_name_suffix="CNNBiLSTM", scenario_name="Fold_01",
        ...     output_dir="./checkpoints", class_names_list_global=["Ictal", "Interictal", "Preictal"]
        ... )
    
    Note:
        - Class weights are computed from TRAINING data only (no data leakage)
        - Best model is saved based on validation loss (if validation enabled)
        - Last epoch model is always saved
        - Gradient clipping (max_norm=1.0) applied for stability
    """
    full_model_name = f"{model_name_suffix}_{scenario_name.replace(' ', '_')}"
    print(f"\n--- Training PyTorch Classifier: {full_model_name} ---")
    num_classes_actual = len(class_names_list_global)
    model.to(device)
    
    perform_validation = validation_split is not None and 0 < validation_split < 1
    
    # Split data for training/validation
    if perform_validation:
        # Check if stratification is possible (requires >1 sample per class)
        unique_labels, counts = np.unique(y_data, return_counts=True)
        can_stratify = len(unique_labels) > 1 and np.min(counts) > 1
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data,
            test_size=validation_split,
            random_state=SEED_VALUE,
            stratify=y_data if can_stratify else None
        )
        
        # CRITICAL: Compute class weights using only training data (no leakage)
        class_weights = get_class_weights(y_train, device=device)
        print(f"  Data split into {len(y_train)} train and {len(y_val)} validation samples.")
    else:
        X_train, y_train = X_data, y_data
        X_val, y_val = None, None
        class_weights = get_class_weights(y_data, device=device)
        print(f"  No validation split. Training on all {len(y_train)} provided samples.")
    
    # Create optimizer and weighted criterion
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create DataLoaders
    train_dataset = GenericDataset(X_train, y_train)
    train_loader = create_optimized_dataloader(train_dataset, batch_size, shuffle=True)
    
    val_loader = None
    if perform_validation and X_val is not None:
        val_dataset = GenericDataset(X_val, y_val)
        val_loader = create_optimized_dataloader(val_dataset, batch_size, shuffle=False)
        print(f"  Training with {len(train_loader)} train batches and {len(val_loader)} validation batches.")
    else:
        print(f"  Training with {len(train_loader)} train batches (no validation).")
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc_weighted_epoch': [],
        'final_metrics_train': {},
        'final_metrics_val': {}
    }
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, f"{full_model_name}_best_val_loss.pt")
    last_model_path = os.path.join(output_dir, f"{full_model_name}_last_epoch.pt")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss, epoch_train_correct, epoch_train_total = 0, 0, 0
        
        pbar_train = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} Train {full_model_name}",
            leave=False
        )
        
        for batch_data, batch_labels in pbar_train:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_total += batch_labels.size(0)
            epoch_train_correct += (predicted == batch_labels).sum().item()
            
            pbar_train.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_train_loss / epoch_train_total if epoch_train_total > 0 else 0
        avg_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # Validation
        avg_val_loss_epoch = float('nan')
        avg_val_acc_epoch = float('nan')
        current_val_auc_weighted_epoch = float('nan')
        
        if perform_validation and val_loader:
            model.eval()
            epoch_val_loss, epoch_val_correct, epoch_val_total = 0, 0, 0
            all_val_probas_epoch_list = []
            all_val_labels_epoch_list = []
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    outputs = model(batch_data)
                    val_loss_batch = criterion(outputs, batch_labels)
                    epoch_val_loss += val_loss_batch.item() * batch_data.size(0)
                    
                    probas_batch = torch.softmax(outputs, dim=1)
                    _, predicted_val = torch.max(probas_batch.data, 1)
                    epoch_val_total += batch_labels.size(0)
                    epoch_val_correct += (predicted_val == batch_labels).sum().item()
                    
                    all_val_probas_epoch_list.append(probas_batch.cpu().numpy())
                    all_val_labels_epoch_list.extend(batch_labels.cpu().numpy())
            
            avg_val_loss_epoch = epoch_val_loss / epoch_val_total if epoch_val_total > 0 else float('inf')
            avg_val_acc_epoch = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0.0
            
            # Compute validation AUC
            if all_val_probas_epoch_list and epoch_val_total > 0:
                all_val_probas_epoch_np = np.concatenate(all_val_probas_epoch_list, axis=0)
                all_val_labels_epoch_np = np.array(all_val_labels_epoch_list)
                try:
                    if len(np.unique(all_val_labels_epoch_np)) > 1:
                        current_val_auc_weighted_epoch = roc_auc_score(
                            all_val_labels_epoch_np, all_val_probas_epoch_np,
                            multi_class='ovr', average='weighted'
                        )
                except ValueError:
                    current_val_auc_weighted_epoch = np.nan
        
        history['val_loss'].append(avg_val_loss_epoch)
        history['val_acc'].append(avg_val_acc_epoch)
        history['val_auc_weighted_epoch'].append(current_val_auc_weighted_epoch)
        
        # Print epoch summary
        print_msg = (
            f"Epoch {epoch}/{epochs} - {full_model_name}: "
            f"Tr L: {avg_train_loss:.4f}, Tr Acc: {avg_train_acc:.4f}"
        )
        if perform_validation:
            print_msg += (
                f" | Val L: {avg_val_loss_epoch:.4f}, "
                f"Val Acc: {avg_val_acc_epoch:.4f}, "
                f"Val AUC_w: {current_val_auc_weighted_epoch:.4f}"
            )
        print(print_msg)
        
        # Early stopping logic
        if perform_validation:
            if avg_val_loss_epoch < best_val_loss:
                best_val_loss = avg_val_loss_epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best {full_model_name} model saved (Epoch {epoch}) with Val Loss: {best_val_loss:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"  Early stopping for {full_model_name} at Epoch {epoch}.")
                    break
    
    # Save last epoch model
    torch.save(model.state_dict(), last_model_path)
    print(f"  {full_model_name} model from last epoch saved to: {last_model_path}")
    
    # Load best model for final evaluation
    final_model_state_path_to_load = (
        best_model_path if (perform_validation and os.path.exists(best_model_path))
        else last_model_path
    )
    
    if os.path.exists(final_model_state_path_to_load):
        print(f"  {full_model_name}: Loading model from {final_model_state_path_to_load} for final train/val metrics.")
        model.load_state_dict(torch.load(final_model_state_path_to_load))
    else:
        print(f"  [WARN] {full_model_name}: No model checkpoint at {final_model_state_path_to_load}. Using current model state.")
    model.eval()
    
    # Helper function to get predictions and probabilities
    def get_predictions_and_probas(loader, desc_prefix):
        all_preds_list, all_labels_list, all_probas_list = [], [], []
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(loader, desc=f"{desc_prefix} {full_model_name}", leave=False):
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                probas_batch = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probas_batch.data, 1)
                all_preds_list.extend(predicted.cpu().numpy())
                all_labels_list.extend(batch_labels.cpu().numpy())
                all_probas_list.append(probas_batch.cpu().numpy())
        if not all_probas_list:
            return np.array([]), np.array([]), np.array([])
        return (
            np.array(all_preds_list),
            np.array(all_labels_list),
            np.concatenate(all_probas_list, axis=0)
        )
    
    # Import plot functions (avoid circular imports by importing here)
    from evaluation.metrics import plot_confusion_matrix, plot_roc_auc_curves
    
    # Final metrics for training set
    print(f"  Calculating final metrics for TRAINING set using model: {os.path.basename(final_model_state_path_to_load)}")
    train_preds_final, train_labels_final, train_probas_final_np = get_predictions_and_probas(
        train_loader, "Final Metrics Train"
    )
    
    if len(train_labels_final) > 0:
        cm_train_final = confusion_matrix(
            train_labels_final, train_preds_final,
            labels=np.arange(num_classes_actual)
        )
        plot_confusion_matrix(
            cm_train_final, class_names_list_global,
            title=f'CM TRAIN (Final Model): {full_model_name}',
            save_path=os.path.join(output_dir, f"CM_TRAIN_Final_{full_model_name}.png")
        )
        train_auc_scores_final = plot_roc_auc_curves(
            train_labels_final, train_probas_final_np, num_classes_actual,
            class_names_list_global,
            title_prefix=f"TRAIN ROC (Final Model): {full_model_name}",
            save_path=os.path.join(output_dir, f"ROC_TRAIN_Final_{full_model_name}.png")
        )
        history['final_metrics_train']['auc_scores'] = train_auc_scores_final
        history['final_metrics_train']['confusion_matrix'] = cm_train_final.tolist()
        history['final_metrics_train']['accuracy'] = accuracy_score(train_labels_final, train_preds_final)
    
    # Final metrics for validation set
    if perform_validation and val_loader:
        print(f"  Calculating final metrics for VALIDATION set using model: {os.path.basename(final_model_state_path_to_load)}")
        val_preds_final, val_labels_final, val_probas_final_np = get_predictions_and_probas(
            val_loader, "Final Metrics Val"
        )
        
        if len(val_labels_final) > 0:
            cm_val_final = confusion_matrix(
                val_labels_final, val_preds_final,
                labels=np.arange(num_classes_actual)
            )
            plot_confusion_matrix(
                cm_val_final, class_names_list_global,
                title=f'CM VAL (Best Model): {full_model_name}',
                save_path=os.path.join(output_dir, f"CM_VAL_Best_{full_model_name}.png")
            )
            val_auc_scores_final = plot_roc_auc_curves(
                val_labels_final, val_probas_final_np, num_classes_actual,
                class_names_list_global,
                title_prefix=f"VAL ROC (Best Model): {full_model_name}",
                save_path=os.path.join(output_dir, f"ROC_VAL_Best_{full_model_name}.png")
            )
            history['final_metrics_val']['auc_scores'] = val_auc_scores_final
            history['final_metrics_val']['confusion_matrix'] = cm_val_final.tolist()
            history['final_metrics_val']['accuracy'] = accuracy_score(val_labels_final, val_preds_final)
    
    # Plot training curves
    _plot_training_curves(history, full_model_name, output_dir, perform_validation)
    
    return model, history


def evaluate_pytorch_classifier(
    model: nn.Module,
    X_test_data: np.ndarray,
    y_test_data: np.ndarray,
    device: torch.device,
    model_name_suffix: str,
    scenario_name: str,
    output_dir: str,
    class_names_list_global: list[str],
    batch_size: int = 32,
) -> dict:
    """
    Evaluates a trained PyTorch classifier on the test set.
    
    Args:
        model: Trained PyTorch classifier
        X_test_data: Test features (N_samples, ...)
        y_test_data: Test labels (N_samples,)
        device: Device to run evaluation on
        model_name_suffix: Model identifier
        scenario_name: Scenario description
        output_dir: Directory to save plots
        class_names_list_global: List of class names
        batch_size: Evaluation batch size (default: 32)
    
    Returns:
        Dictionary containing:
            - test_accuracy: Overall accuracy
            - test_f1_weighted: Weighted F1 score
            - test_f1_micro: Micro-averaged F1 score
            - test_f1_macro: Macro-averaged F1 score
            - report_dict_test: Classification report as dictionary
            - report_str_test: Classification report as string
            - confusion_matrix_test: Confusion matrix
            - auc_scores_test: AUC scores (if probabilities available)
    
    Example:
        >>> results = evaluate_pytorch_classifier(
        ...     model, X_test, y_test, device,
        ...     "CNNBiLSTM", "Fold_01_Test", "./results",
        ...     ["Ictal", "Interictal", "Preictal"]
        ... )
        >>> print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    """
    full_model_name = f"{model_name_suffix}_{scenario_name.replace(' ', '_')}"
    print(f"--- Evaluating PyTorch Classifier on Test Set: {full_model_name} ---")
    model.to(device)
    model.eval()
    
    all_preds_test, all_labels_test, all_probas_test_list = [], [], []
    num_classes_actual = len(class_names_list_global)
    
    test_dataset = GenericDataset(X_test_data, y_test_data)
    test_loader = create_optimized_dataloader(test_dataset, batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc=f"Test {full_model_name}", leave=False):
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            probas_batch = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probas_batch.data, 1)
            
            all_preds_test.extend(predicted.cpu().numpy())
            all_labels_test.extend(batch_labels.numpy())
            all_probas_test_list.append(probas_batch.cpu().numpy())
    
    # Handle empty test set
    if not all_labels_test:
        print(f"  [WARN] Test set for {full_model_name} was empty. No evaluation metrics to report.")
        return {
            "error": "Empty test set provided.",
            "test_accuracy": 0, "test_f1_weighted": 0,
            "test_f1_micro": 0, "test_f1_macro": 0,
        }
    
    # Compute metrics
    acc = accuracy_score(all_labels_test, all_preds_test)
    f1_w = f1_score(all_labels_test, all_preds_test, average='weighted', zero_division=0)
    f1_micro = f1_score(all_labels_test, all_preds_test, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
    
    report_str_test = classification_report(
        all_labels_test, all_preds_test,
        target_names=class_names_list_global, zero_division=0
    )
    report_dict_test = classification_report(
        all_labels_test, all_preds_test,
        target_names=class_names_list_global, output_dict=True, zero_division=0
    )
    
    # Import plot functions
    from evaluation.metrics import plot_confusion_matrix, plot_roc_auc_curves
    
    # Plot confusion matrix
    cm_test = confusion_matrix(all_labels_test, all_preds_test, labels=np.arange(num_classes_actual))
    plot_confusion_matrix(
        cm_test, class_names_list_global,
        title=f'CM TEST: {full_model_name}',
        save_path=os.path.join(output_dir, f"CM_TEST_{full_model_name}.png")
    )
    
    # Plot ROC curves
    test_auc_scores = None
    if all_probas_test_list:
        all_probas_test_np = np.concatenate(all_probas_test_list, axis=0)
        test_auc_scores = plot_roc_auc_curves(
            np.array(all_labels_test), all_probas_test_np, num_classes_actual,
            class_names_list_global,
            title_prefix=f"TEST ROC: {full_model_name}",
            save_path=os.path.join(output_dir, f"ROC_TEST_{full_model_name}.png")
        )
    
    # Print summary
    print(f"  {full_model_name} - Test Accuracy: {acc:.4f}, F1 (Weighted): {f1_w:.4f}, F1 (Macro): {f1_macro:.4f}")
    if test_auc_scores:
        print(f"  Test Weighted AUC (OvR): {test_auc_scores.get('auc_weighted_avg_ovr', np.nan):.4f}")
    print(f"  Classification Report for {full_model_name} (Test Set):\n{report_str_test}")
    
    # Prepare results dictionary
    results_dict = {
        "test_accuracy": acc,
        "test_f1_weighted": f1_w,
        "test_f1_micro": f1_micro,
        "test_f1_macro": f1_macro,
        "report_dict_test": report_dict_test,
        "report_str_test": report_str_test,
        "confusion_matrix_test": cm_test.tolist()
    }
    if test_auc_scores:
        results_dict["auc_scores_test"] = test_auc_scores
    
    return results_dict


def _plot_training_curves(
    history: dict,
    full_model_name: str,
    output_dir: str,
    perform_validation: bool
) -> None:
    """
    Internal helper to plot training and validation curves with publication quality.
    
    Args:
        history: Training history dictionary
        full_model_name: Full model name for title
        output_dir: Directory to save plot
        perform_validation: Whether validation was performed
    """
    # Set aesthetic styling with soft colors
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'grid.alpha': 0.3,
    })
    
    fig_curves, axes_curves = plt.subplots(1, 3, figsize=(18, 6))
    fig_curves.suptitle(f'{full_model_name} Training & Validation Progress', 
                       fontsize=17, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Loss curve with soft colors
    axes_curves[0].plot(history['train_loss'], label='Training', 
                       color='#5DADE2', linewidth=2.5, alpha=0.85)
    if perform_validation and any(not np.isnan(vl) for vl in history['val_loss']):
        axes_curves[0].plot(history['val_loss'], label='Validation', 
                           color='#E85D75', linewidth=2.5, alpha=0.85, linestyle='--')
    axes_curves[0].set_title('Loss Curve', fontsize=15, fontweight='bold', color='#2C3E50')
    axes_curves[0].set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
    axes_curves[0].set_ylabel('Loss Value', fontsize=14, fontweight='600', color='#2C3E50')
    axes_curves[0].legend(loc='best', frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
    axes_curves[0].grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    axes_curves[0].set_xlim(0, len(history['train_loss']) - 1)
    for spine in axes_curves[0].spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Accuracy curve with soft colors
    axes_curves[1].plot(history['train_acc'], label='Training', 
                       color='#48C9B0', linewidth=2.5, alpha=0.85)
    if perform_validation and any(not np.isnan(va) for va in history['val_acc']):
        axes_curves[1].plot(history['val_acc'], label='Validation', 
                           color='#F39C6B', linewidth=2.5, alpha=0.85, linestyle='--')
    axes_curves[1].set_title('Accuracy Curve', fontsize=15, fontweight='bold', color='#2C3E50')
    axes_curves[1].set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
    axes_curves[1].set_ylabel('Accuracy', fontsize=14, fontweight='600', color='#2C3E50')
    axes_curves[1].legend(loc='best', frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
    axes_curves[1].grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
    axes_curves[1].set_xlim(0, len(history['train_acc']) - 1)
    axes_curves[1].set_ylim(0, 1.05)
    for spine in axes_curves[1].spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.2)
    
    # Validation AUC curve (or F1 Score) with soft colors
    if perform_validation and 'val_auc_weighted_epoch' in history:
        valid_auc = [v for v in history['val_auc_weighted_epoch'] if not np.isnan(v)]
        if valid_auc:
            axes_curves[2].plot(history['val_auc_weighted_epoch'], label='Weighted AUC (OvR)', 
                               color='#BB8FCE', linewidth=2.5, alpha=0.85)
            axes_curves[2].set_title('Validation AUC', fontsize=15, fontweight='bold', color='#2C3E50')
            axes_curves[2].set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
            axes_curves[2].set_ylabel('AUC Score', fontsize=14, fontweight='600', color='#2C3E50')
            axes_curves[2].legend(loc='best', frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
            axes_curves[2].grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
            axes_curves[2].set_ylim(0.45, 1.05)
            axes_curves[2].set_xlim(0, len(history['val_auc_weighted_epoch']) - 1)
            for spine in axes_curves[2].spines.values():
                spine.set_edgecolor('#BDBDBD')
                spine.set_linewidth(1.2)
    else:
        # If no AUC, plot F1 score if available
        if 'train_f1' in history:
            axes_curves[2].plot(history['train_f1'], label='Training F1', 
                               color='#85C1E9', linewidth=2.5, alpha=0.85)
            if perform_validation and 'val_f1' in history:
                axes_curves[2].plot(history['val_f1'], label='Validation F1', 
                                   color='#F8B88B', linewidth=2.5, alpha=0.85, linestyle='--')
            axes_curves[2].set_title('F1 Score', fontsize=15, fontweight='bold', color='#2C3E50')
            axes_curves[2].set_xlabel('Epoch', fontsize=14, fontweight='600', color='#2C3E50')
            axes_curves[2].set_ylabel('F1 Score', fontsize=14, fontweight='600', color='#2C3E50')
            axes_curves[2].legend(loc='best', frameon=True, shadow=False, framealpha=0.95, edgecolor='#BDBDBD')
            axes_curves[2].grid(True, linestyle='--', alpha=0.3, linewidth=0.6, color='#BDBDBD')
            axes_curves[2].set_ylim(0, 1.05)
            axes_curves[2].set_xlim(0, len(history['train_f1']) - 1)
            for spine in axes_curves[2].spines.values():
                spine.set_edgecolor('#BDBDBD')
                spine.set_linewidth(1.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{full_model_name}_train_val_curves_detailed.png"),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_curves)
    print(f"  [VISUALIZATION] Publication-quality training curves saved for {full_model_name}")

