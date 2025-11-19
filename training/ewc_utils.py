"""
Elastic Weight Consolidation (EWC) Utilities for Continual Learning

This module provides utilities for computing Fisher Information Matrix and
applying EWC regularization to prevent catastrophic forgetting in sequential
learning scenarios.

Key Components:
    - compute_fisher_information: Computes diagonal Fisher Information Matrix
    - ewc_loss_fn: Computes EWC regularization penalty
    - save_fisher_params: Saves Fisher matrix and model parameters to disk
    - load_fisher_params: Loads Fisher matrix and model parameters from disk

Fisher Information measures parameter importance by averaging squared gradients
of the loss over a dataset. EWC uses this to penalize changes to important
parameters when learning new tasks.

References:
    Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural
    networks." PNAS 114(13): 3521-3526.

Author: GenEEG Team
Date: 2025
"""

import torch
import os
from typing import Dict, Tuple, Optional, Callable, Union
from torch.utils.data import DataLoader


def compute_fisher_information(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device,
    num_batches: Optional[int] = 100,
    amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher Information Matrix by averaging squared gradients.
    
    The Fisher Information Matrix measures the importance of each parameter by
    computing the expected squared gradient of the loss. This diagonal approximation
    is efficient and works well for EWC in practice.
    
    Args:
        model: PyTorch model to compute Fisher Information for
        data_loader: DataLoader providing batches of data
        loss_fn: Loss function that accepts (model, x) and returns a scalar tensor
                 with requires_grad=True. Must NOT call .item() or .detach().
        device: Device to run computation on (cuda/cpu)
        num_batches: Maximum number of batches to use (None = use all batches).
                     Default 100 for efficiency.
        amp: Whether to use automatic mixed precision (default: False)
    
    Returns:
        Dictionary mapping parameter names to Fisher Information tensors (on CPU).
        Each tensor has the same shape as the corresponding parameter.
    
    Raises:
        RuntimeError: If loss_fn returns non-tensor or tensor without requires_grad
    
    Example:
        >>> def vae_loss_fn(model, x):
        ...     recon, mu, logvar = model(x)
        ...     return F.mse_loss(recon, x) + 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
        >>> fisher = compute_fisher_information(vae, train_loader, vae_loss_fn, device)
        >>> # Use fisher for EWC regularization
    
    Note:
        - Model must be in training mode
        - All parameters must have requires_grad=True
        - Fisher tensors are stored on CPU to avoid GPU memory issues
        - Use num_batches to trade off accuracy vs. computation time
    """
    model.to(device)
    
    # Ensure we can take gradients through the whole model
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    
    # Initialize accumulator (on CPU to avoid GPU memory bloat)
    fisher = {
        name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    
    # Autocast if requested
    use_amp = bool(amp and (getattr(device, "type", "") == "cuda"))
    batches = 0
    
    for batch in data_loader:
        # CRITICAL: Pass the full batch to loss_fn, which will handle extraction
        # Different loss functions expect different batch formats:
        # - VAE loss: expects single tensor (x)
        # - LDM loss: expects tuple (z0, x_raw, features, labels)
        # The loss_fn is responsible for unpacking the batch appropriately
        
        # Clear gradients
        model.zero_grad(set_to_none=True)
        
        # Forward pass & compute scalar loss (must keep graph!)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model, batch)
            
            # Validate: must be a tensor that requires grad
            if not torch.is_tensor(loss):
                raise RuntimeError(
                    "loss_fn must return a *tensor*, not a float. "
                    "Ensure loss_fn does not call .item() or convert to Python scalar."
                )
            if not loss.requires_grad:
                raise RuntimeError(
                    "loss returned by loss_fn does not require grad. "
                    "Ensure no .item()/.detach()/no_grad() was applied before returning."
                )
        
        # Backpropagate to compute gradients
        loss.backward()
        
        # Accumulate squared gradients (diagonal Fisher approximation)
        for name, p in model.named_parameters():
            if (p.grad is not None) and p.requires_grad:
                fisher[name] += (p.grad.detach().float().cpu() ** 2)
        
        # CRITICAL: Clear GPU memory after each batch to avoid OOM
        del batch, loss
        if hasattr(device, 'type') and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        batches += 1
        if (num_batches is not None) and (batches >= num_batches):
            break
    
    # Average over all batches
    denom = max(1, batches)
    for name in fisher:
        fisher[name] /= denom
    
    return fisher


def ewc_loss_fn(
    model: torch.nn.Module,
    old_params: Dict[str, torch.Tensor],
    fisher_info: Dict[str, torch.Tensor],
    lambda_ewc: float = 1.0,
) -> torch.Tensor:
    """
    Compute Elastic Weight Consolidation (EWC) regularization penalty.
    
    EWC penalizes changes to important parameters (as measured by Fisher Information)
    when learning new tasks. This helps prevent catastrophic forgetting.
    
    The penalty is computed as:
        L_EWC = (lambda_ewc / 2) * sum_i F_i * (theta_i - theta_i^old)^2
    
    where:
        - F_i is the Fisher Information for parameter i
        - theta_i is the current parameter value
        - theta_i^old is the parameter value after training on the previous task
    
    Args:
        model: Current PyTorch model
        old_params: Dictionary of parameter values from previous task (on CPU or GPU)
        fisher_info: Dictionary of Fisher Information matrices (on CPU)
        lambda_ewc: Regularization strength (default: 1.0)
                    Higher values = stronger preservation of old parameters
    
    Returns:
        Scalar tensor representing the EWC penalty (on same device as model)
    
    Example:
        >>> # After training on task 1
        >>> old_params = {name: p.clone().detach() for name, p in model.named_parameters()}
        >>> fisher = compute_fisher_information(model, task1_loader, loss_fn, device)
        >>> 
        >>> # During training on task 2
        >>> task_loss = F.cross_entropy(output, target)
        >>> ewc_penalty = ewc_loss_fn(model, old_params, fisher, lambda_ewc=5000)
        >>> total_loss = task_loss + ewc_penalty
        >>> total_loss.backward()
    
    Note:
        - old_params and fisher_info must have matching keys
        - Missing parameters in fisher_info are silently ignored
        - lambda_ewc should be tuned based on task similarity and data size
          (typical range: 100 to 10000)
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for name, param in model.named_parameters():
        if name in fisher_info:
            # Move old_params and fisher to current device if needed
            old_p = old_params[name].to(param.device)
            fisher = fisher_info[name].to(param.device)
            
            # Compute weighted squared distance from old parameters
            loss += (fisher * (param - old_p) ** 2).sum()
    
    return lambda_ewc * loss


def save_fisher_params(
    fisher_info: Dict[str, torch.Tensor],
    model_params: Dict[str, torch.Tensor],
    save_dir: str,
    task_id: Union[int, str],
) -> Tuple[str, str]:
    """
    Save Fisher Information Matrix and model parameters to disk.
    
    Args:
        fisher_info: Dictionary of Fisher Information tensors
        model_params: Dictionary of model parameter tensors
        save_dir: Directory to save files to
        task_id: Task identifier (e.g., patient ID, fold number)
    
    Returns:
        Tuple of (fisher_path, params_path) where files were saved
    
    Example:
        >>> fisher = compute_fisher_information(model, loader, loss_fn, device)
        >>> params = {name: p.clone().detach().cpu() for name, p in model.named_parameters()}
        >>> save_fisher_params(fisher, params, "./checkpoints", task_id="patient_01")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fisher_path = os.path.join(save_dir, f"fisher_task_{task_id}.pt")
    params_path = os.path.join(save_dir, f"params_task_{task_id}.pt")
    
    torch.save(fisher_info, fisher_path)
    torch.save(model_params, params_path)
    
    return fisher_path, params_path


def load_fisher_params(
    save_dir: str,
    task_id: Union[int, str],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load Fisher Information Matrix and model parameters from disk.
    
    Args:
        save_dir: Directory where files are saved
        task_id: Task identifier used during saving
        device: Device to load tensors to (default: CPU)
    
    Returns:
        Tuple of (fisher_info, model_params)
    
    Raises:
        FileNotFoundError: If files do not exist
    
    Example:
        >>> fisher, old_params = load_fisher_params("./checkpoints", "patient_01")
        >>> ewc_loss = ewc_loss_fn(model, old_params, fisher, lambda_ewc=5000)
    """
    fisher_path = os.path.join(save_dir, f"fisher_task_{task_id}.pt")
    params_path = os.path.join(save_dir, f"params_task_{task_id}.pt")
    
    if device is None:
        device = torch.device("cpu")
    
    fisher_info = torch.load(fisher_path, map_location=device)
    model_params = torch.load(params_path, map_location=device)
    
    return fisher_info, model_params


def accumulate_fisher(
    fisher_list: list[Dict[str, torch.Tensor]],
    weights: Optional[list[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Accumulate Fisher Information matrices from multiple tasks.
    
    This is useful for continual learning with multiple previous tasks, where
    we want to protect parameters important for all tasks, not just the most recent.
    
    Args:
        fisher_list: List of Fisher Information dictionaries from different tasks
        weights: Optional weights for each Fisher matrix (default: uniform)
                 Should sum to 1.0 for proper averaging
    
    Returns:
        Accumulated Fisher Information dictionary
    
    Example:
        >>> fishers = [fisher_task1, fisher_task2, fisher_task3]
        >>> # Weight recent tasks more heavily
        >>> weights = [0.2, 0.3, 0.5]
        >>> accumulated = accumulate_fisher(fishers, weights)
        >>> ewc_loss = ewc_loss_fn(model, old_params, accumulated, lambda_ewc=5000)
    
    Note:
        - All Fisher matrices must have the same parameter names
        - Weights are normalized if they don't sum to 1.0
    """
    if not fisher_list:
        raise ValueError("fisher_list cannot be empty")
    
    # Default to uniform weights
    if weights is None:
        weights = [1.0 / len(fisher_list)] * len(fisher_list)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Initialize accumulator from first Fisher matrix
    accumulated = {
        name: torch.zeros_like(tensor)
        for name, tensor in fisher_list[0].items()
    }
    
    # Accumulate weighted Fisher matrices
    for fisher, weight in zip(fisher_list, weights):
        for name in accumulated:
            accumulated[name] += weight * fisher[name]
    
    return accumulated
