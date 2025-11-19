"""
Continual Learning Utilities for GenEEG.

This module implements Elastic Weight Consolidation (EWC) and Experience Replay
for continual learning in the leave-one-patient-out (LOPO) setting.

Classes:
    ExperienceReplayBuffer: Store and sample from past experiences
    
Functions:
    compute_fisher_information: Calculate Fisher Information Matrix for EWC
    ewc_loss: Compute EWC regularization loss
"""

import logging
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for continual learning.
    
    Stores samples from previous tasks and allows sampling for 
    replay-based continual learning strategies.
    
    Args:
        max_size: Maximum number of samples to store
        device: PyTorch device for tensor operations
        
    Attributes:
        buffer_x: Stored input samples
        buffer_y: Stored labels
        max_size: Maximum buffer capacity
        current_size: Current number of samples in buffer
    """
    
    def __init__(self, max_size: int = 1000, device: str = 'cpu'):
        """Initialize Experience Replay Buffer."""
        self.max_size = max_size
        self.device = device
        self.buffer_x = []
        self.buffer_y = []
        self.current_size = 0
        
        logger.info(f"Initialized ExperienceReplayBuffer with max_size={max_size}")
    
    def add_samples(self, x: torch.Tensor, y: torch.Tensor):
        """
        Add samples to the buffer using reservoir sampling.
        
        Args:
            x: Input samples (batch_size, num_channels, seq_length)
            y: Labels (batch_size,)
        """
        batch_size = x.size(0)
        
        for i in range(batch_size):
            if self.current_size < self.max_size:
                # Buffer not full, just append
                self.buffer_x.append(x[i].cpu())
                self.buffer_y.append(y[i].cpu())
                self.current_size += 1
            else:
                # Buffer full, reservoir sampling
                idx = random.randint(0, self.current_size)
                if idx < self.max_size:
                    self.buffer_x[idx] = x[i].cpu()
                    self.buffer_y[idx] = y[i].cpu()
            
            self.current_size += 1
        
        logger.debug(f"Added {batch_size} samples to buffer. Current size: {len(self)}")
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            x_batch: Sampled inputs (batch_size, num_channels, seq_length)
            y_batch: Sampled labels (batch_size,)
        """
        if len(self) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample indices
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        
        # Gather samples
        x_batch = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        y_batch = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        
        return x_batch, y_batch
    
    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return min(len(self.buffer_x), self.max_size)
    
    def clear(self):
        """Clear all samples from buffer."""
        self.buffer_x = []
        self.buffer_y = []
        self.current_size = 0
        logger.info("Buffer cleared")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self) == 0:
            return {
                'total_samples': 0,
                'capacity': self.max_size,
                'utilization': 0.0
            }
        
        y_tensor = torch.stack(self.buffer_y[:len(self)])
        unique_labels, counts = torch.unique(y_tensor, return_counts=True)
        
        stats = {
            'total_samples': len(self),
            'capacity': self.max_size,
            'utilization': len(self) / self.max_size,
            'class_distribution': {int(label): int(count) for label, count in zip(unique_labels, counts)}
        }
        
        return stats


def compute_fisher_information(
    model: nn.Module,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str = 'cpu',
    num_samples: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute Fisher Information Matrix for Elastic Weight Consolidation.
    
    The Fisher Information Matrix approximates the importance of each parameter
    for the current task. Parameters with high Fisher information should be
    protected from large changes when learning new tasks.
    
    Args:
        model: PyTorch model
        data_loader: List of (input, label) tuples
        device: Device for computations
        num_samples: Maximum number of samples to use (None = use all)
        
    Returns:
        fisher_dict: Dictionary mapping parameter names to Fisher information values
    """
    logger.info("Computing Fisher Information Matrix...")
    
    model.eval()
    fisher_dict = {}
    
    # Initialize Fisher information to zeros
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    # Compute Fisher information from gradients
    num_processed = 0
    for data, target in data_loader:
        if num_samples is not None and num_processed >= num_samples:
            break
        
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        model.zero_grad()
        output = model(data)
        
        # Compute loss (negative log-likelihood)
        if output.dim() == 2:
            # Classification output
            loss = torch.nn.functional.cross_entropy(output, target)
        else:
            # Handle other output types
            loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Accumulate squared gradients (Fisher information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        num_processed += 1
    
    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= num_processed
    
    logger.info(f"Fisher Information computed from {num_processed} samples")
    
    return fisher_dict


def ewc_loss(
    model: nn.Module,
    fisher_dict: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    ewc_lambda: float = 1000.0
) -> torch.Tensor:
    """
    Compute Elastic Weight Consolidation (EWC) regularization loss.
    
    EWC prevents catastrophic forgetting by penalizing changes to parameters
    that were important for previous tasks.
    
    Loss = λ/2 * Σ F_i (θ_i - θ_i*)²
    
    where:
        - F_i is the Fisher information for parameter i
        - θ_i is the current parameter value
        - θ_i* is the parameter value after learning previous tasks
        - λ is the regularization strength
    
    Args:
        model: Current model
        fisher_dict: Fisher information for each parameter
        old_params: Parameter values from previous task
        ewc_lambda: Regularization strength
        
    Returns:
        ewc_loss: EWC regularization loss
    """
    loss = 0.0
    
    for name, param in model.named_parameters():
        if name in fisher_dict and name in old_params:
            # EWC penalty: fisher * (param - old_param)^2
            fisher = fisher_dict[name]
            old_param = old_params[name]
            loss += (fisher * (param - old_param) ** 2).sum()
    
    return ewc_lambda * loss / 2


def save_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Save a copy of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        param_dict: Dictionary of parameter tensors (detached copies)
    """
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = param.data.clone()
    
    return param_dict


class ContinualLearner:
    """
    Wrapper class for continual learning with EWC and Experience Replay.
    
    Combines EWC regularization with experience replay for effective
    continual learning in LOPO cross-validation.
    
    Args:
        model: PyTorch model to train
        ewc_lambda: EWC regularization strength
        replay_batch_size: Batch size for experience replay
        replay_buffer_size: Maximum size of replay buffer
        device: Device for computations
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        replay_batch_size: int = 32,
        replay_buffer_size: int = 1000,
        device: str = 'cpu'
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.replay_batch_size = replay_batch_size
        self.device = device
        
        # Initialize components
        self.replay_buffer = ExperienceReplayBuffer(max_size=replay_buffer_size, device=device)
        self.fisher_dict = {}
        self.old_params = {}
        self.task_count = 0
        
        logger.info(f"Initialized ContinualLearner with EWC lambda={ewc_lambda}, "
                   f"replay buffer size={replay_buffer_size}")
    
    def consolidate_task(self, data_loader: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Consolidate knowledge after completing a task.
        
        Computes Fisher information and saves current parameters for EWC.
        
        Args:
            data_loader: Data from the completed task
        """
        logger.info(f"Consolidating task {self.task_count + 1}...")
        
        # Compute Fisher information
        self.fisher_dict = compute_fisher_information(
            self.model,
            data_loader,
            device=self.device,
            num_samples=100  # Use subset for efficiency
        )
        
        # Save current parameters
        self.old_params = save_model_params(self.model)
        
        self.task_count += 1
        logger.info(f"Task {self.task_count} consolidated successfully")
    
    def get_ewc_loss(self) -> torch.Tensor:
        """
        Get EWC regularization loss for current model state.
        
        Returns:
            EWC loss (0 if no previous tasks)
        """
        if self.task_count == 0:
            return torch.tensor(0.0).to(self.device)
        
        return ewc_loss(self.model, self.fisher_dict, self.old_params, self.ewc_lambda)
    
    def get_replay_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample a batch from replay buffer if available.
        
        Returns:
            (x_replay, y_replay) or None if buffer is empty
        """
        if len(self.replay_buffer) == 0:
            return None
        
        try:
            return self.replay_buffer.sample(self.replay_batch_size)
        except ValueError:
            return None
    
    def add_to_replay_buffer(self, x: torch.Tensor, y: torch.Tensor):
        """
        Add samples to replay buffer.
        
        Args:
            x: Input samples
            y: Labels
        """
        self.replay_buffer.add_samples(x, y)
    
    def get_statistics(self) -> Dict:
        """
        Get continual learning statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'task_count': self.task_count,
            'ewc_lambda': self.ewc_lambda,
            'replay_buffer': self.replay_buffer.get_statistics(),
            'fisher_mean': np.mean([v.mean().item() for v in self.fisher_dict.values()]) 
                          if self.fisher_dict else 0.0
        }


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Continual Learning Components...")
    
    # Test Experience Replay Buffer
    print("\n1. Testing ExperienceReplayBuffer...")
    buffer = ExperienceReplayBuffer(max_size=100)
    
    # Add some samples
    x = torch.randn(10, 16, 3072)
    y = torch.randint(0, 3, (10,))
    buffer.add_samples(x, y)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Statistics: {buffer.get_statistics()}")
    
    # Sample from buffer
    x_sample, y_sample = buffer.sample(5)
    print(f"   Sampled batch: {x_sample.shape}, {y_sample.shape}")
    
    # Test Fisher Information
    print("\n2. Testing Fisher Information computation...")
    from models import CNNBiLSTM
    
    model = CNNBiLSTM(num_classes=3, num_channels=16)
    data_loader = [(torch.randn(8, 16, 3072), torch.randint(0, 3, (8,))) for _ in range(5)]
    
    fisher = compute_fisher_information(model, data_loader, num_samples=3)
    print(f"   Fisher dict has {len(fisher)} parameters")
    print(f"   Mean Fisher value: {np.mean([v.mean().item() for v in fisher.values()]):.6f}")
    
    # Test EWC Loss
    print("\n3. Testing EWC loss...")
    old_params = save_model_params(model)
    
    # Make small parameter change
    for param in model.parameters():
        param.data += torch.randn_like(param) * 0.01
    
    loss = ewc_loss(model, fisher, old_params, ewc_lambda=1000.0)
    print(f"   EWC loss: {loss.item():.4f}")
    
    # Test ContinualLearner
    print("\n4. Testing ContinualLearner...")
    learner = ContinualLearner(model, ewc_lambda=1000.0, replay_buffer_size=100)
    
    learner.add_to_replay_buffer(x, y)
    learner.consolidate_task(data_loader)
    
    print(f"   Statistics: {learner.get_statistics()}")
    
    print("\n✅ All tests passed!")
