"""Utility functions for GenEEG project."""

import torch
import gc


def cleanup_gpu_memory():
    """Enhanced GPU memory cleanup function."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        print(f"Warning: GPU memory cleanup failed: {e}")


def safe_tensor_operation(func, *args, **kwargs):
    """Safely execute tensor operations with automatic error handling and memory cleanup.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func execution
    
    Raises:
        RuntimeError: If operation fails after retry
    """
    try:
        result = func(*args, **kwargs)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA out of memory detected. Cleaning up and retrying...")
            cleanup_gpu_memory()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
                raise retry_e
        else:
            raise e
    except Exception as e:
        print(f"Tensor operation failed: {e}")
        raise e


def unscale_data(scaled_data_np, means_np, stds_np):
    """Unscales normalized data back to original range.
    
    Args:
        scaled_data_np: Normalized data array
        means_np: Mean values used for normalization
        stds_np: Standard deviation values used for normalization
    
    Returns:
        Unscaled data array
    """
    try:
        means_b = torch.asarray(means_np).reshape(1, -1, 1)
        stds_b = torch.asarray(stds_np).reshape(1, -1, 1)
        return scaled_data_np * stds_b + means_b
    except Exception as e:
        print(f"Warning: Error in unscale_data: {e}")
        # Fallback to simple broadcasting
        return scaled_data_np * stds_np + means_np


__all__ = ['cleanup_gpu_memory', 'safe_tensor_operation', 'unscale_data']
