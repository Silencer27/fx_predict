"""
Evaluation metrics for FX prediction
"""

import numpy as np
import torch


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    
    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values
        targets (np.ndarray or torch.Tensor): True values
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def directional_accuracy(predictions, targets):
    """
    Calculate directional accuracy (whether prediction correctly predicts direction of change)
    
    Args:
        predictions (np.ndarray): Predicted values [num_samples, ...]
        targets (np.ndarray): True values [num_samples, ...]
    
    Returns:
        float: Directional accuracy (0-1)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # For time series, we need previous values to compute direction
    # This is a simplified version assuming we have the direction
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    
    accuracy = np.mean(pred_direction == target_direction)
    return accuracy
