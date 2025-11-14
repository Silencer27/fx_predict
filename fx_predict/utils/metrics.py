"""
Metrics for evaluating FX predictions
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (percentage of correct direction predictions)
    
    Args:
        y_true: True values (with previous values for computing direction)
        y_pred: Predicted values
        
    Returns:
        Directional accuracy as percentage
    """
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy
