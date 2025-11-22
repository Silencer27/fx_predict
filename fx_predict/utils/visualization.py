"""
Visualization utilities for FX predictions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true, y_pred, currency_names=None, save_path=None):
    """
    Plot true vs predicted values
    
    Args:
        y_true: True values (timesteps, num_currencies)
        y_pred: Predicted values (timesteps, num_currencies)
        currency_names: List of currency names
        save_path: Path to save the plot
    """
    num_currencies = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if num_currencies == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if currency_names is None:
        currency_names = [f'Currency {i+1}' for i in range(num_currencies)]
    
    # Create subplots
    fig, axes = plt.subplots(num_currencies, 1, figsize=(12, 4*num_currencies))
    
    if num_currencies == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(y_true[:, i], label='True', linewidth=2)
        ax.plot(y_pred[:, i], label='Predicted', linewidth=2, alpha=0.7)
        ax.set_title(f'{currency_names[i]} Exchange Rate')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Exchange Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss history
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
