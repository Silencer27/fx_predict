"""
Visualization script for TS-GCN predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tsgcn import TSGCN, FXDataLoader, Trainer
import torch
import torch.nn as nn
from config import Config


def generate_synthetic_fx_data(num_samples=500, num_nodes=10):
    """Generate synthetic FX data"""
    np.random.seed(42)
    
    data = []
    base_series = np.cumsum(np.random.randn(num_samples)) * 0.01 + 1.0
    
    for i in range(num_nodes):
        correlation = 0.5 + 0.5 * np.random.rand()
        noise = np.random.randn(num_samples) * 0.005
        series = base_series * correlation + noise + i * 0.1
        data.append(series)
    
    data = np.array(data).T
    data = data.reshape(num_samples, num_nodes, 1)
    
    return data


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(history['val_losses'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('TS-GCN Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_predictions(predictions, targets, num_nodes_to_plot=3, save_path='predictions.png'):
    """Plot predictions vs actual values"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Plot first few nodes
    fig, axes = plt.subplots(num_nodes_to_plot, 1, figsize=(12, 3 * num_nodes_to_plot))
    
    if num_nodes_to_plot == 1:
        axes = [axes]
    
    for i in range(num_nodes_to_plot):
        ax = axes[i]
        
        # Plot actual vs predicted
        x = np.arange(len(predictions[:, i, 0]))
        ax.plot(x, targets[:, i, 0], label=f'Actual FX Pair {i+1}', 
                linewidth=2, alpha=0.7, marker='o', markersize=3)
        ax.plot(x, predictions[:, i, 0], label=f'Predicted FX Pair {i+1}', 
                linewidth=2, alpha=0.7, marker='x', markersize=3)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('FX Rate', fontsize=11)
        ax.set_title(f'Predictions vs Actual - FX Pair {i+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions plot saved to {save_path}")
    plt.close()


def plot_adjacency_matrix(adj_matrix, save_path='adjacency_matrix.png'):
    """Plot adjacency matrix heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Edge Weight')
    plt.xlabel('FX Pair Index', fontsize=12)
    plt.ylabel('FX Pair Index', fontsize=12)
    plt.title('Adjacency Matrix (Correlation-based Graph)', fontsize=14, fontweight='bold')
    
    # Add grid
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            plt.text(j, i, f'{adj_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Adjacency matrix plot saved to {save_path}")
    plt.close()


def main():
    """Main function to train and visualize"""
    
    print("=" * 60)
    print("TS-GCN Visualization Example")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating synthetic FX data...")
    data = generate_synthetic_fx_data(num_samples=500, num_nodes=Config.NUM_NODES)
    
    # Load and preprocess
    print("Loading and preprocessing data...")
    data_loader = FXDataLoader(seq_len=Config.SEQ_LEN, pred_len=Config.PRED_LEN)
    train_data, val_data, test_data, adj_matrix = data_loader.load_data(data)
    
    # Visualize adjacency matrix
    print("\nVisualizing adjacency matrix...")
    plot_adjacency_matrix(adj_matrix)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data, adj_matrix, batch_size=Config.BATCH_SIZE
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = TSGCN(
        num_nodes=Config.NUM_NODES,
        input_dim=Config.INPUT_DIM,
        gcn_hidden_dim=Config.GCN_HIDDEN_DIM,
        gru_hidden_dim=Config.GRU_HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        num_gcn_layers=Config.NUM_GCN_LAYERS,
        dropout=Config.DROPOUT
    )
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    trainer = Trainer(model, optimizer, criterion, device=Config.DEVICE, scheduler=scheduler)
    
    # Train with limited epochs for quick demo
    print("\nTraining model (20 epochs)...")
    history = trainer.train(train_loader, val_loader, num_epochs=20, early_stopping_patience=10)
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history(history)
    
    # Test model
    print("\nTesting model...")
    trainer.load_checkpoint('best_model.pt')
    test_loss, test_metrics, predictions, targets = trainer.test(test_loader)
    
    # Plot predictions
    print("\nGenerating predictions plot...")
    plot_predictions(predictions, targets, num_nodes_to_plot=3)
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check the generated PNG files.")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - training_history.png: Training and validation loss curves")
    print("  - predictions.png: Actual vs predicted values for first 3 FX pairs")
    print("  - adjacency_matrix.png: Correlation-based graph structure")


if __name__ == '__main__':
    main()
