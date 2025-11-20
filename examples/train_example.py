"""
Example training script for TS-GCN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from tsgcn import TSGCN, FXDataLoader, Trainer
from config import Config


def generate_synthetic_fx_data(num_samples=1000, num_nodes=10):
    """
    Generate synthetic FX data for demonstration
    
    Args:
        num_samples (int): Number of time steps
        num_nodes (int): Number of FX pairs
    
    Returns:
        np.ndarray: Synthetic FX data [num_samples, num_nodes, 1]
    """
    np.random.seed(42)
    
    # Generate correlated time series to simulate FX pairs
    data = []
    base_series = np.cumsum(np.random.randn(num_samples)) * 0.01 + 1.0
    
    for i in range(num_nodes):
        # Add correlation with base series and some noise
        correlation = 0.5 + 0.5 * np.random.rand()
        noise = np.random.randn(num_samples) * 0.005
        series = base_series * correlation + noise + i * 0.1
        data.append(series)
    
    data = np.array(data).T  # [num_samples, num_nodes]
    data = data.reshape(num_samples, num_nodes, 1)  # [num_samples, num_nodes, 1]
    
    return data


def main():
    """Main training function"""
    
    # Display configuration
    Config.display()
    
    print("\n" + "=" * 50)
    print("Generating Synthetic FX Data")
    print("=" * 50)
    
    # Generate synthetic data
    data = generate_synthetic_fx_data(num_samples=1000, num_nodes=Config.NUM_NODES)
    print(f"Data shape: {data.shape}")
    
    # Initialize data loader
    print("\n" + "=" * 50)
    print("Loading and Preprocessing Data")
    print("=" * 50)
    
    data_loader = FXDataLoader(seq_len=Config.SEQ_LEN, pred_len=Config.PRED_LEN)
    train_data, val_data, test_data, adj_matrix = data_loader.load_data(data)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data, adj_matrix, batch_size=Config.BATCH_SIZE
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n" + "=" * 50)
    print("Initializing TS-GCN Model")
    print("=" * 50)
    
    model = TSGCN(
        num_nodes=Config.NUM_NODES,
        input_dim=Config.INPUT_DIM,
        gcn_hidden_dim=Config.GCN_HIDDEN_DIM,
        gru_hidden_dim=Config.GRU_HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        num_gcn_layers=Config.NUM_GCN_LAYERS,
        dropout=Config.DROPOUT
    )
    
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        scheduler=scheduler
    )
    
    # Train model
    print("\n" + "=" * 50)
    print("Training TS-GCN Model")
    print("=" * 50)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Test model
    print("\n" + "=" * 50)
    print("Testing TS-GCN Model")
    print("=" * 50)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    test_loss, test_metrics, predictions, targets = trainer.test(test_loader)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == '__main__':
    main()
