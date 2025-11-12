"""
Training script for TSGCN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm

from fx_predict.models.tsgcn import TSGCN
from fx_predict.data.dataset import FXDataset
from fx_predict.data.graph_builder import build_adjacency_matrix, compute_correlation_matrix
from fx_predict.utils.metrics import calculate_metrics
from fx_predict.utils.visualization import plot_training_history
from fx_predict.config.config import load_config, get_default_config


def train_epoch(model, dataloader, criterion, optimizer, device, adj):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x, adj)
        
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, adj):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x, adj)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(config, data, save_dir='checkpoints'):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
        data: FX data array (timesteps, num_currencies, num_features)
        save_dir: Directory to save model checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FXDataset(
        data,
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        train=True,
        train_ratio=config['training']['train_ratio']
    )
    
    val_dataset = FXDataset(
        data,
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        train=False,
        train_ratio=config['training']['train_ratio']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Build adjacency matrix
    if config['data']['graph_method'] == 'correlation':
        corr_matrix = compute_correlation_matrix(data)
        adj = build_adjacency_matrix(
            config['model']['num_nodes'],
            method='correlation',
            correlation_matrix=corr_matrix,
            threshold=config['data']['correlation_threshold']
        )
    else:
        adj = build_adjacency_matrix(
            config['model']['num_nodes'],
            method=config['data']['graph_method']
        )
    
    adj = adj.to(device)
    
    # Create model
    model = TSGCN(
        num_nodes=config['model']['num_nodes'],
        num_features=config['model']['num_features'],
        temporal_channels=config['model']['temporal_channels'],
        spatial_channels=config['model']['spatial_channels'],
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters()}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, adj)
        val_loss = validate(model, val_loader, criterion, device, adj)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, 
                          save_path=os.path.join(save_dir, 'training_history.png'))
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train TSGCN for FX prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FX data file (numpy array)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found, using default configuration")
        config = get_default_config()
    
    # Load data
    data = np.load(args.data)
    print(f"Loaded data with shape: {data.shape}")
    
    # Train model
    train(config, data, args.save_dir)


if __name__ == '__main__':
    main()
