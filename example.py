"""
Example usage of TSGCN for FX prediction
"""

import numpy as np
import torch
from fx_predict.models.tsgcn import TSGCN
from fx_predict.data.dataset import FXDataset
from fx_predict.data.graph_builder import build_adjacency_matrix
from fx_predict.config.config import get_default_config
from torch.utils.data import DataLoader


def generate_sample_data(num_timesteps=1000, num_currencies=7, num_features=4):
    """
    Generate sample FX data for demonstration
    
    Args:
        num_timesteps: Number of time steps
        num_currencies: Number of currencies
        num_features: Number of features (OHLC)
        
    Returns:
        Synthetic FX data
    """
    np.random.seed(42)
    
    # Generate base trends
    trends = np.random.randn(num_currencies) * 0.0001
    
    # Generate data with trends and random walk
    data = np.zeros((num_timesteps, num_currencies, num_features))
    
    for i in range(num_currencies):
        # Initialize with base value around 1.0
        base_value = 1.0 + np.random.randn() * 0.1
        
        for t in range(num_timesteps):
            if t == 0:
                # Open, High, Low, Close for first timestep
                data[t, i, 0] = base_value  # Open
                data[t, i, 1] = base_value + abs(np.random.randn() * 0.01)  # High
                data[t, i, 2] = base_value - abs(np.random.randn() * 0.01)  # Low
                data[t, i, 3] = base_value + np.random.randn() * 0.005  # Close
            else:
                # Use previous close as current open
                data[t, i, 0] = data[t-1, i, 3]
                
                # Add trend and random walk
                change = trends[i] + np.random.randn() * 0.005
                
                # Calculate high, low, close
                data[t, i, 3] = data[t, i, 0] + change
                data[t, i, 1] = max(data[t, i, 0], data[t, i, 3]) + abs(np.random.randn() * 0.002)
                data[t, i, 2] = min(data[t, i, 0], data[t, i, 3]) - abs(np.random.randn() * 0.002)
    
    return data


def main():
    print("=" * 80)
    print("TSGCN for FX Prediction - Example Usage")
    print("=" * 80)
    
    # Get default configuration
    config = get_default_config()
    
    # Generate sample data
    print("\n1. Generating sample FX data...")
    data = generate_sample_data(
        num_timesteps=1000,
        num_currencies=config['model']['num_nodes'],
        num_features=config['model']['num_features']
    )
    print(f"   Data shape: {data.shape}")
    print(f"   (timesteps, num_currencies, num_features)")
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = FXDataset(
        data,
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        train=True,
        train_ratio=0.8
    )
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Build adjacency matrix
    print("\n3. Building adjacency matrix...")
    adj = build_adjacency_matrix(
        config['model']['num_nodes'],
        method='fully_connected'
    )
    print(f"   Adjacency matrix shape: {adj.shape}")
    
    # Create model
    print("\n4. Creating TSGCN model...")
    model = TSGCN(
        num_nodes=config['model']['num_nodes'],
        num_features=config['model']['num_features'],
        temporal_channels=config['model']['temporal_channels'],
        spatial_channels=config['model']['spatial_channels'],
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout']
    )
    print(f"   Model created with {model.count_parameters()} parameters")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    sample_x, sample_y = next(iter(dataloader))
    print(f"   Input shape: {sample_x.shape}")
    print(f"   Target shape: {sample_y.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(sample_x, adj)
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your FX data in shape (timesteps, num_currencies, num_features)")
    print("2. Save data as numpy array: np.save('fx_data.npy', data)")
    print("3. Train model: python train.py --data fx_data.npy")
    print("4. Evaluate model: python evaluate.py --data fx_data.npy")
    print("=" * 80)


if __name__ == '__main__':
    main()
