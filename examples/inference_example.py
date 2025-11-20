"""
Example inference script for TS-GCN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tsgcn import TSGCN, FXDataLoader
from config import Config


def load_model(model_path, device='cpu'):
    """
    Load trained TS-GCN model
    
    Args:
        model_path (str): Path to model checkpoint
        device (str): Device to load model on
    
    Returns:
        TSGCN: Loaded model
    """
    # Initialize model
    model = TSGCN(
        num_nodes=Config.NUM_NODES,
        input_dim=Config.INPUT_DIM,
        gcn_hidden_dim=Config.GCN_HIDDEN_DIM,
        gru_hidden_dim=Config.GRU_HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        num_gcn_layers=Config.NUM_GCN_LAYERS,
        dropout=Config.DROPOUT
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict(model, data, adj_matrix, data_loader, device='cpu'):
    """
    Make predictions using trained model
    
    Args:
        model (TSGCN): Trained model
        data (np.ndarray): Input data [seq_len, num_nodes, features]
        adj_matrix (np.ndarray): Adjacency matrix
        data_loader (FXDataLoader): Data loader for inverse transformation
        device (str): Device to run inference on
    
    Returns:
        np.ndarray: Predictions in original scale
    """
    # Convert to tensor
    x = torch.FloatTensor(data).unsqueeze(0).to(device)  # [1, seq_len, num_nodes, features]
    adj = torch.FloatTensor(adj_matrix).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(x, adj)  # [1, num_nodes, output_dim]
    
    # Convert to numpy
    prediction = prediction.cpu().numpy()
    
    # Inverse transform to original scale
    prediction = data_loader.inverse_transform(prediction)
    
    return prediction


def main():
    """Main inference function"""
    
    print("=" * 50)
    print("TS-GCN Inference Example")
    print("=" * 50)
    
    # Load model
    model_path = 'best_model.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_example.py first to train a model.")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, device=Config.DEVICE)
    print("Model loaded successfully!")
    
    # Generate sample input data
    print("\nGenerating sample input data...")
    np.random.seed(123)
    
    # Create sample time series data
    seq_len = Config.SEQ_LEN
    num_nodes = Config.NUM_NODES
    
    # Sample FX rates (normalized)
    sample_data = np.random.randn(seq_len, num_nodes, 1) * 0.01 + 1.0
    
    print(f"Input data shape: {sample_data.shape}")
    print(f"Input data (last 5 time steps):")
    print(sample_data[-5:, :3, 0])  # Show last 5 steps for first 3 nodes
    
    # Initialize data loader for preprocessing
    data_loader = FXDataLoader(seq_len=Config.SEQ_LEN, pred_len=Config.PRED_LEN)
    
    # Normalize input data
    data_reshaped = sample_data.reshape(-1, 1)
    data_normalized = data_loader.scaler.fit_transform(data_reshaped)
    sample_data_normalized = data_normalized.reshape(seq_len, num_nodes, 1)
    
    # Create adjacency matrix
    adj_matrix = data_loader.compute_adjacency_matrix(sample_data.reshape(1, seq_len, num_nodes, 1)[0])
    
    print(f"\nAdjacency matrix shape: {adj_matrix.shape}")
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = predict(
        model=model,
        data=sample_data_normalized,
        adj_matrix=adj_matrix,
        data_loader=data_loader,
        device=Config.DEVICE
    )
    
    print(f"\nPrediction shape: {prediction.shape}")
    print("\nPredicted FX rates (next time step):")
    print(prediction[0, :, 0])  # [num_nodes]
    
    print("\n" + "=" * 50)
    print("Inference Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
