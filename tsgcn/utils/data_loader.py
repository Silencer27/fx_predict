"""
Data loading and preprocessing utilities for FX data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class FXDataset(Dataset):
    """
    PyTorch Dataset for FX time series data
    
    Args:
        data (np.ndarray): FX data [num_samples, num_nodes, num_features]
        adj (np.ndarray): Adjacency matrix [num_nodes, num_nodes]
        seq_len (int): Length of input sequences
        pred_len (int): Length of prediction horizon
    """
    
    def __init__(self, data, adj, seq_len=10, pred_len=1):
        self.data = data
        self.adj = adj
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(data) - seq_len - pred_len + 1):
            seq = data[i:i + seq_len]
            target = data[i + seq_len:i + seq_len + pred_len]
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.adj),
            torch.FloatTensor(self.targets[idx])
        )


class FXDataLoader:
    """
    Data loader for FX prediction
    
    Handles data loading, preprocessing, and adjacency matrix construction
    """
    
    def __init__(self, data_path=None, seq_len=10, pred_len=1):
        """
        Initialize FX data loader
        
        Args:
            data_path (str): Path to data file (CSV format)
            seq_len (int): Length of input sequences
            pred_len (int): Length of prediction horizon
        """
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = StandardScaler()
    
    def load_data(self, data=None):
        """
        Load and preprocess FX data
        
        Args:
            data (pd.DataFrame or np.ndarray): FX data
                If DataFrame: columns represent different FX pairs
                If ndarray: shape should be [num_samples, num_nodes, num_features]
        
        Returns:
            tuple: (train_data, val_data, test_data, adj_matrix)
        """
        if data is None and self.data_path:
            # Load from CSV
            df = pd.read_csv(self.data_path, index_col=0)
            data = df.values
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        # Ensure data is 3D [num_samples, num_nodes, num_features]
        if data.ndim == 2:
            # Assume each column is a node with single feature
            data = data.reshape(data.shape[0], data.shape[1], 1)
        
        # Normalize data
        num_samples, num_nodes, num_features = data.shape
        data_reshaped = data.reshape(-1, num_features)
        data_normalized = self.scaler.fit_transform(data_reshaped)
        data = data_normalized.reshape(num_samples, num_nodes, num_features)
        
        # Split data: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Compute adjacency matrix based on correlations
        adj_matrix = self.compute_adjacency_matrix(data)
        
        return train_data, val_data, test_data, adj_matrix
    
    def compute_adjacency_matrix(self, data, threshold=0.5):
        """
        Compute adjacency matrix based on correlation between FX pairs
        
        Args:
            data (np.ndarray): FX data [num_samples, num_nodes, num_features]
            threshold (float): Correlation threshold for edge creation
        
        Returns:
            np.ndarray: Normalized adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes = data.shape[1]
        
        # Compute correlation matrix
        # Flatten to [num_samples, num_nodes] using the first feature
        data_2d = data[:, :, 0]
        corr_matrix = np.corrcoef(data_2d.T)
        
        # Apply threshold to create binary adjacency matrix
        adj = np.where(np.abs(corr_matrix) > threshold, 1, 0)
        
        # Add self-loops
        adj = adj + np.eye(num_nodes)
        
        # Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
        adj = self.normalize_adjacency(adj)
        
        return adj
    
    def normalize_adjacency(self, adj):
        """
        Normalize adjacency matrix using symmetric normalization
        
        Args:
            adj (np.ndarray): Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            np.ndarray: Normalized adjacency matrix
        """
        # Compute degree matrix
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        
        # D^(-1/2)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        
        # D^(-1/2) A D^(-1/2)
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_normalized
    
    def create_dataloaders(self, train_data, val_data, test_data, adj_matrix, batch_size=32):
        """
        Create PyTorch DataLoaders
        
        Args:
            train_data (np.ndarray): Training data
            val_data (np.ndarray): Validation data
            test_data (np.ndarray): Test data
            adj_matrix (np.ndarray): Adjacency matrix
            batch_size (int): Batch size
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_dataset = FXDataset(train_data, adj_matrix, self.seq_len, self.pred_len)
        val_dataset = FXDataset(val_data, adj_matrix, self.seq_len, self.pred_len)
        test_dataset = FXDataset(test_data, adj_matrix, self.seq_len, self.pred_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data):
        """
        Inverse transform normalized data back to original scale
        
        Args:
            data (np.ndarray): Normalized data
        
        Returns:
            np.ndarray: Original scale data
        """
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_original = self.scaler.inverse_transform(data_reshaped)
        return data_original.reshape(original_shape)
