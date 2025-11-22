"""
Graph construction utilities for FX currency relationships
"""

import numpy as np
import torch


def build_adjacency_matrix(num_nodes, method='fully_connected', correlation_matrix=None, threshold=0.5):
    """
    Build adjacency matrix for currency graph
    
    Args:
        num_nodes: Number of currency nodes
        method: Method to build adjacency matrix
            - 'fully_connected': All nodes connected
            - 'correlation': Based on correlation matrix
            - 'identity': Self-loops only
        correlation_matrix: Correlation matrix for 'correlation' method
        threshold: Threshold for correlation-based connections
        
    Returns:
        adj: Adjacency matrix as torch tensor (num_nodes, num_nodes)
    """
    
    if method == 'fully_connected':
        # Fully connected graph with self-loops
        adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
        
    elif method == 'correlation':
        # Build graph based on correlation
        if correlation_matrix is None:
            raise ValueError("correlation_matrix must be provided for 'correlation' method")
        
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Add edges where correlation exceeds threshold
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    adj[i, j] = 1.0  # Self-loop
                elif abs(correlation_matrix[i, j]) > threshold:
                    adj[i, j] = abs(correlation_matrix[i, j])
                    
    elif method == 'identity':
        # Only self-loops (independent nodes)
        adj = np.eye(num_nodes, dtype=np.float32)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize adjacency matrix (symmetric normalization)
    adj = normalize_adjacency_matrix(adj)
    
    return torch.FloatTensor(adj)


def normalize_adjacency_matrix(adj):
    """
    Symmetrically normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: Adjacency matrix (num_nodes, num_nodes)
        
    Returns:
        Normalized adjacency matrix
    """
    adj = adj + np.eye(adj.shape[0])  # Add self-loops if not present
    
    # Degree matrix
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Symmetric normalization
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return adj_normalized


def compute_correlation_matrix(data):
    """
    Compute correlation matrix from time series data
    
    Args:
        data: Time series data (timesteps, num_currencies, num_features)
        
    Returns:
        Correlation matrix (num_currencies, num_currencies)
    """
    # Use first feature for correlation
    data_2d = data[:, :, 0]  # (timesteps, num_currencies)
    
    # Compute correlation
    correlation = np.corrcoef(data_2d.T)
    
    return correlation
