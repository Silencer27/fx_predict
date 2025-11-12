"""
Unit tests for data loader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tsgcn import FXDataLoader


def test_data_loader_initialization():
    """Test data loader initialization"""
    seq_len = 10
    pred_len = 1
    
    data_loader = FXDataLoader(seq_len=seq_len, pred_len=pred_len)
    
    assert data_loader.seq_len == seq_len
    assert data_loader.pred_len == pred_len
    
    print("✓ Data loader initialization test passed")


def test_load_data():
    """Test data loading"""
    # Create synthetic data
    num_samples = 100
    num_nodes = 5
    data = np.random.randn(num_samples, num_nodes, 1)
    
    data_loader = FXDataLoader(seq_len=10, pred_len=1)
    train_data, val_data, test_data, adj_matrix = data_loader.load_data(data)
    
    # Check shapes
    assert train_data.shape[1] == num_nodes
    assert val_data.shape[1] == num_nodes
    assert test_data.shape[1] == num_nodes
    assert adj_matrix.shape == (num_nodes, num_nodes)
    
    # Check data split
    assert len(train_data) > len(val_data)
    assert len(train_data) > len(test_data)
    
    print("✓ Data loading test passed")


def test_adjacency_matrix():
    """Test adjacency matrix computation"""
    num_samples = 100
    num_nodes = 5
    data = np.random.randn(num_samples, num_nodes, 1)
    
    data_loader = FXDataLoader()
    adj_matrix = data_loader.compute_adjacency_matrix(data)
    
    # Check shape
    assert adj_matrix.shape == (num_nodes, num_nodes)
    
    # Check symmetry
    assert np.allclose(adj_matrix, adj_matrix.T)
    
    # Check no NaN values
    assert not np.isnan(adj_matrix).any()
    
    print("✓ Adjacency matrix test passed")


def test_create_dataloaders():
    """Test dataloader creation"""
    num_samples = 100
    num_nodes = 5
    data = np.random.randn(num_samples, num_nodes, 1)
    
    data_loader = FXDataLoader(seq_len=10, pred_len=1)
    train_data, val_data, test_data, adj_matrix = data_loader.load_data(data)
    
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data, adj_matrix, batch_size=8
    )
    
    # Check loaders exist
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # Test getting a batch
    x, adj, y = next(iter(train_loader))
    
    assert x.shape[1] == 10  # seq_len
    assert x.shape[2] == num_nodes
    # adj is returned per batch sample, but should be same for all
    assert adj.shape == (num_nodes, num_nodes) or adj.shape[1:] == (num_nodes, num_nodes)
    assert y.shape[1] == 1  # pred_len
    
    print("✓ Dataloader creation test passed")


if __name__ == '__main__':
    test_data_loader_initialization()
    test_load_data()
    test_adjacency_matrix()
    test_create_dataloaders()
    print("\nAll data loader tests passed! ✓")
