"""
Unit tests for TS-GCN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tsgcn import TSGCN


def test_tsgcn_initialization():
    """Test TS-GCN model initialization"""
    num_nodes = 5
    input_dim = 1
    gcn_hidden_dim = 32
    gru_hidden_dim = 32
    output_dim = 1
    
    model = TSGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        gcn_hidden_dim=gcn_hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
        output_dim=output_dim
    )
    
    assert model.num_nodes == num_nodes
    assert model.input_dim == input_dim
    assert model.gcn_hidden_dim == gcn_hidden_dim
    assert model.gru_hidden_dim == gru_hidden_dim
    assert model.output_dim == output_dim
    
    print("✓ TS-GCN initialization test passed")


def test_tsgcn_forward():
    """Test TS-GCN forward pass"""
    batch_size = 4
    seq_len = 10
    num_nodes = 5
    input_dim = 1
    output_dim = 1
    
    model = TSGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        gcn_hidden_dim=32,
        gru_hidden_dim=32,
        output_dim=output_dim
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.eye(num_nodes)
    
    # Forward pass
    output = model(x, adj)
    
    assert output.shape == (batch_size, num_nodes, output_dim)
    assert not torch.isnan(output).any()
    
    print("✓ TS-GCN forward pass test passed")


def test_tsgcn_embeddings():
    """Test TS-GCN embeddings extraction"""
    batch_size = 4
    seq_len = 10
    num_nodes = 5
    input_dim = 1
    gru_hidden_dim = 32
    
    model = TSGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        gcn_hidden_dim=32,
        gru_hidden_dim=gru_hidden_dim,
        output_dim=1
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.eye(num_nodes)
    
    # Get embeddings
    embeddings = model.get_embeddings(x, adj)
    
    assert embeddings.shape == (batch_size, gru_hidden_dim)
    assert not torch.isnan(embeddings).any()
    
    print("✓ TS-GCN embeddings test passed")


def test_tsgcn_gradient_flow():
    """Test gradient flow through TS-GCN"""
    batch_size = 4
    seq_len = 10
    num_nodes = 5
    input_dim = 1
    
    model = TSGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        gcn_hidden_dim=32,
        gru_hidden_dim=32,
        output_dim=1
    )
    
    # Create sample input and target
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.eye(num_nodes)
    target = torch.randn(batch_size, num_nodes, 1)
    
    # Forward pass
    output = model(x, adj)
    
    # Compute loss and backward
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    print("✓ TS-GCN gradient flow test passed")


if __name__ == '__main__':
    test_tsgcn_initialization()
    test_tsgcn_forward()
    test_tsgcn_embeddings()
    test_tsgcn_gradient_flow()
    print("\nAll TS-GCN model tests passed! ✓")
