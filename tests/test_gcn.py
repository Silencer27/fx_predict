"""
Unit tests for GCN layer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tsgcn.models.gcn import GCNLayer


def test_gcn_layer_initialization():
    """Test GCN layer initialization"""
    in_features = 10
    out_features = 20
    
    gcn = GCNLayer(in_features, out_features)
    
    assert gcn.in_features == in_features
    assert gcn.out_features == out_features
    assert gcn.weight.shape == (in_features, out_features)
    assert gcn.bias is not None
    assert gcn.bias.shape == (out_features,)
    
    print("✓ GCN layer initialization test passed")


def test_gcn_layer_forward():
    """Test GCN layer forward pass"""
    batch_size = 4
    num_nodes = 5
    in_features = 10
    out_features = 20
    
    gcn = GCNLayer(in_features, out_features)
    
    # Create sample input
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.eye(num_nodes)  # Identity adjacency matrix
    
    # Forward pass
    output = gcn(x, adj)
    
    assert output.shape == (batch_size, num_nodes, out_features)
    assert not torch.isnan(output).any()
    
    print("✓ GCN layer forward pass test passed")


def test_gcn_layer_no_bias():
    """Test GCN layer without bias"""
    in_features = 10
    out_features = 20
    
    gcn = GCNLayer(in_features, out_features, bias=False)
    
    assert gcn.bias is None
    
    # Test forward pass
    batch_size = 4
    num_nodes = 5
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.eye(num_nodes)
    
    output = gcn(x, adj)
    assert output.shape == (batch_size, num_nodes, out_features)
    
    print("✓ GCN layer no bias test passed")


if __name__ == '__main__':
    test_gcn_layer_initialization()
    test_gcn_layer_forward()
    test_gcn_layer_no_bias()
    print("\nAll GCN layer tests passed! ✓")
