"""
Graph Convolutional Network Layer for TS-GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    
    Implements the spectral graph convolution operation:
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias (default: True)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features [batch_size, num_nodes, in_features]
            adj (torch.Tensor): Normalized adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            torch.Tensor: Output features [batch_size, num_nodes, out_features]
        """
        # x: [batch_size, num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        
        # Linear transformation: XW
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # Graph convolution: AXW
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'
