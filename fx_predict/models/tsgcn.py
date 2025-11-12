"""
Temporal Spatial Graph Convolutional Network (TSGCN) for FX prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fx_predict.models.gcn import GraphConvolution
from fx_predict.models.tcn import TemporalConvNet


class TSGCN(nn.Module):
    """
    Temporal Spatial Graph Convolutional Network
    
    Combines spatial features from currency relationships (GCN) with 
    temporal patterns (TCN) for FX rate prediction.
    
    Args:
        num_nodes: Number of currency nodes in the graph
        num_features: Number of input features per node
        temporal_channels: List of channels for temporal conv layers
        spatial_channels: List of channels for spatial conv layers
        seq_len: Length of input sequence
        pred_len: Length of prediction horizon
        kernel_size: Kernel size for temporal convolutions
        dropout: Dropout rate
    """
    
    def __init__(self, num_nodes, num_features, temporal_channels=[64, 64], 
                 spatial_channels=[64, 32], seq_len=10, pred_len=1,
                 kernel_size=3, dropout=0.2):
        super(TSGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Spatial (Graph) convolution layers
        self.gc1 = GraphConvolution(num_features, spatial_channels[0])
        self.gc2 = GraphConvolution(spatial_channels[0], spatial_channels[1])
        
        # Temporal convolution network
        self.tcn = TemporalConvNet(
            num_inputs=spatial_channels[1] * num_nodes,
            num_channels=temporal_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output layer
        self.fc = nn.Linear(temporal_channels[-1], num_nodes * pred_len)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, num_nodes, num_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            out: Prediction tensor (batch_size, pred_len, num_nodes)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Apply spatial convolutions at each time step
        gcn_outputs = []
        for t in range(seq_len):
            # Extract features at time t
            xt = x[:, t, :, :]  # (batch_size, num_nodes, num_features)
            
            # Apply GCN layers
            h = F.relu(self.gc1(xt, adj))
            h = self.dropout(h)
            h = F.relu(self.gc2(h, adj))
            h = self.dropout(h)
            
            gcn_outputs.append(h)
        
        # Stack temporal features
        # (batch_size, seq_len, num_nodes, spatial_channels[-1])
        gcn_out = torch.stack(gcn_outputs, dim=1)
        
        # Reshape for TCN: (batch_size, num_nodes * spatial_channels[-1], seq_len)
        tcn_input = gcn_out.permute(0, 2, 3, 1).contiguous()
        tcn_input = tcn_input.view(batch_size, -1, seq_len)
        
        # Apply temporal convolutions
        tcn_out = self.tcn(tcn_input)  # (batch_size, temporal_channels[-1], seq_len)
        
        # Take the last time step
        tcn_out = tcn_out[:, :, -1]  # (batch_size, temporal_channels[-1])
        
        # Generate predictions
        out = self.fc(tcn_out)  # (batch_size, num_nodes * pred_len)
        out = out.view(batch_size, self.pred_len, self.num_nodes)
        
        return out
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
