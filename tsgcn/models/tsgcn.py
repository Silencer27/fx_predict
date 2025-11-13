"""
TS-GCN: Temporal Spatial Graph Convolutional Network
"""

import torch
import torch.nn as nn
from .gcn import GCNLayer


class TSGCN(nn.Module):
    """
    Temporal Spatial Graph Convolutional Network for FX Prediction
    
    Combines Graph Convolutional Networks (GCN) for spatial dependencies
    with Gated Recurrent Units (GRU) for temporal dependencies.
    
    Architecture:
    1. GCN layers capture spatial relationships between different FX pairs
    2. GRU layers capture temporal patterns in time series
    3. Output layer predicts future FX rates
    
    Args:
        num_nodes (int): Number of FX pairs (nodes in the graph)
        input_dim (int): Number of input features per node
        gcn_hidden_dim (int): Hidden dimension for GCN layers
        gru_hidden_dim (int): Hidden dimension for GRU layers
        output_dim (int): Number of output features (prediction steps)
        num_gcn_layers (int): Number of GCN layers (default: 2)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        num_nodes,
        input_dim,
        gcn_hidden_dim=64,
        gru_hidden_dim=64,
        output_dim=1,
        num_gcn_layers=2,
        dropout=0.3
    ):
        super(TSGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        
        # GCN layers for spatial feature extraction
        self.gcn_layers = nn.ModuleList()
        
        # First GCN layer
        self.gcn_layers.append(GCNLayer(input_dim, gcn_hidden_dim))
        
        # Additional GCN layers
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(gcn_hidden_dim, gcn_hidden_dim))
        
        # Batch normalization for GCN outputs
        self.bn_gcn = nn.BatchNorm1d(num_nodes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # GRU for temporal feature extraction
        # Input: [batch_size, seq_len, num_nodes * gcn_hidden_dim]
        self.gru = nn.GRU(
            input_size=num_nodes * gcn_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(gru_hidden_dim, gru_hidden_dim // 2)
        self.fc2 = nn.Linear(gru_hidden_dim // 2, num_nodes * output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, num_nodes, input_dim]
            adj (torch.Tensor): Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            torch.Tensor: Predictions [batch_size, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Process each time step through GCN layers
        gcn_outputs = []
        
        for t in range(seq_len):
            # Get features at time t: [batch_size, num_nodes, input_dim]
            x_t = x[:, t, :, :]
            
            # Apply GCN layers
            h = x_t
            for i, gcn_layer in enumerate(self.gcn_layers):
                h = gcn_layer(h, adj)
                h = self.relu(h)
                if i < len(self.gcn_layers) - 1:
                    # Apply batch norm and dropout between GCN layers
                    h = h.transpose(1, 2)  # [batch_size, gcn_hidden_dim, num_nodes]
                    h = self.bn_gcn(h.reshape(-1, num_nodes)).reshape(batch_size, self.gcn_hidden_dim, num_nodes)
                    h = h.transpose(1, 2)  # [batch_size, num_nodes, gcn_hidden_dim]
                    h = self.dropout(h)
            
            # Flatten spatial features: [batch_size, num_nodes * gcn_hidden_dim]
            h_flat = h.reshape(batch_size, -1)
            gcn_outputs.append(h_flat)
        
        # Stack temporal features: [batch_size, seq_len, num_nodes * gcn_hidden_dim]
        temporal_input = torch.stack(gcn_outputs, dim=1)
        
        # Apply GRU for temporal modeling
        gru_out, _ = self.gru(temporal_input)  # [batch_size, seq_len, gru_hidden_dim]
        
        # Use the last time step output
        last_output = gru_out[:, -1, :]  # [batch_size, gru_hidden_dim]
        
        # Prediction layers
        out = self.relu(self.fc1(last_output))  # [batch_size, gru_hidden_dim // 2]
        out = self.dropout(out)
        out = self.fc2(out)  # [batch_size, num_nodes * output_dim]
        
        # Reshape to [batch_size, num_nodes, output_dim]
        out = out.reshape(batch_size, num_nodes, self.output_dim)
        
        return out
    
    def get_embeddings(self, x, adj):
        """
        Get learned embeddings from the model
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, num_nodes, input_dim]
            adj (torch.Tensor): Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            torch.Tensor: Embeddings [batch_size, gru_hidden_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            h = x_t
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, adj)
                h = self.relu(h)
            h_flat = h.reshape(batch_size, -1)
            gcn_outputs.append(h_flat)
        
        temporal_input = torch.stack(gcn_outputs, dim=1)
        gru_out, _ = self.gru(temporal_input)
        
        return gru_out[:, -1, :]
