import torch
import torch.nn as nn
from model.temporal import CausalConv1d
from model.spatial import GATLayer
from model.fusion import FusionLayer

class STGCN_Base(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, dropout=0.2):
        super(STGCN_Base, self).__init__()
        
        # 1. Temporal Module (Section 4.1)
        # Input: (Batch*Nodes, Features, Time)
        self.temporal_layer = CausalConv1d(
            in_channels=num_features, 
            out_channels=hidden_dim, 
            dropout=dropout
        )
        
        # 2. Spatial Module (Section 4.2)
        # Input: (Batch, Time, Nodes, Hidden)
        self.spatial_layer = GATLayer(
            in_dim=hidden_dim, 
            out_dim=hidden_dim, 
            dropout=dropout
        )
        
        # 3. Fusion Layer (Section 4.3)
        self.fusion_layer = FusionLayer(hidden_dim)
        
        # 4. Prediction Layer (Section 4.4)
        # "A simple MLP outputs future exchange rate returns"
        # We take the LAST time step for prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Output: Scalar (Exchange Rate Return)
        )

    def forward(self, x, adj):
        """
        x: (Batch, Time, Nodes, Features)
        adj: (Nodes, Nodes)
        """
        B, T, N, F = x.size()
        
        # --- Temporal Pass ---
        # Reshape to (Batch * Nodes, Features, Time) for Conv1d
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B*N, F, T)
        h_temporal = self.temporal_layer(x_reshaped) 
        # Output is (B*N, Hidden, T)
        
        # Reshape back to (Batch, Time, Nodes, Hidden) for Spatial
        h_temporal = h_temporal.view(B, N, -1, T).permute(0, 3, 1, 2)
        
        # --- Spatial Pass ---
        h_spatial = self.spatial_layer(h_temporal, adj)
        
        # --- Fusion Pass ---
        z = self.fusion_layer(h_temporal, h_spatial)
        # z shape: (Batch, Time, Nodes, Hidden)
        
        # --- Prediction ---
        # We only care about predicting the Next Step based on the Last Step in the sequence
        # Take the last time step T
        z_last = z[:, -1, :, :] # (Batch, Nodes, Hidden)
        
        prediction = self.regressor(z_last) # (Batch, Nodes, 1)
        
        return prediction.squeeze(-1) # (Batch, Nodes)