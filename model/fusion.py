import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Section 4.3: Fusion Layer
    Z_t = W1 * H_t + W2 * H_s
    
    A simple learnable weighted sum or concatenation. 
    Here we implement the weighted sum as per the PDF formula.
    """
    def __init__(self, feature_dim):
        super(FusionLayer, self).__init__()
        # Learnable weights for Temporal (W1) and Spatial (W2)
        # We make them diagonal matrices (element-wise scaling) or scalars. 
        # Using a Linear layer is flexible.
        self.W1 = nn.Linear(feature_dim, feature_dim)
        self.W2 = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, h_temporal, h_spatial):
        """
        Inputs: (Batch, Time, Nodes, Features)
        """
        # Apply weights
        term1 = self.W1(h_temporal)
        term2 = self.W2(h_spatial)
        
        # Sum
        z = term1 + term2
        
        # Optional: LayerNorm or Activation
        return self.layer_norm(z)