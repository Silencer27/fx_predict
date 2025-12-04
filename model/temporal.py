import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """
    Section 4.1: Temporal Module
    A lightweight temporal extractor: H_t = ReLU(Conv1D(X))
    
    We use causal padding to ensure no information leakage from the future.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(CausalConv1d, self).__init__()
        
        # Padding logic: to keep output length same as input length in causal mode,
        # we pad (kernel_size - 1) zeros to the left.
        self.pad_size = kernel_size - 1
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0 # We handle padding manually
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Input x shape: (Batch * Nodes, Features, Time_Steps)
        Output shape:  (Batch * Nodes, Out_Channels, Time_Steps)
        """
        # 1. Causal Padding (Left side only)
        # x is (B, C, T), we pad last dim: (left, right)
        x_padded = torch.nn.functional.pad(x, (self.pad_size, 0))
        
        # 2. Convolution
        out = self.conv1d(x_padded)
        
        # 3. Activation & Dropout
        return self.dropout(self.relu(out))