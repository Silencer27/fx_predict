import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    Section 4.2: Spatial Module
    H(S) = GAT(H_t, A_trade)
    
    Uses a fixed adjacency matrix to mask attention, focusing only on trading partners.
    """
    def __init__(self, in_dim, out_dim, alpha=0.2, dropout=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        
        # Learnable linear transformation for node features
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention mechanism parameters
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        Input h: (Batch, Time, Nodes, Features)
        Input adj: (Nodes, Nodes) - Fixed Trade Graph
        Output:  (Batch, Time, Nodes, Out_Dim)
        """
        B, T, N, F_in = h.size()
        
        # 1. Linear Transformation
        # Flatten Batch and Time to process all snapshots in parallel
        # h_flat: (B*T, N, F_in)
        h_flat = h.reshape(-1, N, F_in) 
        Wh = self.W(h_flat) # (B*T, N, Out_Dim)
        
        # 2. Prepare Attention Mechanism
        # We need to compute pairs (Wh_i, Wh_j) for all i, j
        # Repeat/Expand to create combinations
        # shape: (B*T, N, N, Out_Dim)
        Wh_repeated_in_chunks = Wh.view(-1, N, 1, self.out_dim).repeat(1, 1, N, 1)
        Wh_repeated_alternating = Wh.view(-1, 1, N, self.out_dim).repeat(1, N, 1, 1)
        
        # Concatenate: (B*T, N, N, 2*Out_Dim)
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        
        # 3. Compute Attention Scores
        e = self.leakyrelu(self.a(all_combinations).squeeze(-1)) # (B*T, N, N)
        
        # 4. Mask with Adjacency Matrix
        # adj is (N, N), we treat 0 values as "no connection"
        # We broadcast adj to (B*T, N, N)
        zero_vec = -9e15 * torch.ones_like(e)
        
        # Ensure adj is on correct device
        adj = adj.to(e.device)
        
        # If A_ij > 0, keep score e_ij. If A_ij = 0, set to very small number (softmax -> 0)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 5. Softmax & Dropout
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # 6. Aggregation
        # (B*T, N, N) x (B*T, N, Out_Dim) -> (B*T, N, Out_Dim)
        h_prime = torch.bmm(attention, Wh)
        
        # 7. Reshape back
        return h_prime.view(B, T, N, self.out_dim)