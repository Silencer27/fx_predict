"""
Dataset class for FX time series data
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class FXDataset(Dataset):
    """
    Dataset for FX time series data
    
    Args:
        data: FX rate data, shape (timesteps, num_currencies, num_features)
        seq_len: Length of input sequence
        pred_len: Length of prediction horizon
        train: If True, create training set; otherwise validation/test set
        train_ratio: Ratio of data to use for training
    """
    
    def __init__(self, data, seq_len=10, pred_len=1, train=True, train_ratio=0.8):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Normalize data
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
        
        normalized_data = (data - self.mean) / self.std
        
        # Split into train/test
        split_idx = int(len(normalized_data) * train_ratio)
        if train:
            self.data = normalized_data[:split_idx]
        else:
            self.data = normalized_data[split_idx:]
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.data) - seq_len - pred_len + 1):
            seq = self.data[i:i + seq_len]
            target = self.data[i + seq_len:i + seq_len + pred_len, :, 0]  # Predict first feature
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            seq: Input sequence (seq_len, num_currencies, num_features)
            target: Target values (pred_len, num_currencies)
        """
        seq = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        return seq, target
    
    def inverse_transform(self, normalized_data):
        """
        Convert normalized data back to original scale
        
        Args:
            normalized_data: Normalized data to transform
            
        Returns:
            Original scale data
        """
        return normalized_data * self.std[:, :, 0] + self.mean[:, :, 0]
