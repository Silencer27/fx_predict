import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# Import modules
from config import Config
from model.model_base import STGCN_Base
from data.data_fetch import generate_mock_data, process_data, build_adjacency_matrix

def prepare_sliding_window_data(X_tensor, window_size=12, horizon=1):
    """
    Turns Time-Series into Supervised Learning Samples (Sliding Window).
    X_tensor: (Total_Time, Nodes, Features)
    
    Returns:
    inputs: (Samples, Window_Size, Nodes, Features)
    targets: (Samples, Nodes) -> Predicting 'ExRate_LogRet' for next step
    """
    total_time, num_nodes, num_features = X_tensor.shape
    inputs = []
    targets = []
    
    # Target Index: ExRate_LogRet is the last feature in data_fetch.py
    target_idx = -1 
    
    for t in range(total_time - window_size - horizon + 1):
        # Input: t to t+window
        x_window = X_tensor[t : t+window_size, :, :]
        # Target: t+window+horizon-1 (The specific target feature)
        y_target = X_tensor[t+window_size+horizon-1, :, target_idx]
        
        inputs.append(x_window)
        targets.append(y_target)
        
    return torch.stack(inputs), torch.stack(targets)

def main():
    cfg = Config()
    
    # 1. Data Preparation
    print(">>> Loading Data...")
    raw_data = generate_mock_data()
    X_np, _ = process_data(raw_data) # (Total_Time, Nodes, Features)
    A_np = build_adjacency_matrix()  # (Nodes, Nodes)
    
    # Convert to Tensor
    X_full = torch.tensor(X_np, dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32).to(cfg.DEVICE)
    
    # Create Sliding Windows (Lookback 12 months -> Predict next month)
    WINDOW_SIZE = 12 
    inputs, targets = prepare_sliding_window_data(X_full, window_size=WINDOW_SIZE)
    
    # Train/Test Split
    train_size = int(len(inputs) * cfg.TRAIN_SPLIT)
    train_x, test_x = inputs[:train_size], inputs[train_size:]
    train_y, test_y = targets[:train_size], targets[train_size:]
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"Data Shapes: Inputs {inputs.shape}, Targets {targets.shape}")

    # 2. Model Initialization
    model = STGCN_Base(
        num_nodes=cfg.NUM_NODES,
        num_features=cfg.NUM_FEATURES,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss() # Regression task

    # 3. Training Loop
    print(">>> Starting Training...")
    best_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(cfg.DEVICE), batch_y.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(batch_x, A) # A is static
            
            # Loss
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {avg_loss:.6f}")
            
        # Basic Save Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)

    # 4. Evaluation
    print(">>> Evaluating...")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(cfg.DEVICE), batch_y.to(cfg.DEVICE)
            preds = model(batch_x, A)
            loss = criterion(preds, batch_y)
            test_loss += loss.item()
            
    print(f"Final Test MSE: {test_loss / len(test_loader):.6f}")
    print("Done!")

if __name__ == "__main__":
    main()