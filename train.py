import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# Import modules
import config
from model.model_base import STGCN_Base
from data.data_fetch import generate_mock_data, process_data, build_adjacency_matrix

def plot_results(predictions, actuals, num_nodes, node_names=["CN", "US", "UK", "JP"]):
    """
    绘制预测结果与实际结果的折线图。
    predictions, actuals: (Samples, Nodes)
    """
    if num_nodes != len(node_names):
        print("Warning: Number of nodes does not match default node names. Using default indices.")
        node_names = [f"Node {i}" for i in range(num_nodes)]
        
    # 创建子图 (2x2 grid for 4 nodes)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten() # 展平 2x2 数组以便迭代
    
    for i in range(num_nodes):
        ax = axes[i]
        
        # 绘制实际对数回报
        ax.plot(actuals[:, i], label=f'{node_names[i]} 实际值', color='blue', linewidth=1)
        
        # 绘制预测对数回报
        ax.plot(predictions[:, i], label=f'{node_names[i]} 预测值', color='red', linestyle='--', linewidth=1)
        
        ax.set_title(f'{node_names[i]} 汇率预测 (测试集)', fontsize=14)
        ax.set_xlabel('时间步 (测试样本)', fontsize=12)
        ax.set_ylabel('对数回报率', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle('ST-GCN-Base 模型汇率预测结果对比', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig('exchange_rate_predictions.png')
    plt.close(fig)
    print("Prediction plot saved to exchange_rate_predictions.png")


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
    cfg = config.Config()
    
# --- 1. Data Preparation with Archiving Logic ---
    print(">>> Loading Data...")
    
    # 定义处理后数据的存档路径
    data_dir = './data/'
    X_SAVE_PATH = os.path.join(data_dir, 'X_tensor.pt')
    A_SAVE_PATH = os.path.join(data_dir, 'A_trade.pt') 
    
    # 检查存档数据
    if os.path.exists(X_SAVE_PATH) and os.path.exists(A_SAVE_PATH):
        print("--- Archived data found. Loading directly. ---")
        X_full = torch.load(X_SAVE_PATH)
        A = torch.load(A_SAVE_PATH).to(cfg.DEVICE)
    else:
        print("--- Archived data not found. Generating and processing data. ---")
        raw_data = generate_mock_data()
        X_np, _ = process_data(raw_data) # (Total_Time, Nodes, Features)
        A_np = build_adjacency_matrix()  # (Nodes, Nodes)
        
        # 转换为 Tensor
        X_full = torch.tensor(X_np, dtype=torch.float32)
        A = torch.tensor(A_np, dtype=torch.float32) # 在保存前保持在 CPU
        
        # 存档数据
        print(f"--- Archiving data to {data_dir} ---")
        os.makedirs(data_dir, exist_ok=True)
        torch.save(X_full, X_SAVE_PATH)
        torch.save(A, A_SAVE_PATH)
        
        A = A.to(cfg.DEVICE) # 保存后移至设备
    
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
    # NOTE: Using temporary weights to ensure model.load_state_dict() doesn't fail
    temp_save_path = "temp_stgcn_base_model.pth" 
    torch.save(model.state_dict(), temp_save_path) # Save initial state
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
            
    # Final weights might be in temp path if no improvement occurred
    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        os.rename(temp_save_path, cfg.MODEL_SAVE_PATH)

    # 4. Evaluation
    print(">>> Evaluating...")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, weights_only=True)) 
    model.eval()
    
    all_preds = [] 
    all_targets = [] 
    test_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(cfg.DEVICE), batch_y.to(cfg.DEVICE)
            preds = model(batch_x, A)
            loss = criterion(preds, batch_y)
            test_loss += loss.item()
            
            # NEW: Store results
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    print(f"Final Test MSE: {test_loss:.6f}")

    # NEW: Concatenate and plot results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    plot_results(all_preds, all_targets, cfg.NUM_NODES) 
    
    print("Done!")

if __name__ == "__main__":
    main()