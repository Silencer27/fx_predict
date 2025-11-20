# TS-GCN Architecture for FX Prediction

## Overview

This document describes the architecture of the Temporal Spatial Graph Convolutional Network (TS-GCN) implemented for foreign exchange rate prediction.

## Model Architecture

### High-Level Flow

```
Input Time Series (FX rates)
         ↓
    [Preprocessing]
    - Normalization (StandardScaler)
    - Sequence creation (sliding window)
    - Adjacency matrix computation
         ↓
    [GCN Layers] × N
    - Spatial feature extraction
    - Captures relationships between FX pairs
         ↓
    [Batch Normalization & Dropout]
         ↓
    [GRU Layers]
    - Temporal feature extraction
    - Models time series patterns
         ↓
    [Fully Connected Layers]
    - Feature transformation
    - Final prediction
         ↓
    Output (Predicted FX rates)
```

## Components

### 1. Graph Convolutional Network (GCN) Layer

**File:** `tsgcn/models/gcn.py`

**Purpose:** Captures spatial relationships between different currency pairs

**Operation:**
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

Where:
- `H^(l)`: Node features at layer l
- `A`: Adjacency matrix (correlation-based)
- `D`: Degree matrix
- `W^(l)`: Learnable weight matrix
- `σ`: Activation function (ReLU)

**Key Features:**
- Xavier initialization for stable gradients
- Configurable input/output dimensions
- Optional bias term

### 2. TS-GCN Model

**File:** `tsgcn/models/tsgcn.py`

**Purpose:** Complete spatiotemporal model combining GCN and GRU

**Architecture Details:**

1. **Spatial Processing (per time step):**
   - Input: [batch_size, num_nodes, input_dim]
   - GCN Layer 1: [input_dim → gcn_hidden_dim]
   - GCN Layer 2: [gcn_hidden_dim → gcn_hidden_dim]
   - Batch Normalization & Dropout
   - Output: [batch_size, num_nodes, gcn_hidden_dim]

2. **Temporal Processing:**
   - Flatten spatial features: [batch_size, seq_len, num_nodes * gcn_hidden_dim]
   - 2-layer GRU: [num_nodes * gcn_hidden_dim → gru_hidden_dim]
   - Use last time step output

3. **Prediction:**
   - FC Layer 1: [gru_hidden_dim → gru_hidden_dim // 2]
   - Dropout
   - FC Layer 2: [gru_hidden_dim // 2 → num_nodes * output_dim]
   - Reshape to [batch_size, num_nodes, output_dim]

**Parameters (default):**
- GCN Hidden Dim: 64
- GRU Hidden Dim: 64
- Number of GCN Layers: 2
- Number of GRU Layers: 2
- Dropout: 0.3

### 3. Data Loader

**File:** `tsgcn/utils/data_loader.py`

**Functions:**

1. **Data Preprocessing:**
   - Normalization using StandardScaler
   - Train/Val/Test split (70/15/15)
   - Sequence creation with sliding window

2. **Adjacency Matrix Construction:**
   - Compute correlation between FX pairs
   - Threshold-based edge creation (default: 0.5)
   - Symmetric normalization: D^(-1/2) A D^(-1/2)
   - Self-loops added

3. **Dataset Creation:**
   - PyTorch Dataset wrapper
   - Automatic batching
   - Returns (sequences, adjacency, targets)

### 4. Trainer

**File:** `tsgcn/utils/trainer.py`

**Features:**
- Training loop with validation
- Early stopping based on validation loss
- Learning rate scheduling
- Gradient clipping (max_norm=5.0)
- Model checkpointing
- Comprehensive metrics calculation

### 5. Evaluation Metrics

**File:** `tsgcn/utils/metrics.py`

**Metrics:**
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Relative error
- **R²** (R-squared): Explained variance

## Data Flow Example

### Input Format
```python
# Time series data
data.shape = (1000, 10, 1)
# 1000 time steps, 10 FX pairs, 1 feature (e.g., closing price)
```

### Training Step
```python
# 1. Create sequences
x = data[t:t+seq_len]  # [seq_len, num_nodes, features]
y = data[t+seq_len]    # [num_nodes, features]

# 2. Forward pass
predictions = model(x, adj_matrix)  # [num_nodes, output_dim]

# 3. Compute loss
loss = MSELoss(predictions, y)

# 4. Backward pass
loss.backward()
optimizer.step()
```

### Prediction
```python
# Given recent 10 time steps
x = recent_data[-10:]  # [10, 10, 1]

# Predict next time step
prediction = model(x, adj_matrix)  # [10, 1]
# Predictions for all 10 FX pairs
```

## Usage Example

### Training
```python
from tsgcn import TSGCN, FXDataLoader, Trainer

# Load data
loader = FXDataLoader(seq_len=10, pred_len=1)
train_data, val_data, test_data, adj = loader.load_data(your_data)

# Create model
model = TSGCN(num_nodes=10, input_dim=1)

# Train
trainer = Trainer(model, optimizer, criterion)
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Inference
```python
# Load trained model
model.load_state_dict(checkpoint['model_state_dict'])

# Make prediction
prediction = model(recent_data, adj_matrix)
```

## Hyperparameters

Key hyperparameters (configurable in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| NUM_NODES | 10 | Number of FX pairs |
| INPUT_DIM | 1 | Features per node |
| GCN_HIDDEN_DIM | 64 | GCN layer size |
| GRU_HIDDEN_DIM | 64 | GRU layer size |
| NUM_GCN_LAYERS | 2 | Number of GCN layers |
| DROPOUT | 0.3 | Dropout rate |
| SEQ_LEN | 10 | Input sequence length |
| BATCH_SIZE | 32 | Training batch size |
| LEARNING_RATE | 0.001 | Adam learning rate |

## Model Complexity

For default configuration:
- **Total Parameters:** ~167,000
- **GCN Parameters:** ~4,000
- **GRU Parameters:** ~160,000
- **FC Parameters:** ~3,000

## Testing

All components are tested:
- `tests/test_gcn.py`: GCN layer tests
- `tests/test_tsgcn.py`: Full model tests
- `tests/test_data_loader.py`: Data processing tests

Run tests:
```bash
python tests/test_gcn.py
python tests/test_tsgcn.py
python tests/test_data_loader.py
```

## Extensions

The model can be extended for:
1. **Multiple features:** Change INPUT_DIM (e.g., OHLCV data)
2. **Different prediction horizons:** Adjust PRED_LEN
3. **More complex graphs:** Custom adjacency matrices
4. **Attention mechanisms:** Add attention layers
5. **Multi-task learning:** Predict multiple targets
