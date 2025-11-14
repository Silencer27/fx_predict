# FX Predict

A Temporal Spatial Graph Convolutional Network (TSGCN) for Foreign Exchange (FX) rate prediction.

## Overview

This project implements a deep learning model that combines:
- **Graph Convolutional Networks (GCN)**: Capture spatial relationships between different currency pairs
- **Temporal Convolutional Networks (TCN)**: Model temporal dependencies in FX rate time series

The TSGCN architecture is designed to predict future FX rates by learning both the correlations between different currencies and the temporal patterns in their historical movements.

## Features

- **Flexible Architecture**: Configurable model parameters for different prediction tasks
- **Graph-based Learning**: Models currency relationships through graph structures
- **Temporal Modeling**: Captures time series patterns with dilated causal convolutions
- **Easy Configuration**: YAML-based configuration for experiments
- **Comprehensive Evaluation**: Multiple metrics for model assessment
- **Visualization Tools**: Plot predictions and training history

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- PyYAML >= 6.0

### Install from source

```bash
git clone https://github.com/Silencer27/fx_predict.git
cd fx_predict
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Run the Example

```bash
python example.py
```

This will demonstrate the basic usage with synthetic data.

### 2. Prepare Your Data

Your FX data should be in the shape: `(timesteps, num_currencies, num_features)`

Example with OHLC data for 7 currencies:
```python
import numpy as np

# Generate or load your data
# Shape: (1000 timesteps, 7 currencies, 4 features)
fx_data = np.random.randn(1000, 7, 4)

# Save data
np.save('fx_data.npy', fx_data)
```

### 3. Configure the Model

Edit `config.yaml` to customize model parameters:

```yaml
model:
  num_nodes: 7          # Number of currencies
  num_features: 4       # Number of features (e.g., OHLC)
  seq_len: 10          # Input sequence length
  pred_len: 1          # Prediction horizon
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

### 4. Train the Model

```bash
python train.py --data fx_data.npy --config config.yaml
```

### 5. Evaluate the Model

```bash
python evaluate.py --data fx_data.npy --model checkpoints/best_model.pth
```

## Architecture

### TSGCN Model

The model consists of three main components:

1. **Spatial Module (GCN)**
   - Captures relationships between currency pairs
   - Two graph convolutional layers
   - Processes each time step independently

2. **Temporal Module (TCN)**
   - Models temporal dependencies
   - Dilated causal convolutions
   - Maintains causality for time series prediction

3. **Output Layer**
   - Fully connected layer
   - Generates predictions for all currencies

### Model Flow

```
Input (batch, seq_len, num_nodes, features)
    ↓
GCN Layers (for each time step)
    ↓
Reshape for temporal processing
    ↓
TCN Layers (dilated causal convolutions)
    ↓
Fully Connected Layer
    ↓
Output (batch, pred_len, num_nodes)
```

## Configuration Options

### Model Parameters

- `num_nodes`: Number of currency nodes
- `num_features`: Number of input features per node
- `temporal_channels`: List of channel sizes for TCN layers
- `spatial_channels`: List of channel sizes for GCN layers
- `seq_len`: Input sequence length
- `pred_len`: Prediction horizon
- `kernel_size`: Kernel size for temporal convolutions
- `dropout`: Dropout rate

### Training Parameters

- `batch_size`: Batch size for training
- `epochs`: Maximum number of epochs
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: L2 regularization parameter
- `early_stopping_patience`: Patience for early stopping
- `train_ratio`: Ratio of data for training

### Data Parameters

- `graph_method`: Method to build adjacency matrix
  - `'fully_connected'`: All nodes connected
  - `'correlation'`: Based on correlation matrix
  - `'identity'`: Only self-loops
- `correlation_threshold`: Threshold for correlation-based edges

## API Usage

### Basic Usage

```python
import torch
import numpy as np
from fx_predict import TSGCN
from fx_predict.data import FXDataset, build_adjacency_matrix
from torch.utils.data import DataLoader

# Load data
data = np.load('fx_data.npy')

# Create dataset
dataset = FXDataset(data, seq_len=10, pred_len=1)
dataloader = DataLoader(dataset, batch_size=32)

# Build adjacency matrix
adj = build_adjacency_matrix(num_nodes=7, method='fully_connected')

# Create model
model = TSGCN(
    num_nodes=7,
    num_features=4,
    temporal_channels=[64, 64],
    spatial_channels=[64, 32],
    seq_len=10,
    pred_len=1
)

# Forward pass
x, y = next(iter(dataloader))
output = model(x, adj)
```

## Evaluation Metrics

The model is evaluated using:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: R-squared score

## Project Structure

```
fx_predict/
├── fx_predict/
│   ├── models/
│   │   ├── gcn.py          # Graph Convolutional Network layer
│   │   ├── tcn.py          # Temporal Convolutional Network
│   │   └── tsgcn.py        # Main TSGCN model
│   ├── data/
│   │   ├── dataset.py      # Dataset class
│   │   └── graph_builder.py # Graph construction utilities
│   ├── utils/
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── visualization.py # Plotting functions
│   └── config/
│       └── config.py       # Configuration management
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── example.py             # Example usage
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fx_predict,
  title = {FX Predict: A Temporal Spatial Graph Convolutional Network for FX Prediction},
  author = {Silencer27},
  year = {2024},
  url = {https://github.com/Silencer27/fx_predict}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Graph Convolutional Networks: [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907)
- Temporal Convolutional Networks: [Bai et al. (2018)](https://arxiv.org/abs/1803.01271)
