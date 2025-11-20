# FX Predict - TS-GCN Model

A Temporal Spatial Graph Convolutional Network (TS-GCN) implementation for Foreign Exchange (FX) rate prediction.

## Overview

TS-GCN combines Graph Convolutional Networks (GCN) for capturing spatial dependencies between different currency pairs with Gated Recurrent Units (GRU) for modeling temporal patterns. This architecture is particularly effective for multivariate time series forecasting where relationships between different series are important.

### Key Features

- **Spatial Modeling**: GCN layers capture relationships between different FX pairs
- **Temporal Modeling**: GRU layers capture temporal patterns in time series
- **Flexible Architecture**: Configurable number of layers, hidden dimensions, and other hyperparameters
- **Complete Training Pipeline**: Includes data loading, preprocessing, training, and evaluation utilities
- **Extensible**: Easy to adapt for other spatiotemporal prediction tasks

## Architecture

```
Input (Time Series) 
    ↓
GCN Layers (Spatial Feature Extraction)
    ↓
GRU Layers (Temporal Feature Extraction)
    ↓
Fully Connected Layers
    ↓
Output (Predictions)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Silencer27/fx_predict.git
cd fx_predict
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0

## Quick Start

### Training Example

```python
from tsgcn import TSGCN, FXDataLoader, Trainer
import torch

# Load and preprocess data
data_loader = FXDataLoader(seq_len=10, pred_len=1)
train_data, val_data, test_data, adj_matrix = data_loader.load_data(your_data)

# Create dataloaders
train_loader, val_loader, test_loader = data_loader.create_dataloaders(
    train_data, val_data, test_data, adj_matrix, batch_size=32
)

# Initialize model
model = TSGCN(
    num_nodes=10,
    input_dim=1,
    gcn_hidden_dim=64,
    gru_hidden_dim=64,
    output_dim=1
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
trainer = Trainer(model, optimizer, criterion, device='cpu')

# Train
history = trainer.train(train_loader, val_loader, num_epochs=100)

# Evaluate
test_loss, metrics, predictions, targets = trainer.test(test_loader)
```

### Running Examples

Train on synthetic data:
```bash
python examples/train_example.py
```

Run inference:
```bash
python examples/inference_example.py
```

## Project Structure

```
fx_predict/
├── tsgcn/                      # Main package
│   ├── models/                 # Model implementations
│   │   ├── gcn.py             # GCN layer
│   │   └── tsgcn.py           # TS-GCN model
│   └── utils/                  # Utilities
│       ├── data_loader.py     # Data loading and preprocessing
│       ├── trainer.py         # Training utilities
│       └── metrics.py         # Evaluation metrics
├── examples/                   # Example scripts
│   ├── train_example.py       # Training example
│   └── inference_example.py   # Inference example
├── tests/                      # Unit tests
│   ├── test_gcn.py
│   ├── test_tsgcn.py
│   └── test_data_loader.py
├── config.py                   # Configuration file
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Configuration

Model and training parameters can be configured in `config.py`:

```python
class Config:
    # Model parameters
    NUM_NODES = 10              # Number of FX pairs
    INPUT_DIM = 1               # Features per node
    GCN_HIDDEN_DIM = 64         # GCN hidden dimension
    GRU_HIDDEN_DIM = 64         # GRU hidden dimension
    OUTPUT_DIM = 1              # Output features
    NUM_GCN_LAYERS = 2          # Number of GCN layers
    DROPOUT = 0.3               # Dropout rate
    
    # Data parameters
    SEQ_LEN = 10                # Input sequence length
    PRED_LEN = 1                # Prediction horizon
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
```

## Model Components

### GCN Layer

Graph Convolutional Network layer implements spectral graph convolution:

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

Where:
- H^(l): Features at layer l
- A: Adjacency matrix
- D: Degree matrix
- W^(l): Learnable weight matrix
- σ: Activation function

### TS-GCN Model

The complete TS-GCN model architecture:

1. **Input**: Time series data [batch_size, seq_len, num_nodes, input_dim]
2. **GCN Layers**: Extract spatial features at each time step
3. **GRU Layers**: Model temporal dependencies
4. **Output Layers**: Generate predictions

## Data Format

The model expects data in the following format:

- **Input**: `[num_samples, num_nodes, num_features]`
  - `num_samples`: Number of time steps
  - `num_nodes`: Number of FX pairs (or other entities)
  - `num_features`: Number of features per node (e.g., close price, volume)

- **Adjacency Matrix**: `[num_nodes, num_nodes]`
  - Represents relationships between nodes
  - Can be computed from correlations or provided manually

## Testing

Run unit tests:

```bash
python tests/test_gcn.py
python tests/test_tsgcn.py
python tests/test_data_loader.py
```

Or run all tests:
```bash
python -m pytest tests/
```

## Evaluation Metrics

The model provides several evaluation metrics:

- **MAE** (Mean Absolute Error): Average absolute difference
- **RMSE** (Root Mean Square Error): Square root of average squared difference
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **R²** (R-squared): Proportion of variance explained

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fx_predict_tsgcn,
  title = {TS-GCN for FX Prediction},
  author = {Silencer27},
  year = {2025},
  url = {https://github.com/Silencer27/fx_predict}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation is inspired by research in spatiotemporal graph neural networks for time series forecasting.
