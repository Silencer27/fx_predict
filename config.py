"""
Configuration file for TS-GCN model
"""


class Config:
    """
    Configuration parameters for TS-GCN FX prediction
    """
    
    # Model parameters
    NUM_NODES = 10  # Number of FX pairs to predict
    INPUT_DIM = 1  # Number of features per node (e.g., closing price)
    GCN_HIDDEN_DIM = 64  # Hidden dimension for GCN layers
    GRU_HIDDEN_DIM = 64  # Hidden dimension for GRU layers
    OUTPUT_DIM = 1  # Number of output features
    NUM_GCN_LAYERS = 2  # Number of GCN layers
    DROPOUT = 0.3  # Dropout rate
    
    # Data parameters
    SEQ_LEN = 10  # Length of input sequences (lookback window)
    PRED_LEN = 1  # Length of prediction horizon
    CORRELATION_THRESHOLD = 0.5  # Threshold for adjacency matrix construction
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    
    # Device
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Paths
    DATA_PATH = None  # Path to data file (if using CSV)
    MODEL_SAVE_PATH = 'checkpoints/'
    
    @classmethod
    def display(cls):
        """Display configuration"""
        print("=" * 50)
        print("TS-GCN Configuration")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)
