import torch

class Config:
    # Data Parameters
    NUM_NODES = 4        # CN, US, UK, JP
    NUM_FEATURES = 6     # CPI, PolicyRate, RealGDP, Equity_Ret, BondYield_10Y, ExRate_LogRet
    PREDICT_STEPS = 1    # Predicting t+1
    
    # Model Hyperparameters
    HIDDEN_DIM = 32      # Hidden dimension size
    DROPOUT = 0.2
    KERNEL_SIZE = 3      # Convolution kernel size
    
    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 100
    TRAIN_SPLIT = 0.8
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    MODEL_SAVE_PATH = "stgcn_base_model.pth"