"""
Configuration management
"""

import yaml
import os


def get_default_config():
    """
    Get default configuration
    
    Returns:
        Dictionary with default configuration
    """
    config = {
        'model': {
            'num_nodes': 7,  # Number of currencies
            'num_features': 4,  # OHLC data
            'temporal_channels': [64, 64],
            'spatial_channels': [64, 32],
            'seq_len': 10,
            'pred_len': 1,
            'kernel_size': 3,
            'dropout': 0.2
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,
            'train_ratio': 0.8
        },
        'data': {
            'graph_method': 'fully_connected',  # 'fully_connected', 'correlation', 'identity'
            'correlation_threshold': 0.5
        }
    }
    return config


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with default config
    default_config = get_default_config()
    
    # Update defaults with loaded config
    for key in config:
        if key in default_config and isinstance(default_config[key], dict):
            default_config[key].update(config[key])
        else:
            default_config[key] = config[key]
    
    return default_config


def save_config(config, config_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {config_path}")
