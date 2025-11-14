"""
Evaluation script for TSGCN model
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os

from fx_predict.models.tsgcn import TSGCN
from fx_predict.data.dataset import FXDataset
from fx_predict.data.graph_builder import build_adjacency_matrix, compute_correlation_matrix
from fx_predict.utils.metrics import calculate_metrics
from fx_predict.utils.visualization import plot_predictions
from fx_predict.config.config import load_config, get_default_config


def evaluate(model, dataloader, device, adj, dataset):
    """
    Evaluate model on dataset
    
    Args:
        model: TSGCN model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        adj: Adjacency matrix
        dataset: Dataset object for inverse transform
        
    Returns:
        Predictions and true values
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x, adj)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Reshape for metrics calculation
    predictions = predictions.reshape(-1, predictions.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])
    
    # Inverse transform to original scale
    predictions_orig = dataset.inverse_transform(predictions)
    targets_orig = dataset.inverse_transform(targets)
    
    return predictions_orig, targets_orig


def main():
    parser = argparse.ArgumentParser(description='Evaluate TSGCN for FX prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FX data file (numpy array)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found, using default configuration")
        config = get_default_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = np.load(args.data)
    print(f"Loaded data with shape: {data.shape}")
    
    # Create test dataset
    test_dataset = FXDataset(
        data,
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        train=False,
        train_ratio=config['training']['train_ratio']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Build adjacency matrix
    if config['data']['graph_method'] == 'correlation':
        corr_matrix = compute_correlation_matrix(data)
        adj = build_adjacency_matrix(
            config['model']['num_nodes'],
            method='correlation',
            correlation_matrix=corr_matrix,
            threshold=config['data']['correlation_threshold']
        )
    else:
        adj = build_adjacency_matrix(
            config['model']['num_nodes'],
            method=config['data']['graph_method']
        )
    
    adj = adj.to(device)
    
    # Create model
    model = TSGCN(
        num_nodes=config['model']['num_nodes'],
        num_features=config['model']['num_features'],
        temporal_channels=config['model']['temporal_channels'],
        spatial_channels=config['model']['spatial_channels'],
        seq_len=config['model']['seq_len'],
        pred_len=config['model']['pred_len'],
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load trained model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")
    
    # Evaluate
    print("Evaluating model...")
    predictions, targets = evaluate(model, test_loader, device, adj, test_dataset)
    
    # Calculate metrics for each currency
    print("\nEvaluation Results:")
    print("=" * 80)
    
    for i in range(config['model']['num_nodes']):
        metrics = calculate_metrics(targets[:, i], predictions[:, i])
        print(f"\nCurrency {i+1}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")
    
    # Overall metrics
    overall_metrics = calculate_metrics(targets.flatten(), predictions.flatten())
    print(f"\nOverall Metrics:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    
    # Save predictions
    np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(args.output_dir, 'targets.npy'), targets)
    
    # Plot predictions
    currency_names = [f'Currency {i+1}' for i in range(config['model']['num_nodes'])]
    plot_predictions(
        targets,
        predictions,
        currency_names=currency_names,
        save_path=os.path.join(args.output_dir, 'predictions.png')
    )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
