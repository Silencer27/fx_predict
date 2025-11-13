"""
Training utilities for TS-GCN model
"""

import torch
import torch.nn as nn
import numpy as np
from .metrics import calculate_metrics


class Trainer:
    """
    Trainer class for TS-GCN model
    
    Args:
        model (nn.Module): TS-GCN model
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (str): Device to train on ('cuda' or 'cpu')
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler (optional)
    """
    
    def __init__(self, model, optimizer, criterion, device='cpu', scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, adj, y) in enumerate(train_loader):
            # Move to device
            x = x.to(self.device)
            adj = adj.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x, adj)
            
            # Compute loss
            # y shape: [batch_size, pred_len, num_nodes, features]
            # predictions shape: [batch_size, num_nodes, output_dim]
            # Take the last prediction step
            y_target = y[:, -1, :, :]  # [batch_size, num_nodes, features]
            
            loss = self.criterion(predictions, y_target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            tuple: (average validation loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, adj, y in val_loader:
                # Move to device
                x = x.to(self.device)
                adj = adj.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                predictions = self.model(x, adj)
                
                # Compute loss
                y_target = y[:, -1, :, :]
                loss = self.criterion(predictions, y_target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(y_target.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            early_stopping_patience (int): Patience for early stopping
        
        Returns:
            dict: Training history
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val MAE: {val_metrics['MAE']:.6f}, Val RMSE: {val_metrics['RMSE']:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def test(self, test_loader):
        """
        Test the model
        
        Args:
            test_loader (DataLoader): Test data loader
        
        Returns:
            tuple: (test loss, metrics dict, predictions, targets)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, adj, y in test_loader:
                # Move to device
                x = x.to(self.device)
                adj = adj.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                predictions = self.model(x, adj)
                
                # Compute loss
                y_target = y[:, -1, :, :]
                loss = self.criterion(predictions, y_target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu())
                all_targets.append(y_target.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        print(f"Test Loss: {avg_loss:.6f}")
        print(f"Test Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        return avg_loss, metrics, all_predictions, all_targets
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint
        
        Args:
            path (str): Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint
        
        Args:
            path (str): Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
