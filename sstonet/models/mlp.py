# -*- coding: utf-8 -*-
"""Pure MLP baseline model for surrogate modeling.

This module provides a simple MLP baseline for comparison with operator learning methods.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional

from .deeponet import init_weights, ACTIVATIONS
from ..utils.metrics import compute_metrics


class MLP(nn.Module):
    """Pure MLP model: Direct mapping from input parameters to output field.

    A simple baseline that directly maps input parameters to the full output field
    without any structural assumptions about the operator.

    Args:
        input_dim: Input dimension (number of parameters)
        output_dim: Output dimension (number of output points)
        hidden_dims: List of hidden layer dimensions
        bottleneck_dim: Optional bottleneck dimension inserted before output
        activation: Activation function name
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout rate
        initializer: Weight initializer name
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256, 256, 128],
        bottleneck_dim: Optional[int] = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout: float = 0.1,
        initializer: str = 'he_uniform',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim

        if bottleneck_dim is not None and bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be a positive integer when provided.")

        act_class = ACTIVATIONS[activation]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_class())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        if bottleneck_dim is not None:
            layers.append(nn.Linear(prev_dim, bottleneck_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(bottleneck_dim))
            layers.append(act_class())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = bottleneck_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        init_weights(self.net, initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input parameters (batch, input_dim)

        Returns:
            Output field (batch, output_dim)
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPTrainer:
    """Trainer for MLP model with data normalization and learning rate scheduling."""

    def __init__(
        self,
        model: MLP,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        self.model = model
        if device and device != "auto":
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = None
        self.criterion = nn.MSELoss()

        # Normalization statistics
        self.X_mean = None
        self.X_std = None
        self.T_mean = None
        self.T_std = None

    def _normalize_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize input features."""
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std

    def _normalize_T(self, T: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize output field."""
        if fit:
            self.T_mean = T.mean()
            self.T_std = T.std() + 1e-8
        return (T - self.T_mean) / self.T_std

    def _denormalize_T(self, T: np.ndarray) -> np.ndarray:
        """Denormalize output field."""
        return T * self.T_std + self.T_mean

    def fit(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch_size: int = 64,
        verbose: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            X_train: Training input (n_samples, input_dim)
            T_train: Training output (n_samples, output_dim)
            X_val: Validation input
            T_val: Validation output
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            save_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Normalize data
        X_train_norm = self._normalize_X(X_train, fit=True)
        T_train_norm = self._normalize_T(T_train, fit=True)

        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        T_tensor = torch.FloatTensor(T_train_norm).to(self.device)

        dataset = TensorDataset(X_tensor, T_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_norm = self._normalize_X(X_val)
            T_val_norm = self._normalize_T(T_val)
            X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
            T_val_tensor = torch.FloatTensor(T_val_norm).to(self.device)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_state = None
        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, T_batch in dataloader:
                self.optimizer.zero_grad()
                T_pred = self.model(X_batch)
                loss = self.criterion(T_pred, T_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(dataloader)
            history['train_loss'].append(train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    T_val_pred = self.model(X_val_tensor)
                    val_loss = self.criterion(T_val_pred, T_val_tensor).item()
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            self.scheduler.step()
            lr_value = self.optimizer.param_groups[0]['lr']
            history['lr'].append(lr_value)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                msg += f", LR: {lr_value:.2e}"
                print(msg)

            # Save checkpoint
            if save_dir and (epoch + 1) % 100 == 0:
                self.save(os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt'))

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save final model
        if save_dir:
            self.save(os.path.join(save_dir, 'model_final.pt'))

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input parameters (n_samples, input_dim)

        Returns:
            Predicted output field (n_samples, output_dim)
        """
        self.model.eval()
        X_norm = self._normalize_X(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            T_pred_norm = self.model(X_tensor).cpu().numpy()

        return self._denormalize_T(T_pred_norm)

    def evaluate(self, X: np.ndarray, T_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Input parameters
            T_true: Ground truth output

        Returns:
            Dictionary of metrics
        """
        T_pred = self.predict(X)
        return compute_metrics(T_pred, T_true)

    def save(self, path: str) -> None:
        """Save model and normalization statistics."""
        torch.save({
            'model_state': self.model.state_dict(),
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'T_mean': self.T_mean,
            'T_std': self.T_std,
        }, path)

    def load(self, path: str) -> None:
        """Load a trusted checkpoint created by this trainer."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.X_mean = checkpoint['X_mean']
        self.X_std = checkpoint['X_std']
        self.T_mean = checkpoint['T_mean']
        self.T_std = checkpoint['T_std']
