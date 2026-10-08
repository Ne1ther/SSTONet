# -*- coding: utf-8 -*-
"""Trainers for DeepONet, MIONet, PODDeepONet, and PODMIONet.

This module provides unified training interfaces for all operator network models
with support for:
- Data normalization (StandardScaler)
- Learning rate scheduling (CosineAnnealingLR)
- Gradient clipping
- Early stopping with best model restoration
- Checkpoint saving/loading
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from ..models.deeponet import DeepONet, MIONet, PODDeepONet, PODMIONet
from ..utils.metrics import compute_metrics


def _is_pod_model(model: nn.Module) -> bool:
    """Check if model is a POD-based model that handles its own output scaling."""
    return isinstance(model, (PODDeepONet, PODMIONet))


class BaseTrainer:
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model: nn.Module,
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

        # Normalization
        self.scaler_X = StandardScaler()
        self.scaler_coords = StandardScaler()
        self.scaler_T = StandardScaler()

    def _setup_scheduler(self, epochs: int, eta_min: float = 1e-6):
        """Setup learning rate scheduler."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=eta_min
        )

    def _normalize_data(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        T: np.ndarray,
        fit: bool = False,
        skip_T: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize input data.

        Args:
            skip_T: If True, don't normalize T (for POD models that handle their own scaling)
        """
        if fit:
            X_norm = self.scaler_X.fit_transform(X)
            coords_norm = self.scaler_coords.fit_transform(coords)
            if skip_T:
                T_norm = T
                self._skip_T_norm = True
            else:
                T_flat = T.reshape(-1, 1)
                self.scaler_T.fit(T_flat)
                T_norm = self.scaler_T.transform(T_flat).reshape(T.shape)
                self._skip_T_norm = False
        else:
            X_norm = self.scaler_X.transform(X)
            coords_norm = self.scaler_coords.transform(coords)
            if skip_T or getattr(self, '_skip_T_norm', False):
                T_norm = T
            else:
                T_norm = self.scaler_T.transform(T.reshape(-1, 1)).reshape(T.shape)
        return X_norm, coords_norm, T_norm

    def _denormalize_T(self, T: np.ndarray) -> np.ndarray:
        """Denormalize temperature field."""
        if getattr(self, '_skip_T_norm', False):
            return T
        return self.scaler_T.inverse_transform(T.reshape(-1, 1)).reshape(T.shape)

    def evaluate(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        T_true: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Input parameters
            coords: Spatial coordinates
            T_true: Ground truth temperature field

        Returns:
            Dictionary of metrics
        """
        T_pred = self.predict(X, coords)
        return compute_metrics(T_pred, T_true)

    def save(self, path: str) -> None:
        """Save model and training state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_coords': self.scaler_coords,
            'scaler_T': self.scaler_T,
            '_skip_T_norm': getattr(self, '_skip_T_norm', False),
        }, path)

    def load(self, path: str) -> None:
        """Load a trusted checkpoint created by this trainer."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_coords = checkpoint['scaler_coords']
        self.scaler_T = checkpoint['scaler_T']
        self._skip_T_norm = checkpoint.get('_skip_T_norm', False)


class DeepONetTrainer(BaseTrainer):
    """Trainer for DeepONet and PODDeepONet models."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        super().__init__(model, lr, weight_decay, device)

    def fit(
        self,
        X_train: np.ndarray,
        coords: np.ndarray,
        T_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        verbose: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            X_train: Training input parameters (n_samples, input_dim)
            coords: Spatial coordinates (n_points, coord_dim)
            T_train: Training temperature field (n_samples, n_points)
            X_val: Validation input parameters
            T_val: Validation temperature field
            epochs: Number of training epochs
            batch_size: Batch size
            grad_clip: Gradient clipping value
            verbose: Whether to print progress
            save_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        # Normalize data (skip T normalization for POD models)
        skip_T = _is_pod_model(self.model)
        X_norm, coords_norm, T_norm = self._normalize_data(X_train, coords, T_train, fit=True, skip_T=skip_T)

        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        coords_tensor = torch.FloatTensor(coords_norm).to(self.device)
        T_tensor = torch.FloatTensor(T_norm).to(self.device)
        self._eval_batch_size = int(batch_size)

        if X_val is not None:
            X_val_norm, _, T_val_norm = self._normalize_data(X_val, coords, T_val)
            X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
            T_val_tensor = torch.FloatTensor(T_val_norm).to(self.device)

        # Setup scheduler
        self._setup_scheduler(epochs)

        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        n_samples = len(X_train)

        for epoch in range(epochs):
            self.model.train()

            # Shuffle data
            perm = torch.randperm(n_samples)
            train_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                X_batch = X_tensor[idx]
                T_batch = T_tensor[idx]

                self.optimizer.zero_grad()
                T_pred = self.model(X_batch, coords_tensor)
                loss = self.criterion(T_pred, T_batch)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            history['train_loss'].append(train_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss_sum = 0.0
                    val_count = 0
                    for j in range(0, X_val_tensor.size(0), batch_size):
                        X_val_batch = X_val_tensor[j:j + batch_size]
                        T_val_batch = T_val_tensor[j:j + batch_size]
                        T_val_pred = self.model(X_val_batch, coords_tensor)
                        val_loss_sum += self.criterion(T_val_pred, T_val_batch).item() * X_val_batch.size(0)
                        val_count += X_val_batch.size(0)
                    val_loss = val_loss_sum / max(1, val_count)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    # Save best model
                    if save_dir:
                        self.save(os.path.join(save_dir, 'model_best.pt'))

            self.scheduler.step()

            # Logging
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                msg += f", LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                print(msg, flush=True)

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

    def predict(self, X: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input parameters (n_samples, input_dim)
            coords: Spatial coordinates (n_points, coord_dim)

        Returns:
            Predicted temperature field (n_samples, n_points)
        """
        X_norm = self.scaler_X.transform(X)
        coords_norm = self.scaler_coords.transform(coords)

        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        coords_tensor = torch.FloatTensor(coords_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_size = int(getattr(self, "_eval_batch_size", 64))
            preds = []
            for i in range(0, X_tensor.size(0), batch_size):
                preds.append(self.model(X_tensor[i:i + batch_size], coords_tensor).cpu())
            T_pred_norm = torch.cat(preds, dim=0).numpy()

        return self._denormalize_T(T_pred_norm)


class MIONetTrainer(BaseTrainer):
    """Trainer for MIONet and PODMIONet models."""

    # Input dimension mapping for satellite thermal analysis
    INPUT_DIMS = {
        'solar': 3,
        'radiation': 1328,
        'load': 20,
        'optical': 56,
        'coupling': 15
    }

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None,
        input_dims: Optional[Dict[str, int]] = None
    ):
        super().__init__(model, lr, weight_decay, device)
        self.input_dims = input_dims or self.INPUT_DIMS
        self.scalers = {}

    def _split_inputs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Split input array into separate input groups."""
        idx = 0
        inputs = {}
        for name, dim in self.input_dims.items():
            inputs[name] = X[:, idx:idx+dim]
            idx += dim
        return inputs

    def _normalize_inputs(
        self,
        inputs: Dict[str, np.ndarray],
        fit: bool = False
    ) -> Dict[str, np.ndarray]:
        """Normalize each input group separately."""
        inputs_norm = {}
        for name, data in inputs.items():
            if fit:
                self.scalers[name] = StandardScaler()
                inputs_norm[name] = self.scalers[name].fit_transform(data)
            else:
                inputs_norm[name] = self.scalers[name].transform(data)
        return inputs_norm

    def fit(
        self,
        X_train: np.ndarray,
        coords: np.ndarray,
        T_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        verbose: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model."""
        # Split and normalize inputs
        inputs_train = self._split_inputs(X_train)
        inputs_train_norm = self._normalize_inputs(inputs_train, fit=True)

        # Normalize coordinates and temperature (skip T for POD models)
        coords_norm = self.scaler_coords.fit_transform(coords)
        skip_T = _is_pod_model(self.model)
        if skip_T:
            T_norm = T_train
            self._skip_T_norm = True
        else:
            T_flat = T_train.reshape(-1, 1)
            self.scaler_T.fit(T_flat)
            T_norm = self.scaler_T.transform(T_flat).reshape(T_train.shape)
            self._skip_T_norm = False

        # Convert to tensors
        inputs_tensor = {
            name: torch.FloatTensor(data).to(self.device)
            for name, data in inputs_train_norm.items()
        }
        coords_tensor = torch.FloatTensor(coords_norm).to(self.device)
        T_tensor = torch.FloatTensor(T_norm).to(self.device)

        if X_val is not None:
            inputs_val = self._split_inputs(X_val)
            inputs_val_norm = self._normalize_inputs(inputs_val)
            inputs_val_tensor = {
                name: torch.FloatTensor(data).to(self.device)
                for name, data in inputs_val_norm.items()
            }
            if skip_T:
                T_val_norm = T_val
            else:
                T_val_norm = self.scaler_T.transform(T_val.reshape(-1, 1)).reshape(T_val.shape)
            T_val_tensor = torch.FloatTensor(T_val_norm).to(self.device)

        # Setup scheduler
        self._setup_scheduler(epochs)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        n_samples = len(X_train)

        for epoch in range(epochs):
            self.model.train()

            perm = torch.randperm(n_samples)
            train_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                inputs_batch = {name: tensor[idx] for name, tensor in inputs_tensor.items()}
                T_batch = T_tensor[idx]

                self.optimizer.zero_grad()
                T_pred = self.model(inputs_batch, coords_tensor)
                loss = self.criterion(T_pred, T_batch)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            history['train_loss'].append(train_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    T_val_pred = self.model(inputs_val_tensor, coords_tensor)
                    val_loss = self.criterion(T_val_pred, T_val_tensor).item()
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    # Save best model
                    if save_dir:
                        self.save(os.path.join(save_dir, 'model_best.pt'))

            self.scheduler.step()

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                msg += f", LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                print(msg)

            if save_dir and (epoch + 1) % 100 == 0:
                self.save(os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt'))

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_dir:
            self.save(os.path.join(save_dir, 'model_final.pt'))

        return history

    def predict(self, X: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Make predictions."""
        inputs = self._split_inputs(X)
        inputs_norm = self._normalize_inputs(inputs)
        coords_norm = self.scaler_coords.transform(coords)

        inputs_tensor = {
            name: torch.FloatTensor(data).to(self.device)
            for name, data in inputs_norm.items()
        }
        coords_tensor = torch.FloatTensor(coords_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            T_pred_norm = self.model(inputs_tensor, coords_tensor).cpu().numpy()

        return self._denormalize_T(T_pred_norm)

    def save(self, path: str) -> None:
        """Save model and training state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scalers': self.scalers,
            'scaler_coords': self.scaler_coords,
            'scaler_T': self.scaler_T,
            '_skip_T_norm': getattr(self, '_skip_T_norm', False),
            'input_dims': self.input_dims,
        }, path)

    def load(self, path: str) -> None:
        """Load a trusted checkpoint created by this trainer."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scalers = checkpoint['scalers']
        self.scaler_coords = checkpoint['scaler_coords']
        self.scaler_T = checkpoint['scaler_T']
        self.input_dims = checkpoint['input_dims']
        self._skip_T_norm = checkpoint.get('_skip_T_norm', False)


# Aliases for POD variants (use same trainers)
PODDeepONetTrainer = DeepONetTrainer
PODMIONetTrainer = MIONetTrainer
