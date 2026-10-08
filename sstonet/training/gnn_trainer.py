# -*- coding: utf-8 -*-
"""Trainer for GNN-DeepONet with proper normalization."""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class GNNDeepONetTrainer:
    """Trainer for GNN-enhanced DeepONet."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None,
        use_edge_attr: bool = False,
    ):
        if device is None or device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.use_edge_attr = use_edge_attr

        # Scalers for normalization
        self.scaler_X = StandardScaler()
        self.scaler_coords = StandardScaler()
        self.scaler_T = StandardScaler()

    def _normalize_data(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        T: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize input data."""
        if fit:
            X_norm = self.scaler_X.fit_transform(X)
            coords_norm = self.scaler_coords.fit_transform(coords)
            T_flat = T.reshape(-1, 1)
            self.scaler_T.fit(T_flat)
            T_norm = self.scaler_T.transform(T_flat).reshape(T.shape)
        else:
            X_norm = self.scaler_X.transform(X)
            coords_norm = self.scaler_coords.transform(coords)
            T_norm = self.scaler_T.transform(T.reshape(-1, 1)).reshape(T.shape)
        return X_norm, coords_norm, T_norm

    def _denormalize_T(self, T: np.ndarray) -> np.ndarray:
        """Denormalize temperature field."""
        return self.scaler_T.inverse_transform(T.reshape(-1, 1)).reshape(T.shape)

    def fit(
        self,
        X_train: np.ndarray,
        coords: np.ndarray,
        edge_index: np.ndarray,
        T_train: np.ndarray,
        X_val: np.ndarray = None,
        T_val: np.ndarray = None,
        epochs: int = 500,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        verbose: bool = True,
        save_dir: str = None,
    ) -> Dict[str, List[float]]:
        """Train the model."""

        # Compute edge weight if use_edge_attr is enabled
        if self.use_edge_attr:
            src, dst = edge_index
            edge_attr = np.linalg.norm(coords[src] - coords[dst], axis=1)
            edge_weight = np.exp(-edge_attr ** 2 / (2 * edge_attr.mean() ** 2))
        else:
            edge_weight = None

        # Normalize data
        X_train_norm, coords_norm, T_train_norm = self._normalize_data(
            X_train, coords, T_train, fit=True
        )

        if X_val is not None:
            X_val_norm, _, T_val_norm = self._normalize_data(X_val, coords, T_val)

        # Convert the data to tensors.
        coords_t = torch.FloatTensor(coords_norm).to(self.device)
        edge_index_t = torch.LongTensor(edge_index).to(self.device)
        edge_weight_t = torch.FloatTensor(edge_weight).to(self.device) if edge_weight is not None else None
        X_train_t = torch.FloatTensor(X_train_norm).to(self.device)
        T_train_t = torch.FloatTensor(T_train_norm).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
            T_val_t = torch.FloatTensor(T_val_norm).to(self.device)

        # Configure the learning-rate scheduler.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        n_samples = len(X_train)

        # Detect whether the terminal is interactive.
        is_tty = sys.stdout.isatty()

        # Use a tqdm progress bar for interactive terminals and text output otherwise.
        epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if is_tty else range(epochs)
        for epoch in epoch_iter:
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            # Shuffle the training samples.
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                X_batch = X_train_t[idx]
                T_batch = T_train_t[idx]

                self.optimizer.zero_grad()
                pred = self.model(X_batch, coords_t, edge_index_t, edge_weight_t)
                loss = self.criterion(pred, T_batch)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(scheduler.get_last_lr()[0])

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t, coords_t, edge_index_t, edge_weight_t)
                    val_loss = self.criterion(val_pred, T_val_t).item()
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    if save_dir:
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        self.save(Path(save_dir) / 'model_best.pt')

            # Save a checkpoint every 100 epochs.
            if save_dir and (epoch + 1) % 100 == 0:
                self.save(Path(save_dir) / f'model_epoch_{epoch+1}.pt')

            scheduler.step()

            # Update the progress display.
            if verbose:
                if is_tty:
                    postfix = {'train': f'{avg_train_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.1e}'}
                    if X_val is not None:
                        postfix['val'] = f'{val_loss:.4f}'
                    epoch_iter.set_postfix(postfix)
                else:
                    msg = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f}"
                    if X_val is not None:
                        msg += f" | Val: {val_loss:.4f}"
                    msg += f" | LR: {scheduler.get_last_lr()[0]:.1e}"
                    print(msg, flush=True)

        # Restore the best model state.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save the final model.
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.save(Path(save_dir) / 'model_final.pt')

        return history

    def predict(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Predict temperatures."""
        # Compute edge weight if use_edge_attr is enabled
        if self.use_edge_attr:
            src, dst = edge_index
            edge_attr = np.linalg.norm(coords[src] - coords[dst], axis=1)
            edge_weight = np.exp(-edge_attr ** 2 / (2 * edge_attr.mean() ** 2))
        else:
            edge_weight = None

        # Normalize inputs
        X_norm = self.scaler_X.transform(X)
        coords_norm = self.scaler_coords.transform(coords)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_norm).to(self.device)
            coords_t = torch.FloatTensor(coords_norm).to(self.device)
            edge_index_t = torch.LongTensor(edge_index).to(self.device)
            edge_weight_t = torch.FloatTensor(edge_weight).to(self.device) if edge_weight is not None else None
            pred_norm = self.model(X_t, coords_t, edge_index_t, edge_weight_t)

        # Denormalize output
        pred = self._denormalize_T(pred_norm.cpu().numpy())
        return pred

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_coords': self.scaler_coords,
            'scaler_T': self.scaler_T,
        }, path)

    def load(self, path: str):
        """Load a trusted checkpoint created by this trainer."""
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_X' in checkpoint:
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_coords = checkpoint['scaler_coords']
            self.scaler_T = checkpoint['scaler_T']
