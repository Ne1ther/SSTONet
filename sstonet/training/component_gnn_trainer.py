# -*- coding: utf-8 -*-
"""Trainer for ComponentAwareGNNDeepONet."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class ComponentAwareGNNTrainer:
    """Trainer for ComponentAwareGNNDeepONet with dynamic graph construction."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None,
    ):
        if device is None or device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

        self.scaler_X = StandardScaler()
        self.scaler_coords = StandardScaler()
        self.scaler_T = StandardScaler()

    def _normalize_data(self, X: np.ndarray, coords: np.ndarray, T: np.ndarray, fit: bool = False):
        if fit:
            X_norm = self.scaler_X.fit_transform(X)
            coords_norm = self.scaler_coords.fit_transform(coords)
            self.scaler_T.fit(T.reshape(-1, 1))
            T_norm = self.scaler_T.transform(T.reshape(-1, 1)).reshape(T.shape)
        else:
            X_norm = self.scaler_X.transform(X)
            coords_norm = self.scaler_coords.transform(coords)
            T_norm = self.scaler_T.transform(T.reshape(-1, 1)).reshape(T.shape)
        return X_norm, coords_norm, T_norm

    def _denormalize_T(self, T: np.ndarray) -> np.ndarray:
        return self.scaler_T.inverse_transform(T.reshape(-1, 1)).reshape(T.shape)

    def fit(
        self,
        X_train: np.ndarray,
        coords: np.ndarray,
        comp_ids: np.ndarray,
        T_train: np.ndarray,
        X_val: np.ndarray = None,
        T_val: np.ndarray = None,
        epochs: int = 500,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        verbose: bool = True,
        save_dir: str = None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            X_train: Training inputs (n_samples, branch_input_dim)
            coords: Node coordinates (N, 3) - raw, unnormalized
            comp_ids: Component IDs (N,)
            T_train: Training targets (n_samples, N)
            X_val: Validation inputs
            T_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            grad_clip: Gradient clipping value
            verbose: Print progress
            save_dir: Directory to save checkpoints
        """
        # Build graph from raw coordinates BEFORE normalization
        print("Building component-aware graph...")
        graph_result = self.model.build_graph(coords, comp_ids)
        if self.model.use_edge_attr:
            edge_index, edge_attr = graph_result
            # Convert distance to weight using Gaussian kernel
            edge_weight = np.exp(-edge_attr ** 2 / (2 * edge_attr.mean() ** 2))
        else:
            edge_index = graph_result
            edge_weight = None
        print(f"Graph built: {coords.shape[0]} nodes, {edge_index.shape[1]} edges")

        # Normalize data
        X_train_norm, coords_norm, T_train_norm = self._normalize_data(X_train, coords, T_train, fit=True)

        if X_val is not None:
            X_val_norm, _, T_val_norm = self._normalize_data(X_val, coords, T_val)

        # Convert to tensors
        coords_t = torch.FloatTensor(coords_norm).to(self.device)
        edge_index_t = torch.LongTensor(edge_index).to(self.device)
        edge_weight_t = torch.FloatTensor(edge_weight).to(self.device) if edge_weight is not None else None
        X_train_t = torch.FloatTensor(X_train_norm).to(self.device)
        T_train_t = torch.FloatTensor(T_train_norm).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
            T_val_t = torch.FloatTensor(T_val_norm).to(self.device)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        n_samples = len(X_train)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
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

            scheduler.step()

            # Save periodic checkpoint
            if save_dir and (epoch + 1) % 100 == 0:
                self.save(Path(save_dir) / f'checkpoint_epoch{epoch+1}.pt')

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                msg += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.save(Path(save_dir) / 'model_final.pt')

        return history

    def predict(self, X: np.ndarray, coords: np.ndarray, comp_ids: np.ndarray) -> np.ndarray:
        """Predict temperatures.

        Note: If coords differ from training, graph will be rebuilt.
        """
        # Build graph from raw coordinates
        graph_result = self.model.build_graph(coords, comp_ids)
        if self.model.use_edge_attr:
            edge_index, edge_attr = graph_result
            edge_weight = np.exp(-edge_attr ** 2 / (2 * edge_attr.mean() ** 2))
        else:
            edge_index = graph_result
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

        return self._denormalize_T(pred_norm.cpu().numpy())

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_coords': self.scaler_coords,
            'scaler_T': self.scaler_T,
        }, path)

    def load(self, path: str):
        """Load a trusted checkpoint created by this trainer."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_X' in checkpoint:
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_coords = checkpoint['scaler_coords']
            self.scaler_T = checkpoint['scaler_T']
