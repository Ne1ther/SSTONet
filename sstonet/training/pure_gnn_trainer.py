# -*- coding: utf-8 -*-
"""Trainer for Pure GNN baseline model."""

import sys
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class PureGNNTrainer:
    """Trainer for Pure GNN baseline model.

    Similar to GNNDeepONetTrainer but handles the per-sample GNN forward pass
    required by PureGNN architecture.
    """

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
        batch_log_interval: int = 1,
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
        print("Normalizing data...", flush=True)
        X_train_norm, coords_norm, T_train_norm = self._normalize_data(
            X_train, coords, T_train, fit=True
        )

        if X_val is not None:
            X_val_norm, _, T_val_norm = self._normalize_data(X_val, coords, T_val)
        print("Data normalization done.", flush=True)

        # Convert to tensors
        print(f"Converting to tensors and moving to {self.device}...", flush=True)
        coords_t = torch.FloatTensor(coords_norm).to(self.device)
        edge_index_t = torch.LongTensor(edge_index).to(self.device)
        edge_weight_t = torch.FloatTensor(edge_weight).to(self.device) if edge_weight is not None else None
        X_train_t = torch.FloatTensor(X_train_norm).to(self.device)
        T_train_t = torch.FloatTensor(T_train_norm).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
            T_val_t = torch.FloatTensor(T_val_norm).to(self.device)
        print(f"Tensors ready. X_train: {X_train_t.shape}, T_train: {T_train_t.shape}", flush=True)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        n_samples = len(X_train)

        # Keep unscaled validation temperatures to compute RMSE in physical units.
        T_val_orig = T_val if X_val is not None else None

        # Detect whether the terminal is interactive.
        is_tty = sys.stdout.isatty()

        # Use a tqdm progress bar for interactive terminals and text output otherwise.
        print(f"Starting training for {epochs} epochs, batch_size={batch_size}...", flush=True)
        epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if is_tty else range(epochs)
        for epoch in epoch_iter:
            self.model.train()
            total_sse = torch.zeros((), device=self.device)  # sum of squared error (normalized)
            total_elems = 0
            n_batches = 0  # number of optimizer steps

            # Random shuffle
            perm = torch.randperm(n_samples, device=self.device)
            total_batches = int(math.ceil(n_samples / batch_size))

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
                n_batches += 1
                total_sse += loss.detach() * pred.numel()
                total_elems += int(pred.numel())

                if verbose and batch_log_interval > 0 and (n_batches % batch_log_interval == 0):
                    print(
                        f"Epoch {epoch+1}/{epochs} | Batch {n_batches}/{total_batches}",
                        flush=True,
                    )

            avg_train_loss = (total_sse / max(1, total_elems)).item()
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(scheduler.get_last_lr()[0])

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    # PureGNN is expensive; evaluate in mini-batches to avoid huge
                    # memory spikes from running the full validation set at once.
                    val_sse = torch.zeros((), device=self.device)
                    val_elems = 0
                    for j in range(0, X_val_t.size(0), batch_size):
                        Xb = X_val_t[j:j + batch_size]
                        Tb = T_val_t[j:j + batch_size]
                        pred_b = self.model(Xb, coords_t, edge_index_t, edge_weight_t)
                        diff = pred_b - Tb
                        val_sse += (diff * diff).sum()
                        val_elems += int(diff.numel())
                    val_loss = (val_sse / max(1, val_elems)).item()

                    # StandardScaler is fitted globally on T; RMSE on original scale
                    # is simply sqrt(MSE_norm) * std.
                    val_rmse = float(np.sqrt(val_loss) * float(self.scaler_T.scale_[0]))

                history['val_loss'].append(val_loss)
                history['val_rmse'].append(val_rmse)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    if save_dir:
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        self.save(Path(save_dir) / 'model_best.pt')

            # Save checkpoint every 100 epochs
            if save_dir and (epoch + 1) % 100 == 0:
                self.save(Path(save_dir) / f'model_epoch_{epoch+1}.pt')

            scheduler.step()

            # Update the progress display.
            if verbose:
                if is_tty:
                    postfix = {'loss': f'{avg_train_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.1e}'}
                    if X_val is not None:
                        postfix['val_loss'] = f'{val_loss:.4f}'
                        postfix['RMSE'] = f'{val_rmse:.2f}K'
                    epoch_iter.set_postfix(postfix)
                else:
                    # Print every epoch in non-interactive mode.
                    msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f}"
                    if X_val is not None:
                        msg += f" | Val: {val_loss:.4f} | RMSE: {val_rmse:.2f}K"
                    msg += f" | LR: {scheduler.get_last_lr()[0]:.1e}"
                    print(msg, flush=True)

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save final model
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.save(Path(save_dir) / 'model_final.pt')

        return history

    def predict(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        edge_index: np.ndarray,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Predict in bounded batches, preserving case and reference-node order.

        The full-field graph expands every case over all nodes. Sending an
        entire test set to MPS at once can exceed safe tensor/indexing limits.
        Only one prediction batch is kept on the accelerator at a time.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if len(X) == 0:
            return np.empty((0, len(coords)), dtype=np.float32)
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
            coords_t = torch.FloatTensor(coords_norm).to(self.device)
            edge_index_t = torch.LongTensor(edge_index).to(self.device)
            edge_weight_t = torch.FloatTensor(edge_weight).to(self.device) if edge_weight is not None else None
            pred = np.empty((len(X), len(coords)), dtype=np.float32)
            for start in range(0, len(X), batch_size):
                stop = min(start + batch_size, len(X))
                X_t = torch.FloatTensor(X_norm[start:stop]).to(self.device)
                pred_norm = self.model(X_t, coords_t, edge_index_t, edge_weight_t)
                pred_batch = self._denormalize_T(pred_norm.cpu().numpy())
                if pred_batch.shape != (stop - start, len(coords)):
                    raise ValueError("prediction shape does not match case/node ordering")
                if not np.isfinite(pred_batch).all():
                    raise FloatingPointError(f"non-finite predictions in cases {start}:{stop}")
                pred[start:stop] = pred_batch
                del X_t, pred_norm
        return pred

    def validate_prediction(self, X_val, coords, edge_index, T_val, history, batch_size=4):
        """Replay the selected checkpoint on validation data before test export.

        This checks evaluation fidelity, not test accuracy: a legitimately poor
        model must still be reported if its predictions match training history.
        """
        recorded = np.asarray(history['val_rmse'], dtype=float)
        if recorded.size == 0 or not np.isfinite(recorded).all():
            raise ValueError("finite validation history is required")
        pred = self.predict(X_val, coords, edge_index, batch_size=batch_size)
        rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - T_val) ** 2)))
        expected = float(recorded.min())
        if not np.isclose(rmse, expected, rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"Validation replay disagrees with training history: {rmse:.8g} K "
                f"vs {expected:.8g} K. Check checkpoint, scalers, node order and batch size."
            )
        return {
            'protocol': 'pure_gnn_batched_v1',
            'batch_size': int(batch_size),
            'validation_count': int(len(X_val)),
            'best_epoch': int(recorded.argmin()) + 1,
            'validation_rmse_K': rmse,
            'expected_validation_rmse_K': expected,
            'validation_replay_passed': True,
            'replay_rtol': 1e-4,
            'replay_atol_K': 1e-4,
        }

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
        """Load model checkpoint."""
        # Trusted local checkpoints also contain fitted sklearn scalers.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_X' in checkpoint:
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_coords = checkpoint['scaler_coords']
            self.scaler_T = checkpoint['scaler_T']
