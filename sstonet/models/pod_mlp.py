# -*- coding: utf-8 -*-
"""POD Analyzer and POD+MLP model"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from ..utils.metrics import compute_metrics

class PODAnalyzer:
    """Compute a proper orthogonal decomposition of temperature fields."""

    def __init__(self):
        self.mean = None
        self.basis = None
        self.singular_values = None
        self.energy_ratio = None

    def fit(self, T: np.ndarray, n_modes: int = None) -> 'PODAnalyzer':
        self.mean = T.mean(axis=0)
        T_centered = T - self.mean

        U, S, Vt = np.linalg.svd(T_centered, full_matrices=False)

        self.singular_values = S
        self.energy_ratio = (S ** 2) / (S ** 2).sum()
        self.cumulative_energy = np.cumsum(self.energy_ratio)

        if n_modes is None:
            n_modes = len(S)
        self.basis = Vt[:n_modes].T

        return self

    def project(self, T: np.ndarray) -> np.ndarray:
        T_centered = T - self.mean
        return T_centered @ self.basis

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return coeffs @ self.basis.T + self.mean

    def reconstruction_error(self, T: np.ndarray, n_modes: int = None) -> float:
        if n_modes is not None:
            basis = self.basis[:, :n_modes]
            T_centered = T - self.mean
            coeffs = T_centered @ basis
            T_recon = coeffs @ basis.T + self.mean
        else:
            coeffs = self.project(T)
            T_recon = self.reconstruct(coeffs)

        return np.sqrt(((T - T_recon) ** 2).mean())

    def plot_energy(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].semilogy(self.energy_ratio[:100], 'b-o', markersize=3)
        axes[0].set_xlabel('Mode Index')
        axes[0].set_ylabel('Energy Ratio')
        axes[0].set_title('Individual Mode Energy')
        axes[0].grid(True)

        axes[1].plot(self.cumulative_energy[:100], 'r-o', markersize=3)
        axes[1].axhline(y=0.99, color='g', linestyle='--', label='99%')
        axes[1].axhline(y=0.999, color='orange', linestyle='--', label='99.9%')
        axes[1].set_xlabel('Number of Modes')
        axes[1].set_ylabel('Cumulative Energy')
        axes[1].set_title('Cumulative Energy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

        for threshold in [0.9, 0.95, 0.99, 0.999]:
            n_modes = np.searchsorted(self.cumulative_energy, threshold) + 1
            print(f"{threshold*100:.1f}% energy: {n_modes} modes")


class PODMLP:
    """Thermal surrogate combining a POD basis and an MLP coefficient model."""

    def __init__(self, n_modes: int = 50, hidden_layers: List[int] = [256, 128], dropout: float = 0.1):
        self.n_modes = n_modes
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.pod = PODAnalyzer()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def _build_model(self, input_dim: int):
        import torch.nn as nn

        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.n_modes))

        return nn.Sequential(*layers)

    def fit(self, X_train: np.ndarray, T_train: np.ndarray,
            X_val: np.ndarray = None, T_val: np.ndarray = None,
            epochs: int = 500, lr: float = 1e-3, batch_size: int = 32,
            verbose: bool = True, save_dir: Optional[str] = None,
            device: Optional[str] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print("Computing POD basis...")
        self.pod.fit(T_train, n_modes=self.n_modes)

        y_train = self.pod.project(T_train)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        self.model = self._build_model(X_train.shape[1])
        if device is not None and device != 'auto':
            device = torch.device(device)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        self.model.to(device)

        X_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_tensor = torch.FloatTensor(y_train_scaled).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            y_val = self.pod.project(T_val)
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.MSELoss()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        def save_checkpoint(path: str) -> None:
            torch.save({
                "model_state": self.model.state_dict(),
                "n_modes": self.n_modes,
                "hidden_layers": self.hidden_layers,
                "pod_mean": self.pod.mean,
                "pod_basis": self.pod.basis,
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
            }, path)

        best_val_loss = float('inf')
        best_state = None
        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(dataloader)
            history['train_loss'].append(train_loss)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = criterion(y_val_pred, y_val_tensor).item()
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }

            scheduler.step()
            lr_value = optimizer.param_groups[0]['lr']
            history['lr'].append(lr_value)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                msg += f", LR: {lr_value:.2e}"
                print(msg)

            if save_dir and (epoch + 1) % 100 == 0:
                save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt'))

        if X_val is not None:
            self.model.load_state_dict(best_state)

        if save_dir:
            save_checkpoint(os.path.join(save_dir, 'model_final.pt'))

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        T_pred = self.pod.reconstruct(y_pred)

        return T_pred

    def evaluate(self, X: np.ndarray, T_true: np.ndarray) -> Dict[str, float]:
        T_pred = self.predict(X)
        return compute_metrics(T_pred, T_true)
