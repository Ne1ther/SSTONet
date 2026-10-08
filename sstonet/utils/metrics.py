# -*- coding: utf-8 -*-
"""Evaluation metrics"""

import numpy as np
from typing import Dict


def compute_metrics(T_pred: np.ndarray, T_true: np.ndarray) -> Dict[str, float]:
    """Compute the full set of evaluation metrics.

    Args:
        T_pred: Predicted temperature fields, with shape (n_samples, n_points).
        T_true: Reference temperature fields, with shape (n_samples, n_points).

    Returns:
        Dictionary containing all metrics.
    """
    # RMSE
    mse = ((T_pred - T_true) ** 2).mean()
    rmse = np.sqrt(mse)

    # MAE
    mae = np.abs(T_pred - T_true).mean()

    # MaxError
    max_error = np.abs(T_pred - T_true).max()

    # Relative L2 error, commonly used to evaluate operator networks.
    rel_l2 = np.mean(
        np.linalg.norm(T_pred - T_true, axis=1) /
        (np.linalg.norm(T_true, axis=1) + 1e-8)
    )

    # R² score
    ss_res = ((T_true - T_pred) ** 2).sum()
    ss_tot = ((T_true - T_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # Normalized RMSE.
    T_range = T_true.max() - T_true.min()
    nrmse = rmse / T_range if T_range > 0 else 0.0

    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MaxError': float(max_error),
        'RelL2': float(rel_l2),
        'R2': float(r2),
        'nRMSE': float(nrmse),
    }


def print_metrics(model_name: str, metrics: Dict[str, float]) -> None:
    """Print formatted evaluation metrics."""
    print(f"\n[{model_name}] Test Metrics:")
    print(f"  RMSE:     {metrics['RMSE']:.4f} K")
    print(f"  MAE:      {metrics['MAE']:.4f} K")
    print(f"  MaxError: {metrics['MaxError']:.4f} K")
    print(f"  RelL2:    {metrics['RelL2']*100:.4f}%")
    print(f"  R²:       {metrics['R2']:.6f}")
    print(f"  nRMSE:    {metrics['nRMSE']*100:.4f}%")
