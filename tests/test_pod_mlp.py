# -*- coding: utf-8 -*-
"""Tests for PODAnalyzer and PODMLP."""

import pytest
import numpy as np
import torch
from sstonet.models import PODAnalyzer, PODMLP


def test_pod_analyzer_fit():
    """Test PODAnalyzer fit method."""
    T = np.random.randn(50, 100)  # 50 samples, 100 points
    pod = PODAnalyzer()
    pod.fit(T, n_modes=10)

    assert pod.mean is not None
    assert pod.basis.shape == (100, 10)
    assert pod.singular_values is not None
    assert pod.energy_ratio is not None


def test_pod_analyzer_project_reconstruct():
    """Test PODAnalyzer project and reconstruct."""
    T = np.random.randn(50, 100)
    pod = PODAnalyzer().fit(T, n_modes=20)

    coeffs = pod.project(T)
    assert coeffs.shape == (50, 20)

    T_recon = pod.reconstruct(coeffs)
    assert T_recon.shape == T.shape


def test_pod_analyzer_reconstruction_error():
    """Test PODAnalyzer reconstruction error."""
    T = np.random.randn(50, 100)
    pod = PODAnalyzer().fit(T, n_modes=30)

    error = pod.reconstruction_error(T)
    assert error >= 0

    # More modes should give lower error
    error_10 = pod.reconstruction_error(T, n_modes=10)
    error_20 = pod.reconstruction_error(T, n_modes=20)
    assert error_10 >= error_20


def test_podmlp_creation():
    """Test PODMLP creation."""
    model = PODMLP(n_modes=10, hidden_layers=[32, 16])
    assert model.n_modes == 10
    assert model.hidden_layers == [32, 16]


def test_podmlp_fit():
    """Test PODMLP fit method."""
    X_train = np.random.randn(30, 10)
    T_train = np.random.randn(30, 50)

    model = PODMLP(n_modes=5, hidden_layers=[16])
    history = model.fit(
        X_train, T_train,
        epochs=3,
        batch_size=8,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 3


def test_podmlp_predict():
    """Test PODMLP predict method."""
    X_train = np.random.randn(30, 10)
    T_train = np.random.randn(30, 50)

    model = PODMLP(n_modes=5, hidden_layers=[16])
    model.fit(X_train, T_train, epochs=2, verbose=False)

    X_test = np.random.randn(5, 10)
    T_pred = model.predict(X_test)

    assert T_pred.shape == (5, 50)


def test_podmlp_evaluate():
    """Test PODMLP evaluate method."""
    X_train = np.random.randn(30, 10)
    T_train = np.random.randn(30, 50)

    model = PODMLP(n_modes=5, hidden_layers=[16])
    model.fit(X_train, T_train, epochs=2, verbose=False)

    X_test = np.random.randn(5, 10)
    T_test = np.random.randn(5, 50)
    metrics = model.evaluate(X_test, T_test)

    assert 'RMSE' in metrics
    assert 'MAE' in metrics


def test_podmlp_restores_and_saves_best_validation_epoch(monkeypatch, tmp_path):
    """Later updates must not overwrite the best validation model snapshot."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # A rank-one field with zero validation coefficients gives exact losses.
    inputs = np.array([[-1.0], [0.0], [1.0]])
    temperatures = np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    model = PODMLP(n_modes=1, hidden_layers=[], dropout=0.0)
    updates = iter((0.25, 1.0, 2.0))
    snapshots = []

    def controlled_optimizer_step(optimizer, *args, **kwargs):
        value = next(updates)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.fill_(value if parameter.ndim == 1 else 0.0)
        snapshots.append({key: tensor.detach().clone()
                          for key, tensor in model.model.state_dict().items()})

    monkeypatch.setattr(torch.optim.Adam, "step", controlled_optimizer_step)
    history = model.fit(
        inputs, temperatures,
        X_val=np.zeros((1, 1)), T_val=np.zeros((1, 2)),
        epochs=3, batch_size=3, verbose=False, save_dir=str(tmp_path),
    )

    np.testing.assert_allclose(history["val_loss"], [0.0625, 1.0, 4.0])
    assert not torch.equal(snapshots[0]["0.bias"], snapshots[-1]["0.bias"])
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, snapshots[0][key], rtol=0, atol=0)

    # This locally generated checkpoint includes fitted sklearn scalers.
    checkpoint = torch.load(tmp_path / "model_final.pt", weights_only=False)
    for key, tensor in checkpoint["model_state"].items():
        torch.testing.assert_close(tensor, snapshots[0][key], rtol=0, atol=0)
