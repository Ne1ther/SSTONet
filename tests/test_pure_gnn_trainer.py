# -*- coding: utf-8 -*-
"""Smoke tests for PureGNNTrainer.

Keep this lightweight: small graph + tiny network so it runs quickly on CPU.
"""

import numpy as np
import pytest
import torch

from sstonet.models import PureGNN
from sstonet.training.pure_gnn_trainer import PureGNNTrainer


def _create_simple_edge_index(n_nodes: int) -> np.ndarray:
    """Create a simple chain graph edge index (bidirectional)."""
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    return np.array([src, dst])


def test_pure_gnn_trainer_fit_prints_progress(capsys):
    n_points = 25
    n_train = 10
    param_dim = 6

    model = PureGNN(
        param_dim=param_dim,
        coord_dim=3,
        hidden_dims=[16, 16],
        num_gnn_layers=1,
        activation="silu",
        use_layer_norm=True,
        dropout=0.0,
    )
    trainer = PureGNNTrainer(model, lr=1e-3, device="cpu")

    X_train = np.random.randn(n_train, param_dim).astype(np.float32)
    coords = np.random.randn(n_points, 3).astype(np.float32)
    edge_index = _create_simple_edge_index(n_points)
    T_train = np.random.randn(n_train, n_points).astype(np.float32)

    history = trainer.fit(
        X_train,
        coords,
        edge_index,
        T_train,
        epochs=2,
        batch_size=4,
        verbose=True,
        batch_log_interval=1,
    )

    assert "train_loss" in history
    assert len(history["train_loss"]) == 2

    out = capsys.readouterr().out
    # Batch-level progress
    assert "Batch" in out
    # Epoch-level summary
    assert "Epoch 1/2 | Loss:" in out
    assert "Epoch 2/2 | Loss:" in out


def prediction_fixture():
    rng = np.random.RandomState(42)
    torch.manual_seed(42)
    model = PureGNN(param_dim=3, hidden_dims=[8, 8], num_gnn_layers=1)
    trainer = PureGNNTrainer(model, device="cpu", use_edge_attr=True)
    X = rng.randn(11, 3)
    coords = rng.randn(6, 3)
    T = rng.randn(11, 6) * 5 + 300
    edges = _create_simple_edge_index(6)
    trainer._normalize_data(X, coords, T, fit=True)
    return trainer, X, coords, edges


def test_prediction_batches_preserve_order_and_partial_tail():
    trainer, X, coords, edges = prediction_fixture()
    # A test-set-sized input must not become one accelerator-sized tensor.
    X = np.tile(X, (46, 1))[:500]
    seen = []
    hook = trainer.model.register_forward_pre_hook(lambda _m, args: seen.append(len(args[0])))
    batched = trainer.predict(X, coords, edges)
    hook.remove()
    assert len(seen) == 125 and max(seen) == 4
    singles = trainer.predict(X[:11], coords, edges, batch_size=1)
    partial = trainer.predict(X[:11], coords, edges, batch_size=4)
    np.testing.assert_allclose(partial, singles, atol=1e-4)
    np.testing.assert_allclose(batched[:11], singles, atol=1e-4)
    reversed_pred = trainer.predict(X[10::-1], coords, edges, batch_size=4)
    np.testing.assert_allclose(reversed_pred[::-1], singles, atol=1e-4)


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_prediction_rejects_invalid_batch(batch):
    trainer, X, coords, edges = prediction_fixture()
    with pytest.raises(ValueError, match="positive integer"):
        trainer.predict(X, coords, edges, batch_size=batch)


def test_prediction_empty_and_nonfinite():
    trainer, X, coords, edges = prediction_fixture()
    assert trainer.predict(X[:0], coords, edges).shape == (0, 6)
    with torch.no_grad():
        trainer.model.output_mlp[-1].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.predict(X, coords, edges)


def test_validation_replay_detects_corrupt_scaler_and_loads_local_checkpoint(tmp_path):
    trainer, X, coords, edges = prediction_fixture()
    pred = trainer.predict(X, coords, edges)
    truth = pred.astype(float) + 2.0
    history = {"val_rmse": [4.0, 2.0, 3.0]}
    evidence = trainer.validate_prediction(X, coords, edges, truth, history)
    assert evidence["best_epoch"] == 2
    assert evidence["validation_replay_passed"]
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save(checkpoint)
    trainer.load(checkpoint)
    np.testing.assert_allclose(trainer.predict(X, coords, edges), pred)
    trainer.scaler_T.mean_ += 20.0
    with pytest.raises(RuntimeError, match="disagrees"):
        trainer.validate_prediction(X, coords, edges, truth, history)


def test_fitted_best_model_replays_validation_with_different_batch_size(tmp_path):
    trainer, X, coords, edges = prediction_fixture()
    truth = 300 + X[:, :1] + coords[None, :, 0]
    history = trainer.fit(X[:8], coords, edges, truth[:8], X_val=X[8:], T_val=truth[8:],
                          epochs=3, batch_size=3, verbose=False, save_dir=str(tmp_path))
    evidence = trainer.validate_prediction(X[8:], coords, edges, truth[8:], history, batch_size=2)
    assert evidence["validation_replay_passed"]
    assert evidence["best_epoch"] == int(np.argmin(history["val_loss"])) + 1
