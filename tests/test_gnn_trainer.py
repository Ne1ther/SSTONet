# -*- coding: utf-8 -*-
"""Tests for GNNDeepONetTrainer."""

import pytest
import numpy as np
from sstonet.models import GNNDeepONet
from sstonet.training import GNNDeepONetTrainer


def _create_simple_edge_index(n_nodes: int) -> np.ndarray:
    """Create a simple chain graph edge index."""
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    return np.array([src, dst])


def test_gnn_trainer_creation():
    """Test GNNDeepONetTrainer creation."""
    model = GNNDeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
    )
    trainer = GNNDeepONetTrainer(model, lr=1e-3, device='cpu')
    assert trainer.model is not None
    assert trainer.optimizer is not None


def test_gnn_trainer_fit():
    """Test GNNDeepONetTrainer fit method."""
    n_points = 20
    model = GNNDeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_gnn_layers=1,
    )
    trainer = GNNDeepONetTrainer(model, lr=1e-3, device='cpu')

    X_train = np.random.randn(15, 10)
    coords = np.random.randn(n_points, 3)
    edge_index = _create_simple_edge_index(n_points)
    T_train = np.random.randn(15, n_points)

    history = trainer.fit(
        X_train, coords, edge_index, T_train,
        epochs=3,
        batch_size=4,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 3


def test_gnn_trainer_predict():
    """Test GNNDeepONetTrainer predict method."""
    n_points = 20
    model = GNNDeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_gnn_layers=1,
    )
    trainer = GNNDeepONetTrainer(model, lr=1e-3, device='cpu')

    X_train = np.random.randn(15, 10)
    coords = np.random.randn(n_points, 3)
    edge_index = _create_simple_edge_index(n_points)
    T_train = np.random.randn(15, n_points)

    trainer.fit(X_train, coords, edge_index, T_train, epochs=2, verbose=False)

    X_test = np.random.randn(5, 10)
    T_pred = trainer.predict(X_test, coords, edge_index)

    assert T_pred.shape == (5, n_points)
