# -*- coding: utf-8 -*-
"""Tests for DualGraphDeepONet."""

import numpy as np
import torch

from sstonet.models import DualGraphDeepONet
from sstonet.training import GNNDeepONetTrainer


def _create_simple_edge_index(n_nodes: int) -> np.ndarray:
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    return np.array([src, dst], dtype=np.int64)


def test_dual_graph_forward():
    n_rad = 12
    n_out = 20
    input_dim = 3 + n_rad + 7

    model = DualGraphDeepONet(
        branch_input_dim=input_dim,
        radiation_coords=np.random.randn(n_rad, 3).astype(np.float32),
        radiation_edge_index=_create_simple_edge_index(n_rad),
        radiation_start_idx=3,
        radiation_dim=n_rad,
        scalar_branch_hidden=[16],
        radiation_hidden=[16],
        fusion_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_branch_gnn_layers=1,
        num_trunk_gnn_layers=1,
        activation="silu",
        use_layer_norm=True,
        dropout=0.0,
    )

    params = torch.randn(4, input_dim)
    coords = torch.randn(n_out, 3)
    edge_index = torch.LongTensor(_create_simple_edge_index(n_out))
    out = model(params, coords, edge_index)
    assert out.shape == (4, n_out)


def test_dual_graph_trainer_smoke():
    n_rad = 10
    n_out = 18
    input_dim = 3 + n_rad + 5

    model = DualGraphDeepONet(
        branch_input_dim=input_dim,
        radiation_coords=np.random.randn(n_rad, 3).astype(np.float32),
        radiation_edge_index=_create_simple_edge_index(n_rad),
        radiation_start_idx=3,
        radiation_dim=n_rad,
        scalar_branch_hidden=[16],
        radiation_hidden=[16],
        fusion_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_branch_gnn_layers=1,
        num_trunk_gnn_layers=1,
        activation="silu",
        use_layer_norm=True,
        dropout=0.0,
    )
    trainer = GNNDeepONetTrainer(model, lr=1e-3, device="cpu", use_edge_attr=False)

    X_train = np.random.randn(12, input_dim).astype(np.float32)
    coords = np.random.randn(n_out, 3).astype(np.float32)
    edge_index = _create_simple_edge_index(n_out)
    T_train = np.random.randn(12, n_out).astype(np.float32)

    history = trainer.fit(
        X_train,
        coords,
        edge_index,
        T_train,
        epochs=2,
        batch_size=4,
        verbose=False,
    )
    assert len(history["train_loss"]) == 2

    X_test = np.random.randn(3, input_dim).astype(np.float32)
    pred = trainer.predict(X_test, coords, edge_index)
    assert pred.shape == (3, n_out)
