# -*- coding: utf-8 -*-
"""Tests for GNNDeepONet model."""

import pytest
import torch
import numpy as np
from sstonet.models import GNNDeepONet


def _create_simple_edge_index(n_nodes: int) -> np.ndarray:
    """Create a simple chain graph edge index."""
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    return np.array([src, dst])


def test_gnn_deeponet_creation():
    """Test GNNDeepONet model creation."""
    model = GNNDeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[32, 32],
        trunk_hidden=[32, 32],
        output_dim=16,
        num_gnn_layers=2,
    )
    assert model.output_dim == 16
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gnn_deeponet_forward():
    """Test GNNDeepONet forward pass."""
    model = GNNDeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_gnn_layers=1,
    )

    batch_size = 4
    n_points = 20

    params = torch.randn(batch_size, 10)
    coords = torch.randn(n_points, 3)
    edge_index = torch.LongTensor(_create_simple_edge_index(n_points))

    output = model(params, coords, edge_index)
    assert output.shape == (batch_size, n_points)


def test_gnn_deeponet_different_activations():
    """Test GNNDeepONet with different activations."""
    for activation in ['relu', 'silu', 'gelu', 'tanh']:
        model = GNNDeepONet(
            branch_input_dim=5,
            trunk_input_dim=3,
            branch_hidden=[16],
            trunk_hidden=[16],
            output_dim=8,
            activation=activation,
        )
        params = torch.randn(2, 5)
        coords = torch.randn(10, 3)
        edge_index = torch.LongTensor(_create_simple_edge_index(10))
        output = model(params, coords, edge_index)
        assert output.shape == (2, 10)
