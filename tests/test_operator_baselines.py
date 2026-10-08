# -*- coding: utf-8 -*-
"""Forward-shape tests for reusable operator baseline adapters."""

import torch

from sstonet.models import (
    AdaptedFourierDeepONet,
    AdaptedGeomDeepONet,
    AdaptedTransolver,
    FourierDeepONetLike,
    GeomDeepONetLike,
    TransolverLikeDeepONet,
)


def test_geom_deeponet_like_forward():
    model = GeomDeepONetLike(
        branch_input_dim=12,
        trunk_input_dim=8,
        branch_hidden=[16],
        trunk_hidden=[16],
        output_dim=8,
        num_frequencies=2,
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0


def test_fourier_deeponet_like_forward():
    model = FourierDeepONetLike(
        branch_input_dim=12,
        trunk_input_dim=8,
        branch_hidden=[16],
        spectral_hidden=[16],
        output_dim=8,
        num_frequencies=2,
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0


def test_transolver_like_forward():
    model = TransolverLikeDeepONet(
        branch_input_dim=12,
        trunk_input_dim=8,
        branch_hidden=[16],
        hidden_dim=16,
        output_dim=8,
        num_slices=4,
        num_heads=4,
        num_attention_layers=1,
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0


def test_adapted_geom_deeponet_forward():
    model = AdaptedGeomDeepONet(
        branch_input_dim=12,
        trunk_input_dim=8,
        hidden_dim=16,
        output_dim=8,
        branch_pre_hidden=[16],
        branch_post_hidden=[16],
        trunk_pre_hidden=[16],
        trunk_post_hidden=[16],
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0


def test_adapted_transolver_forward():
    model = AdaptedTransolver(
        branch_input_dim=12,
        trunk_input_dim=8,
        branch_hidden=[16],
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        num_slices=4,
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0


def test_adapted_fourier_deeponet_forward():
    model = AdaptedFourierDeepONet(
        branch_input_dim=12,
        trunk_input_dim=8,
        branch_hidden=[16],
        grid_shape=[5, 5, 5],
        width=4,
        modes=[2, 2, 2],
        num_fno_layers=1,
    )
    output = model(torch.randn(3, 12), torch.randn(20, 8))
    assert output.shape == (3, 20)
    assert model.count_parameters() > 0
