# -*- coding: utf-8 -*-
"""Tests for model creation and forward propagation."""

import pytest
import torch
import numpy as np
from sstonet.models import DeepONet, PODDeepONet, MIONet, PODMIONet


def test_deeponet_creation():
    """Test DeepONet model creation."""
    model = DeepONet(
        branch_input_dim=100,
        trunk_input_dim=3,
        branch_hidden=[64, 64],
        trunk_hidden=[64, 64],
        output_dim=32
    )
    assert model.branch_input_dim == 100
    assert model.trunk_input_dim == 3
    assert model.output_dim == 32
    assert model.count_parameters() > 0


def test_deeponet_forward():
    """Test DeepONet forward pass."""
    model = DeepONet(
        branch_input_dim=10,
        trunk_input_dim=3,
        branch_hidden=[32],
        trunk_hidden=[32],
        output_dim=16
    )

    params = torch.randn(4, 10)  # batch=4
    coords = torch.randn(50, 3)  # 50 points

    output = model(params, coords)
    assert output.shape == (4, 50)


def test_poddeeponet_creation():
    """Test PODDeepONet model creation."""
    pod_basis = np.random.randn(100, 10)  # 100 points, 10 modes
    model = PODDeepONet(
        pod_basis=pod_basis,
        branch_input_dim=50,
        branch_hidden=[64, 64]
    )
    assert model.n_modes == 10
    assert model.count_parameters() > 0


def test_poddeeponet_forward():
    """Test PODDeepONet forward pass."""
    pod_basis = np.random.randn(100, 10)
    model = PODDeepONet(
        pod_basis=pod_basis,
        branch_input_dim=20,
        branch_hidden=[32]
    )

    params = torch.randn(4, 20)
    output = model(params)
    assert output.shape == (4, 100)


def test_mionet_creation():
    """Test MIONet model creation."""
    input_dims = {'input1': 10, 'input2': 20}
    model = MIONet(
        input_dims=input_dims,
        trunk_input_dim=3,
        branch_hidden=[32],
        trunk_hidden=[32],
        output_dim=16
    )
    assert model.input_dims == input_dims
    assert model.output_dim == 16


def test_mionet_forward():
    """Test MIONet forward pass."""
    input_dims = {'input1': 10, 'input2': 20}
    model = MIONet(
        input_dims=input_dims,
        trunk_input_dim=3,
        branch_hidden=[32],
        trunk_hidden=[32],
        output_dim=16
    )

    inputs = {
        'input1': torch.randn(4, 10),
        'input2': torch.randn(4, 20)
    }
    coords = torch.randn(50, 3)

    output = model(inputs, coords)
    assert output.shape == (4, 50)


def test_podmionet_creation():
    """Test PODMIONet model creation."""
    pod_basis = np.random.randn(100, 10)
    input_dims = {'input1': 10, 'input2': 20}
    model = PODMIONet(
        pod_basis=pod_basis,
        input_dims=input_dims,
        branch_hidden=[32]
    )
    assert model.n_modes == 10
    assert model.input_dims == input_dims


def test_podmionet_forward():
    """Test PODMIONet forward pass."""
    pod_basis = np.random.randn(100, 10)
    input_dims = {'input1': 10, 'input2': 20}
    model = PODMIONet(
        pod_basis=pod_basis,
        input_dims=input_dims,
        branch_hidden=[32]
    )

    inputs = {
        'input1': torch.randn(4, 10),
        'input2': torch.randn(4, 20)
    }

    output = model(inputs)
    assert output.shape == (4, 100)
