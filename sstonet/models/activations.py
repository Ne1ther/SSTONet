# -*- coding: utf-8 -*-
"""Activation functions for neural networks.

Provides common activation functions following deepxde conventions.
"""

import torch.nn as nn


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
    'softplus': nn.Softplus,
}


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation function name ('relu', 'tanh', 'gelu', 'silu',
              'elu', 'leaky_relu', 'softplus')

    Returns:
        Activation module instance

    Raises:
        ValueError: If activation name is unknown
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()
