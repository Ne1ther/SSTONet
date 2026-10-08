# -*- coding: utf-8 -*-
"""Weight initialization utilities for neural networks.

Provides common weight initialization strategies following deepxde conventions.
"""

import torch.nn as nn
from typing import Callable


INITIALIZERS = {
    'glorot_normal': nn.init.xavier_normal_,
    'glorot_uniform': nn.init.xavier_uniform_,
    'he_normal': nn.init.kaiming_normal_,
    'he_uniform': nn.init.kaiming_uniform_,
    'zeros': nn.init.zeros_,
    'ones': nn.init.ones_,
}


def get_initializer(name: str) -> Callable:
    """Get weight initializer by name.

    Args:
        name: Initializer name ('glorot_normal', 'glorot_uniform',
              'he_normal', 'he_uniform', 'zeros', 'ones')

    Returns:
        Initialization function

    Raises:
        ValueError: If initializer name is unknown
    """
    if name not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {name}. Available: {list(INITIALIZERS.keys())}")
    return INITIALIZERS[name]


def init_weights(module: nn.Module, initializer: str = 'glorot_uniform') -> None:
    """Initialize weights of a module.

    Args:
        module: PyTorch module to initialize
        initializer: Name of initializer
    """
    init_fn = get_initializer(initializer)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
