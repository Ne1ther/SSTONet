# -*- coding: utf-8 -*-
"""SSTONet: Satellite Steady-state Thermal Operator Network

DeepONet/MIONet surrogate model for 1U CubeSat thermal analysis.

Available models:
- DeepONet: Deep Operator Network
- PODDeepONet: DeepONet with POD basis
- MIONet: Multiple-Input Operator Network
- PODMIONet: MIONet with POD basis
- MLP: Pure MLP baseline
- PODMLP: POD + MLP model
"""

from .config import Config, get_config, set_config, load_config
from .data.loader import DataLoader, train_val_test_split
from .models import (
    DeepONet, PODDeepONet, MIONet, PODMIONet,
    MLP, MLPTrainer, PODAnalyzer, PODMLP,
    build_mlp, init_weights, get_initializer, get_activation,
    INITIALIZERS, ACTIVATIONS
)
from .training import (
    DeepONetTrainer, MIONetTrainer,
    PODDeepONetTrainer, PODMIONetTrainer
)
from .callbacks import EarlyStopping, ModelCheckpoint

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

__all__ = [
    # Config
    'Config', 'get_config', 'set_config', 'load_config',
    # Data
    'DataLoader', 'train_val_test_split',
    # Models - Operator Networks
    'DeepONet', 'PODDeepONet', 'MIONet', 'PODMIONet',
    # Models - Baselines
    'MLP', 'MLPTrainer', 'PODAnalyzer', 'PODMLP',
    # Model utilities
    'build_mlp', 'init_weights', 'get_initializer', 'get_activation',
    'INITIALIZERS', 'ACTIVATIONS',
    # Training
    'DeepONetTrainer', 'MIONetTrainer',
    'PODDeepONetTrainer', 'PODMIONetTrainer',
    # Callbacks
    'EarlyStopping', 'ModelCheckpoint',
]
