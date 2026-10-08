# -*- coding: utf-8 -*-
"""SSTONet training module.

Provides unified training interfaces for all models:
- DeepONetTrainer: For DeepONet and PODDeepONet
- MIONetTrainer: For MIONet and PODMIONet
- GNNDeepONetTrainer: For GNNDeepONet
- ComponentAwareGNNTrainer: For ComponentAwareGNNDeepONet
- MLPTrainer: For pure MLP baseline (in models.mlp)
"""

from .trainer import (
    BaseTrainer,
    DeepONetTrainer,
    MIONetTrainer,
    PODDeepONetTrainer,
    PODMIONetTrainer,
)
from .gnn_trainer import GNNDeepONetTrainer
from .component_gnn_trainer import ComponentAwareGNNTrainer

__all__ = [
    'BaseTrainer',
    'DeepONetTrainer',
    'MIONetTrainer',
    'PODDeepONetTrainer',
    'PODMIONetTrainer',
    'GNNDeepONetTrainer',
    'ComponentAwareGNNTrainer',
]
