# -*- coding: utf-8 -*-
"""SSTONet models module.

Available models:
- DeepONet: Deep Operator Network
- PODDeepONet: DeepONet with POD basis
- MIONet: Multiple-Input Operator Network
- PODMIONet: MIONet with POD basis
- GNNDeepONet: DeepONet with GNN-enhanced Trunk
- MLP: Pure MLP baseline
- PODMLP: POD + MLP model
"""

from .initializers import INITIALIZERS, get_initializer, init_weights
from .activations import ACTIVATIONS, get_activation
from .deeponet import (
    DeepONet,
    PODDeepONet,
    MIONet,
    PODMIONet,
    GNNDeepONet,
    ComponentAwareGNNDeepONet,
    PureGNN,
    build_mlp,
)
from .dual_graph import DualGraphDeepONet, BranchGraphDeepONet
from .graph_utils import build_component_aware_graph
from .mlp import MLP, MLPTrainer
from .pod_mlp import PODAnalyzer, PODMLP
from .recent_baselines import (
    AdaptedFourierDeepONet,
    AdaptedGeomDeepONet,
    AdaptedTransolver,
    FourierDeepONetLike,
    GeomDeepONetLike,
    TransolverLikeDeepONet,
)

__all__ = [
    # Operator networks
    'DeepONet',
    'PODDeepONet',
    'MIONet',
    'PODMIONet',
    'GNNDeepONet',
    'ComponentAwareGNNDeepONet',
    'DualGraphDeepONet',
    'BranchGraphDeepONet',
    'GeomDeepONetLike',
    'TransolverLikeDeepONet',
    'FourierDeepONetLike',
    'AdaptedGeomDeepONet',
    'AdaptedTransolver',
    'AdaptedFourierDeepONet',
    # Baselines
    'MLP',
    'MLPTrainer',
    'PODMLP',
    'PODAnalyzer',
    'PureGNN',
    # Utilities
    'build_mlp',
    'build_component_aware_graph',
    'init_weights',
    'get_initializer',
    'get_activation',
    'INITIALIZERS',
    'ACTIVATIONS',
]
