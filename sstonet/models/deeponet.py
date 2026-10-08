# -*- coding: utf-8 -*-
"""DeepONet, MIONet, PODDeepONet, and PODMIONet models.

References:
    [1] Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
        Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operators. Nature Machine Intelligence, 3(3), 218-229.
        https://doi.org/10.1038/s42256-021-00302-5

    [2] Jin, P., Meng, S., & Lu, L. (2022). MIONet: Learning multiple-input
        operators via tensor product. SIAM Journal on Scientific Computing, 44(6), A3490-A3514.
        https://doi.org/10.1137/22M1477751

    [3] Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., & Karniadakis, G. E. (2022).
        A comprehensive and fair comparison of two neural operators (with practical extensions)
        based on FAIR data. Computer Methods in Applied Mechanics and Engineering, 393, 114778.
        https://arxiv.org/abs/2111.05512
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Union

from .initializers import INITIALIZERS, get_initializer, init_weights
from .activations import ACTIVATIONS, get_activation

# =============================================================================
# MLP Builder
# =============================================================================

def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = 'relu',
    use_layer_norm: bool = False,
    dropout: float = 0.0,
    initializer: str = 'glorot_uniform',
    last_activation: bool = False
) -> nn.Sequential:
    """Build a Multi-Layer Perceptron.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function name
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate (0 to disable)
        initializer: Weight initializer name
        last_activation: Whether to apply activation after the last layer

    Returns:
        nn.Sequential: MLP network
    """
    layers = []
    prev_dim = input_dim
    act_class = ACTIVATIONS[activation]

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_class())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    if last_activation:
        layers.append(act_class())

    net = nn.Sequential(*layers)
    init_weights(net, initializer)
    return net


# =============================================================================
# DeepONet
# =============================================================================

class DeepONet(nn.Module):
    """Deep Operator Network (DeepONet).

    DeepONet learns nonlinear operators by decomposing them into a branch network
    (encoding input functions) and a trunk network (encoding output locations).

    Output = sum_i(Branch_i(u) * Trunk_i(y)) + bias

    Args:
        branch_input_dim: Input dimension for branch network
        trunk_input_dim: Input dimension for trunk network (default: 3 for 3D coordinates)
        branch_hidden: Hidden layer dimensions for branch network
        trunk_hidden: Hidden layer dimensions for trunk network
        output_dim: Output dimension (number of basis functions)
        activation: Activation function name
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate
        initializer: Weight initializer name

    References:
        Lu et al. Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operators. Nat Mach Intell, 2021.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 3,
        branch_hidden: List[int] = [256, 256, 256],
        trunk_hidden: List[int] = [256, 256, 256],
        output_dim: int = 128,
        activation: str = 'relu',
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform'
    ):
        super().__init__()

        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim

        self.branch = build_mlp(
            branch_input_dim, branch_hidden, output_dim,
            activation, use_layer_norm, dropout, initializer
        )

        self.trunk = build_mlp(
            trunk_input_dim, trunk_hidden, output_dim,
            activation, use_layer_norm, dropout, initializer,
            last_activation=True  # Trunk typically has activation on last layer
        )

        self.bias = nn.Parameter(torch.zeros(1))

        # Optional transforms
        self._input_transform = None
        self._output_transform = None

    def set_input_transform(self, transform: Callable) -> None:
        """Set input transform for trunk network."""
        self._input_transform = transform

    def set_output_transform(self, transform: Callable) -> None:
        """Set output transform."""
        self._output_transform = transform

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            params: Input parameters (batch, branch_input_dim)
            coords: Spatial coordinates (n_points, trunk_input_dim)

        Returns:
            Output field (batch, n_points)
        """
        # Apply input transform to coordinates
        if self._input_transform is not None:
            coords = self._input_transform(coords)

        branch_out = self.branch(params)  # (batch, output_dim)
        trunk_out = self.trunk(coords)    # (n_points, output_dim)

        # Dot product: (batch, output_dim) x (n_points, output_dim)^T -> (batch, n_points)
        output = torch.einsum("bi,ni->bn", branch_out, trunk_out) + self.bias

        # Apply output transform
        if self._output_transform is not None:
            output = self._output_transform((params, coords), output)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PODDeepONet
# =============================================================================

class PODDeepONet(nn.Module):
    """DeepONet with Proper Orthogonal Decomposition (POD) basis.

    Uses pre-computed POD basis vectors as the trunk network output, optionally
    combined with a learnable trunk network for residual correction.

    Output = sum_i(Branch_i(u) * POD_basis_i) + pod_mean + [optional: Trunk correction]

    Args:
        pod_basis: POD basis matrix (n_points, n_modes)
        pod_mean: POD mean vector (n_points,), required for correct reconstruction
        branch_input_dim: Input dimension for branch network
        branch_hidden: Hidden layer dimensions for branch network
        activation: Activation function name
        trunk_hidden: Optional trunk network for residual (None to use POD only)
        trunk_input_dim: Input dimension for trunk network
        initializer: Weight initializer name

    References:
        Lu et al. A comprehensive and fair comparison of two neural operators
        (with practical extensions) based on FAIR data. arXiv:2111.05512, 2021.
    """

    def __init__(
        self,
        pod_basis: np.ndarray,
        branch_input_dim: int,
        pod_mean: Optional[np.ndarray] = None,
        branch_hidden: List[int] = [256, 256, 256],
        activation: str = 'relu',
        trunk_hidden: Optional[List[int]] = None,
        trunk_input_dim: int = 3,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform'
    ):
        super().__init__()

        self.n_modes = pod_basis.shape[1]
        self.register_buffer('pod_basis', torch.as_tensor(pod_basis, dtype=torch.float32))
        if pod_mean is not None:
            self.register_buffer('pod_mean', torch.as_tensor(pod_mean, dtype=torch.float32))
        else:
            self.pod_mean = None

        # Branch network outputs POD coefficients
        self.branch = build_mlp(
            branch_input_dim, branch_hidden, self.n_modes,
            activation, use_layer_norm, dropout, initializer
        )

        # Optional trunk network for residual correction
        self.trunk = None
        if trunk_hidden is not None:
            self.trunk = build_mlp(
                trunk_input_dim, trunk_hidden, self.n_modes,
                activation, use_layer_norm, dropout, initializer,
                last_activation=True
            )
            self.bias = nn.Parameter(torch.zeros(1))

        self._input_transform = None
        self._output_transform = None

    def forward(self, params: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            params: Input parameters (batch, branch_input_dim)
            coords: Spatial coordinates (n_points, trunk_input_dim), required if trunk is used

        Returns:
            Output field (batch, n_points)
        """
        # Branch outputs POD coefficients
        coeffs = self.branch(params)  # (batch, n_modes)

        if self.trunk is None:
            # POD only: reconstruct using POD basis
            output = torch.einsum("bi,ni->bn", coeffs, self.pod_basis)
        else:
            # POD + trunk correction
            if coords is None:
                raise ValueError("coords required when trunk network is used")
            if self._input_transform is not None:
                coords = self._input_transform(coords)
            trunk_out = self.trunk(coords)  # (n_points, n_modes)
            combined_basis = torch.cat([self.pod_basis, trunk_out], dim=1)
            # Need to expand coeffs to match combined basis
            coeffs_expanded = torch.cat([coeffs, coeffs], dim=1)
            output = torch.einsum("bi,ni->bn", coeffs_expanded, combined_basis) + self.bias

        # Add POD mean for correct reconstruction
        if self.pod_mean is not None:
            output = output + self.pod_mean

        if self._output_transform is not None:
            output = self._output_transform((params, coords), output)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# MIONet
# =============================================================================

class MIONet(nn.Module):
    """Multiple-Input Operator Network (MIONet).

    Extends DeepONet to handle multiple input functions through separate branch
    networks, which are then merged via element-wise product, sum, or concatenation.

    Args:
        input_dims: Dictionary mapping input names to their dimensions
        trunk_input_dim: Input dimension for trunk network
        branch_hidden: Hidden layer dimensions for each branch network
        trunk_hidden: Hidden layer dimensions for trunk network
        output_dim: Output dimension (number of basis functions)
        merge: Merge strategy ('product', 'sum', 'concat')
        activation: Activation function name
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate
        initializer: Weight initializer name

    References:
        Jin et al. MIONet: Learning multiple-input operators via tensor product.
        SIAM J. Sci. Comput., 2022.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        trunk_input_dim: int = 3,
        branch_hidden: List[int] = [256, 256],
        trunk_hidden: List[int] = [256, 256, 256],
        output_dim: int = 128,
        merge: str = 'product',
        activation: str = 'relu',
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform'
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.merge = merge

        # Create branch networks for each input
        self.branches = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.branches[name] = build_mlp(
                dim, branch_hidden, output_dim,
                activation, use_layer_norm, dropout, initializer
            )

        # Trunk network
        self.trunk = build_mlp(
            trunk_input_dim, trunk_hidden, output_dim,
            activation, use_layer_norm, dropout, initializer,
            last_activation=True
        )

        # Merge layer for concatenation
        if merge == 'concat':
            self.merge_layer = nn.Linear(output_dim * len(input_dims), output_dim)
            init_weights(self.merge_layer, initializer)
        else:
            self.merge_layer = None

        self.bias = nn.Parameter(torch.zeros(1))

        self._input_transform = None
        self._output_transform = None

    def forward(self, inputs: Dict[str, torch.Tensor], coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Dictionary of input tensors (batch, input_dim) for each branch
            coords: Spatial coordinates (n_points, trunk_input_dim)

        Returns:
            Output field (batch, n_points)
        """
        if self._input_transform is not None:
            coords = self._input_transform(coords)

        # Process each branch
        branch_outputs = []
        for name in self.input_dims.keys():
            out = self.branches[name](inputs[name])
            branch_outputs.append(out)

        # Merge branch outputs
        if self.merge == 'product':
            merged = branch_outputs[0]
            for out in branch_outputs[1:]:
                merged = merged * out
        elif self.merge == 'sum':
            merged = sum(branch_outputs)
        elif self.merge == 'concat':
            merged = torch.cat(branch_outputs, dim=-1)
            merged = self.merge_layer(merged)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge}")

        # Trunk network
        trunk_out = self.trunk(coords)

        # Dot product
        output = torch.einsum("bi,ni->bn", merged, trunk_out) + self.bias

        if self._output_transform is not None:
            output = self._output_transform((inputs, coords), output)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PODMIONet
# =============================================================================

class PODMIONet(nn.Module):
    """MIONet with POD basis.

    Combines multiple-input operator learning with POD-based output representation.

    Args:
        pod_basis: POD basis matrix (n_points, n_modes)
        input_dims: Dictionary mapping input names to their dimensions
        pod_mean: POD mean vector (n_points,), required for correct reconstruction
        branch_hidden: Hidden layer dimensions for each branch network
        merge: Merge strategy ('product', 'sum', 'concat')
        activation: Activation function name
        trunk_hidden: Optional trunk network for residual correction
        trunk_input_dim: Input dimension for trunk network
        initializer: Weight initializer name

    References:
        Jin et al. MIONet: Learning multiple-input operators via tensor product.
        SIAM J. Sci. Comput., 2022.
    """

    def __init__(
        self,
        pod_basis: np.ndarray,
        input_dims: Dict[str, int],
        pod_mean: Optional[np.ndarray] = None,
        branch_hidden: List[int] = [256, 256],
        merge: str = 'product',
        activation: str = 'relu',
        trunk_hidden: Optional[List[int]] = None,
        trunk_input_dim: int = 3,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform'
    ):
        super().__init__()

        self.input_dims = input_dims
        self.n_modes = pod_basis.shape[1]
        self.merge = merge
        self.register_buffer('pod_basis', torch.as_tensor(pod_basis, dtype=torch.float32))
        if pod_mean is not None:
            self.register_buffer('pod_mean', torch.as_tensor(pod_mean, dtype=torch.float32))
        else:
            self.pod_mean = None

        # Create branch networks for each input
        self.branches = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.branches[name] = build_mlp(
                dim, branch_hidden, self.n_modes,
                activation, use_layer_norm, dropout, initializer
            )

        # Merge layer for concatenation
        if merge == 'concat':
            self.merge_layer = nn.Linear(self.n_modes * len(input_dims), self.n_modes)
            init_weights(self.merge_layer, initializer)
        else:
            self.merge_layer = None

        # Optional trunk network
        self.trunk = None
        if trunk_hidden is not None:
            self.trunk = build_mlp(
                trunk_input_dim, trunk_hidden, self.n_modes,
                activation, use_layer_norm, dropout, initializer,
                last_activation=True
            )
            self.bias = nn.Parameter(torch.zeros(1))

        self._input_transform = None
        self._output_transform = None

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Dictionary of input tensors for each branch
            coords: Spatial coordinates (required if trunk is used)

        Returns:
            Output field (batch, n_points)
        """
        # Process each branch
        branch_outputs = []
        for name in self.input_dims.keys():
            out = self.branches[name](inputs[name])
            branch_outputs.append(out)

        # Merge branch outputs
        if self.merge == 'product':
            merged = branch_outputs[0]
            for out in branch_outputs[1:]:
                merged = merged * out
        elif self.merge == 'sum':
            merged = sum(branch_outputs)
        elif self.merge == 'concat':
            merged = torch.cat(branch_outputs, dim=-1)
            merged = self.merge_layer(merged)

        # Reconstruct using POD basis
        if self.trunk is None:
            output = torch.einsum("bi,ni->bn", merged, self.pod_basis)
        else:
            if coords is None:
                raise ValueError("coords required when trunk network is used")
            if self._input_transform is not None:
                coords = self._input_transform(coords)
            trunk_out = self.trunk(coords)
            combined_basis = torch.cat([self.pod_basis, trunk_out], dim=1)
            merged_expanded = torch.cat([merged, merged], dim=1)
            output = torch.einsum("bi,ni->bn", merged_expanded, combined_basis) + self.bias

        # Add POD mean for correct reconstruction
        if self.pod_mean is not None:
            output = output + self.pod_mean

        if self._output_transform is not None:
            output = self._output_transform((inputs, coords), output)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# GNN-Enhanced DeepONet
# =============================================================================

class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer (pure PyTorch).

    Supports both 2D input (N, in_dim) and 3D batched input (batch, N, in_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        # Cache for normalized edge weights. We avoid `.cpu()` round-trips because
        # that is extremely slow on MPS (sync + device transfer).
        self._cached_norm: Optional[torch.Tensor] = None
        self._cached_norm_key = None

    @staticmethod
    def _make_norm_cache_key(
        N: int,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ):
        """Build a cheap cache key without comparing tensor values."""
        # NOTE: Rely on storage pointers + metadata. This is sufficient for the
        # common case where the same graph tensor is reused across batches.
        edge_ptr = edge_index.untyped_storage().data_ptr() if edge_index.numel() else 0
        weight_ptr = (
            edge_weight.untyped_storage().data_ptr()
            if (edge_weight is not None and edge_weight.numel())
            else 0
        )
        return (
            int(N),
            str(edge_index.device),
            edge_index.dtype,
            tuple(edge_index.shape),
            int(edge_ptr),
            int(weight_ptr),
        )

    def _get_norm(self, N: int, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Get or create cached normalization coefficients on the current device."""
        key = self._make_norm_cache_key(N, edge_index, edge_weight)
        if self._cached_norm is not None and self._cached_norm_key == key:
            return self._cached_norm

        src, dst = edge_index[0], edge_index[1]

        # Compute node degrees for normalization.
        deg = torch.zeros(N, device=edge_index.device, dtype=torch.float32)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Compute normalization coefficients.
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        if edge_weight is not None:
            norm = norm * edge_weight.to(dtype=norm.dtype)

        # Cache (on-device, no copies)
        self._cached_norm = norm.detach()
        self._cached_norm_key = key

        return norm

    @staticmethod
    def _choose_edge_chunk_size(
        *,
        n_edges: int,
        batch_size: int,
        feat_dim: int,
        elem_size: int,
        target_bytes: int = 128 * 1024 * 1024,  # ~128MB
        min_chunk: int = 1024,
    ) -> int:
        """Choose a chunk size to bound intermediate (batch, chunk, feat) tensors."""
        if n_edges <= 0:
            return 0
        denom = max(1, batch_size * feat_dim * elem_size)
        chunk = target_bytes // denom
        # Round down to a multiple of 1024 for more stable performance.
        chunk = int((chunk // 1024) * 1024)
        chunk = max(min_chunk, chunk)
        return int(min(n_edges, chunk))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_dim) or (batch, N, in_dim)
            edge_index: Edge indices (2, E), [src, dst]
            edge_weight: Optional edge weights (E,)
        Returns:
            Updated node features (N, out_dim) or (batch, N, out_dim)
        """
        is_batched = x.dim() == 3
        if is_batched:
            batch_size, N, _ = x.shape
        else:
            N = x.size(0)

        # Retrieve normalization coefficients.
        norm = self._get_norm(N, edge_index, edge_weight)

        # Apply the linear transformation first.
        x_transformed = self.linear(x)

        # Ensure norm can be applied in-place without dtype upcasting errors.
        norm = norm.to(dtype=x_transformed.dtype)
        src, dst = edge_index[0], edge_index[1]
        n_edges = int(src.numel())
        if n_edges == 0:
            return torch.zeros_like(x_transformed)

        # Chunk edges to avoid allocating a massive (batch, E, out_dim) tensor.
        feat_dim = int(x_transformed.size(-1))
        elem_size = int(x_transformed.element_size())
        chunk_size = self._choose_edge_chunk_size(
            n_edges=n_edges,
            batch_size=int(batch_size) if is_batched else 1,
            feat_dim=feat_dim,
            elem_size=elem_size,
        )

        out = torch.zeros_like(x_transformed)
        if is_batched:
            # Batched: out.index_add_(dim=1) supports (batch, chunk, feat) src.
            for start in range(0, n_edges, chunk_size):
                end = min(n_edges, start + chunk_size)
                src_c = src[start:end]
                dst_c = dst[start:end]
                msg = x_transformed.index_select(1, src_c)  # (B, C, F)
                msg.mul_(norm[start:end].view(1, -1, 1))
                out.index_add_(1, dst_c, msg)
        else:
            for start in range(0, n_edges, chunk_size):
                end = min(n_edges, start + chunk_size)
                src_c = src[start:end]
                dst_c = dst[start:end]
                msg = x_transformed.index_select(0, src_c)  # (C, F)
                msg.mul_(norm[start:end].unsqueeze(1))
                out.index_add_(0, dst_c, msg)

        return out


class GNNTrunk(nn.Module):
    """GNN-based Trunk network for DeepONet."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256],
        output_dim: int = 128,
        num_gnn_layers: int = 2,
        activation: str = 'silu',
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers

        # Configure the activation function.
        activations = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }
        act_fn = activations.get(activation, nn.SiLU)

        # Initial MLP: coordinates to hidden features.
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            act_fn(),
        )

        # Graph message-passing layers.
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GCNLayer(hidden_dims[0], hidden_dims[0]))
            if use_layer_norm:
                self.gnn_norms.append(nn.LayerNorm(hidden_dims[0]))
            else:
                self.gnn_norms.append(nn.Identity())

        # MLP after the graph layers.
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.output_mlp = nn.Sequential(*layers)

        self.activation = act_fn()

    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coords: Node coordinates (N, 3)
            edge_index: Graph edges (2, E)
            edge_weight: Optional edge weights (E,)
        Returns:
            Trunk output (N, output_dim)
        """
        # Compute the initial node embeddings.
        x = self.input_mlp(coords)

        # Apply graph message passing.
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = gnn(x, edge_index, edge_weight)
            x_new = norm(x_new)
            x_new = self.activation(x_new)
            x = x + x_new  # Residual connection.

        # Apply the output MLP.
        out = self.output_mlp(x)
        return out


class GNNDeepONet(nn.Module):
    """DeepONet with GNN-enhanced Trunk."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 3,
        branch_hidden: List[int] = [512, 512, 512, 512],
        trunk_hidden: List[int] = [512, 512, 512, 512],
        output_dim: int = 256,
        num_gnn_layers: int = 2,
        activation: str = 'silu',
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform',
    ):
        super().__init__()
        self.output_dim = output_dim

        # Configure the activation function.
        activations = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }
        act_fn = activations.get(activation, nn.SiLU)

        # Branch network (MLP).
        branch_layers = []
        in_dim = branch_input_dim
        for h_dim in branch_hidden:
            branch_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                branch_layers.append(nn.LayerNorm(h_dim))
            branch_layers.append(act_fn())
            if dropout > 0:
                branch_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        branch_layers.append(nn.Linear(in_dim, output_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network (GNN).
        self.trunk = GNNTrunk(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # Output bias.
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize the network.
        self._init_weights(initializer)

    def _init_weights(self, initializer: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == 'glorot_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif initializer == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        branch_input: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            branch_input: (batch_size, branch_input_dim)
            coords: (N, 3) node coordinates
            edge_index: (2, E) graph edges
            edge_weight: (E,) optional edge weights
        Returns:
            predictions: (batch_size, N)
        """
        # Branch: (batch, output_dim)
        branch_out = self.branch(branch_input)

        # Trunk with GNN: (N, output_dim)
        trunk_out = self.trunk(coords, edge_index, edge_weight)

        # Inner product: (batch, N).
        output = torch.matmul(branch_out, trunk_out.T) + self.bias

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Component-Aware GNN DeepONet (Resolution Generalization)
# =============================================================================

# =============================================================================
# Pure GNN Baseline (MeshGraphNets-style)
# =============================================================================

class PureGNN(nn.Module):
    """Pure GNN baseline for comparison with GNN-DeepONet.

    Unlike DeepONet which separates parameter encoding (Branch) and spatial
    encoding (Trunk), PureGNN concatenates parameters to each node and processes
    everything through GNN layers.

    Architecture:
        1. Node features: coords (N, 3) concat params broadcast -> (N, 3+param_dim)
        2. GNN layers with residual connections -> (N, hidden)
        3. Output MLP -> (N, 1)

    Note: This requires separate GNN forward pass for each batch sample,
    making it slower than DeepONet but providing a fair baseline comparison.
    """

    def __init__(
        self,
        param_dim: int,
        coord_dim: int = 3,
        hidden_dims: List[int] = [512, 512, 512, 512],
        num_gnn_layers: int = 2,
        activation: str = 'silu',
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform',
    ):
        super().__init__()
        self.param_dim = param_dim
        self.coord_dim = coord_dim
        self.num_gnn_layers = num_gnn_layers

        # Activation function
        activations = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
        act_fn = activations.get(activation, nn.SiLU)

        # Input MLP: (coords + params) -> hidden
        input_dim = coord_dim + param_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            act_fn(),
        )

        # GNN layers with residual connections
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GCNLayer(hidden_dims[0], hidden_dims[0]))
            if use_layer_norm:
                self.gnn_norms.append(nn.LayerNorm(hidden_dims[0]))
            else:
                self.gnn_norms.append(nn.Identity())

        # Output MLP: hidden -> ... -> 1
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.output_mlp = nn.Sequential(*layers)

        self.activation = act_fn()
        self._init_weights(initializer)

    def _init_weights(self, initializer: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == 'glorot_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif initializer == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single sample.

        Args:
            params: (param_dim,) single sample parameters
            coords: (N, coord_dim) node coordinates
            edge_index: (2, E) graph edges
            edge_weight: (E,) optional edge weights

        Returns:
            predictions: (N,) temperature at each node
        """
        N = coords.size(0)

        # Broadcast params to all nodes: (N, param_dim)
        params_broadcast = params.unsqueeze(0).expand(N, -1)

        # Concatenate: (N, coord_dim + param_dim)
        x = torch.cat([coords, params_broadcast], dim=-1)

        # Input MLP
        x = self.input_mlp(x)

        # GNN message passing with residual connections
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = gnn(x, edge_index, edge_weight)
            x_new = norm(x_new)
            x_new = self.activation(x_new)
            x = x + x_new  # Residual connection

        # Output MLP: (N, 1) -> (N,)
        out = self.output_mlp(x).squeeze(-1)
        return out

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for batch of samples (vectorized).

        Args:
            params: (batch_size, param_dim) batch of parameters
            coords: (N, coord_dim) node coordinates (shared across batch)
            edge_index: (2, E) graph edges (shared across batch)
            edge_weight: (E,) optional edge weights

        Returns:
            predictions: (batch_size, N) temperature field for each sample
        """
        batch_size = params.size(0)
        N = coords.size(0)

        # Broadcast params to all nodes: (batch_size, N, param_dim)
        params_broadcast = params.unsqueeze(1).expand(-1, N, -1)

        # Broadcast coords to all batches: (batch_size, N, coord_dim)
        coords_broadcast = coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate: (batch_size, N, coord_dim + param_dim)
        x = torch.cat([coords_broadcast, params_broadcast], dim=-1)

        # Input MLP: (batch_size, N, hidden)
        # Reshape -> MLP -> Reshape back
        x = x.view(batch_size * N, -1)
        x = self.input_mlp(x)
        x = x.view(batch_size, N, -1)

        # GNN message passing with residual connections (vectorized over batch)
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = gnn(x, edge_index, edge_weight)  # (batch, N, hidden)
            # LayerNorm needs (batch*N, hidden) shape
            x_new = x_new.view(batch_size * N, -1)
            x_new = norm(x_new)
            x_new = x_new.view(batch_size, N, -1)
            x_new = self.activation(x_new)
            x = x + x_new  # Residual connection

        # Output MLP: (batch_size, N, 1) -> (batch_size, N)
        x = x.view(batch_size * N, -1)
        out = self.output_mlp(x)
        out = out.view(batch_size, N)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComponentAwareGNNDeepONet(nn.Module):
    """GNN DeepONet with dynamic component-aware graph construction.

    Supports resolution generalization by building graphs dynamically from
    coordinates and component IDs rather than fixed FEM topology.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 3,
        branch_hidden: List[int] = [512, 512, 512, 512],
        trunk_hidden: List[int] = [512, 512, 512, 512],
        output_dim: int = 256,
        num_gnn_layers: int = 2,
        k: int = 15,
        cross_component_radius: float = 2.0,
        activation: str = 'silu',
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = 'glorot_uniform',
        use_edge_attr: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.k = k
        self.cross_component_radius = cross_component_radius
        self.use_edge_attr = use_edge_attr

        activations = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
        act_fn = activations.get(activation, nn.SiLU)

        # Branch network (MLP)
        branch_layers = []
        in_dim = branch_input_dim
        for h_dim in branch_hidden:
            branch_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                branch_layers.append(nn.LayerNorm(h_dim))
            branch_layers.append(act_fn())
            if dropout > 0:
                branch_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        branch_layers.append(nn.Linear(in_dim, output_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network (GNN)
        self.trunk = GNNTrunk(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights(initializer)

    def _init_weights(self, initializer: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == 'glorot_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif initializer == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_graph(self, coords: np.ndarray, comp_ids: np.ndarray):
        """Build graph from raw (unnormalized) coordinates.

        Args:
            coords: Raw coordinates (N, 3) - NOT normalized
            comp_ids: Component IDs (N,)

        Returns:
            edge_index: (2, E)
            edge_attr: (E,) edge distances, only if use_edge_attr=True
        """
        from .graph_utils import build_component_aware_graph
        return build_component_aware_graph(
            coords, comp_ids, self.k, self.cross_component_radius,
            return_edge_attr=self.use_edge_attr
        )

    def forward(
        self,
        branch_input: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-built graph.

        Args:
            branch_input: (batch_size, branch_input_dim)
            coords: (N, 3) node coordinates (can be normalized)
            edge_index: (2, E) pre-built graph edges
            edge_weight: (E,) optional edge weights

        Returns:
            predictions: (batch_size, N)
        """
        # Branch: (batch, output_dim)
        branch_out = self.branch(branch_input)

        # Trunk with GNN: (N, output_dim)
        trunk_out = self.trunk(coords, edge_index, edge_weight)

        # Inner product: (batch, N)
        output = torch.matmul(branch_out, trunk_out.T) + self.bias
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
