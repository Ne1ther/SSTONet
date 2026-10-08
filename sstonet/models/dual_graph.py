# -*- coding: utf-8 -*-
"""Dual-graph DeepONet variants.

SSTONet-DualGraph adds a graph encoder on the Branch side for the
radiation field while keeping a graph-based Trunk for the output field.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .deeponet import GCNLayer, GNNTrunk, build_mlp


class RadiationBranchGNNEncoder(nn.Module):
    """Encode a radiation field defined on graph nodes into a global latent."""

    def __init__(
        self,
        radiation_coords: np.ndarray,
        radiation_edge_index: np.ndarray,
        radiation_edge_weight: Optional[np.ndarray] = None,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 256,
        num_gnn_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension")

        self.n_nodes = int(radiation_coords.shape[0])
        coords = np.asarray(radiation_coords, dtype=np.float32)
        coord_mean = coords.mean(axis=0, keepdims=True)
        coord_std = coords.std(axis=0, keepdims=True)
        coord_std = np.where(coord_std < 1e-8, 1.0, coord_std)
        coords_norm = (coords - coord_mean) / coord_std

        self.register_buffer("radiation_coords", torch.as_tensor(coords_norm, dtype=torch.float32))
        self.register_buffer(
            "radiation_edge_index",
            torch.as_tensor(radiation_edge_index, dtype=torch.long),
        )
        if radiation_edge_weight is None:
            self.radiation_edge_weight = None
        else:
            self.register_buffer(
                "radiation_edge_weight",
                torch.as_tensor(radiation_edge_weight, dtype=torch.float32),
            )

        node_hidden_dim = hidden_dims[0]
        self.input_mlp = build_mlp(
            input_dim=4,
            hidden_dims=[],
            output_dim=node_hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=0.0,
            initializer=initializer,
            last_activation=True,
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GCNLayer(node_hidden_dim, node_hidden_dim))
            if use_layer_norm:
                self.gnn_norms.append(nn.LayerNorm(node_hidden_dim))
            else:
                self.gnn_norms.append(nn.Identity())

        self.output_mlp = build_mlp(
            input_dim=node_hidden_dim * 2,
            hidden_dims=hidden_dims[1:],
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }.get(activation, nn.SiLU)()

    def forward(self, radiation_values: torch.Tensor) -> torch.Tensor:
        """Encode the radiation field.

        Args:
            radiation_values: (batch, n_radiation_nodes)
        Returns:
            (batch, output_dim) latent embedding
        """
        if radiation_values.dim() != 2:
            raise ValueError("radiation_values must have shape (batch, n_nodes)")
        if radiation_values.size(1) != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} radiation nodes, got {radiation_values.size(1)}"
            )

        batch_size = radiation_values.size(0)
        coords = self.radiation_coords.unsqueeze(0).expand(batch_size, -1, -1)
        rad = radiation_values.unsqueeze(-1)
        x = torch.cat([rad, coords], dim=-1)

        x = x.reshape(batch_size * self.n_nodes, -1)
        x = self.input_mlp(x)
        x = x.reshape(batch_size, self.n_nodes, -1)

        edge_weight = self.radiation_edge_weight
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = gnn(x, self.radiation_edge_index, edge_weight)
            x_new = x_new.reshape(batch_size * self.n_nodes, -1)
            x_new = norm(x_new)
            x_new = x_new.reshape(batch_size, self.n_nodes, -1)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.output_mlp(pooled)


class DualGraphDeepONet(nn.Module):
    """DeepONet with a graph Branch encoder for radiation and graph Trunk."""

    def __init__(
        self,
        branch_input_dim: int,
        radiation_coords: np.ndarray,
        radiation_edge_index: np.ndarray,
        radiation_edge_weight: Optional[np.ndarray] = None,
        radiation_start_idx: int = 3,
        radiation_dim: int = 1328,
        scalar_branch_hidden: List[int] = [512, 512, 512, 512],
        radiation_hidden: List[int] = [256, 256],
        fusion_hidden: List[int] = [256],
        trunk_input_dim: int = 3,
        trunk_hidden: List[int] = [512, 512, 512, 512],
        output_dim: int = 256,
        num_branch_gnn_layers: int = 2,
        num_trunk_gnn_layers: int = 3,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        if radiation_start_idx < 0 or radiation_dim <= 0:
            raise ValueError("radiation slice must be valid")
        if radiation_start_idx + radiation_dim > branch_input_dim:
            raise ValueError("radiation slice exceeds branch_input_dim")

        self.output_dim = output_dim
        self.radiation_start_idx = radiation_start_idx
        self.radiation_dim = radiation_dim
        self.radiation_end_idx = radiation_start_idx + radiation_dim
        self.scalar_dim = branch_input_dim - radiation_dim

        self.radiation_branch = RadiationBranchGNNEncoder(
            radiation_coords=radiation_coords,
            radiation_edge_index=radiation_edge_index,
            radiation_edge_weight=radiation_edge_weight,
            hidden_dims=radiation_hidden,
            output_dim=output_dim,
            num_gnn_layers=num_branch_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.scalar_branch = build_mlp(
            input_dim=self.scalar_dim,
            hidden_dims=scalar_branch_hidden,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.branch_fusion = build_mlp(
            input_dim=output_dim * 2,
            hidden_dims=fusion_hidden,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.trunk = GNNTrunk(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden,
            output_dim=output_dim,
            num_gnn_layers=num_trunk_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _split_branch_inputs(self, branch_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        radiation = branch_input[:, self.radiation_start_idx:self.radiation_end_idx]
        scalar = torch.cat(
            [
                branch_input[:, : self.radiation_start_idx],
                branch_input[:, self.radiation_end_idx :],
            ],
            dim=-1,
        )
        return radiation, scalar

    def forward(
        self,
        branch_input: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        radiation, scalar = self._split_branch_inputs(branch_input)
        z_rad = self.radiation_branch(radiation)
        z_scalar = self.scalar_branch(scalar)
        z_branch = self.branch_fusion(torch.cat([z_rad, z_scalar], dim=-1))
        z_trunk = self.trunk(coords, edge_index, edge_weight)
        return torch.matmul(z_branch, z_trunk.T) + self.bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BranchGraphDeepONet(nn.Module):
    """DeepONet with a graph Branch encoder for radiation and an MLP Trunk."""

    def __init__(
        self,
        branch_input_dim: int,
        radiation_coords: np.ndarray,
        radiation_edge_index: np.ndarray,
        radiation_edge_weight: Optional[np.ndarray] = None,
        radiation_start_idx: int = 3,
        radiation_dim: int = 1328,
        scalar_branch_hidden: List[int] = [512, 512, 512, 512],
        radiation_hidden: List[int] = [256, 256],
        fusion_hidden: List[int] = [256],
        trunk_input_dim: int = 3,
        trunk_hidden: List[int] = [512, 512, 512, 512],
        output_dim: int = 256,
        num_branch_gnn_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        if radiation_start_idx < 0 or radiation_dim <= 0:
            raise ValueError("radiation slice must be valid")
        if radiation_start_idx + radiation_dim > branch_input_dim:
            raise ValueError("radiation slice exceeds branch_input_dim")

        self.output_dim = output_dim
        self.radiation_start_idx = radiation_start_idx
        self.radiation_dim = radiation_dim
        self.radiation_end_idx = radiation_start_idx + radiation_dim
        self.scalar_dim = branch_input_dim - radiation_dim

        self.radiation_branch = RadiationBranchGNNEncoder(
            radiation_coords=radiation_coords,
            radiation_edge_index=radiation_edge_index,
            radiation_edge_weight=radiation_edge_weight,
            hidden_dims=radiation_hidden,
            output_dim=output_dim,
            num_gnn_layers=num_branch_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.scalar_branch = build_mlp(
            input_dim=self.scalar_dim,
            hidden_dims=scalar_branch_hidden,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.branch_fusion = build_mlp(
            input_dim=output_dim * 2,
            hidden_dims=fusion_hidden,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.trunk = build_mlp(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _split_branch_inputs(self, branch_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        radiation = branch_input[:, self.radiation_start_idx:self.radiation_end_idx]
        scalar = torch.cat(
            [
                branch_input[:, : self.radiation_start_idx],
                branch_input[:, self.radiation_end_idx :],
            ],
            dim=-1,
        )
        return radiation, scalar

    def forward(self, branch_input: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        radiation, scalar = self._split_branch_inputs(branch_input)
        z_rad = self.radiation_branch(radiation)
        z_scalar = self.scalar_branch(scalar)
        z_branch = self.branch_fusion(torch.cat([z_rad, z_scalar], dim=-1))
        z_trunk = self.trunk(coords)
        return torch.matmul(z_branch, z_trunk.T) + self.bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
