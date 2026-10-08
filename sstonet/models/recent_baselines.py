# -*- coding: utf-8 -*-
"""Neural-operator baseline variants for the fixed-node SSTONet benchmark.

These models are scoped adaptations for the fixed-node SSTONet benchmark. They
are intentionally named as "-like" baselines rather than faithful reproductions
of the cited papers, because the spacecraft benchmark has a fixed irregular FEM
node set and a 1422-dimensional operating-condition input.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .activations import ACTIVATIONS
from .deeponet import build_mlp

warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized since it had shape \[\].*",
    category=UserWarning,
)


def _count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _activation(name: str) -> nn.Module:
    return ACTIVATIONS.get(name, nn.SiLU)()


class FourierFeatureEncoder(nn.Module):
    """Encode arbitrary node features with axis-wise sinusoidal frequencies."""

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 6,
        max_frequency: float = 16.0,
        include_input: bool = True,
    ):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        if max_frequency <= 0:
            raise ValueError("max_frequency must be > 0")

        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        powers = torch.linspace(0.0, math.log2(float(max_frequency)), num_frequencies)
        self.register_buffer("frequencies", torch.pow(2.0, powers), persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = math.pi * x.unsqueeze(-1) * self.frequencies.to(dtype=x.dtype, device=x.device)
        parts = []
        if self.include_input:
            parts.append(x)
        parts.append(torch.sin(args).flatten(start_dim=1))
        parts.append(torch.cos(args).flatten(start_dim=1))
        return torch.cat(parts, dim=-1)


class GeomDeepONetLike(nn.Module):
    """Point-cloud geometry Trunk baseline inspired by Geom-DeepONet."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden: List[int] = [512, 512, 512, 512],
        trunk_hidden: List[int] = [512, 512, 512],
        output_dim: int = 256,
        num_frequencies: int = 6,
        max_frequency: float = 16.0,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.branch = build_mlp(
            branch_input_dim,
            branch_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.geometry_encoder = FourierFeatureEncoder(
            trunk_input_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            include_input=True,
        )
        self.trunk = build_mlp(
            self.geometry_encoder.output_dim,
            trunk_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(params)
        encoded = self.geometry_encoder(node_features)
        trunk_out = self.trunk(encoded)
        return torch.matmul(branch_out, trunk_out.T) + self.bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FourierDeepONetLike(nn.Module):
    """Fourier-DeepONet-like spectral Trunk for arbitrary fixed nodes."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden: List[int] = [512, 512, 512, 512],
        spectral_hidden: List[int] = [512, 512, 512],
        output_dim: int = 256,
        num_frequencies: int = 8,
        max_frequency: float = 32.0,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.branch = build_mlp(
            branch_input_dim,
            branch_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.spectral_encoder = FourierFeatureEncoder(
            trunk_input_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            include_input=True,
        )
        self.spectral_decoder = build_mlp(
            self.spectral_encoder.output_dim,
            spectral_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(params)
        basis = self.spectral_decoder(self.spectral_encoder(node_features))
        return torch.matmul(coeffs, basis.T) + self.bias

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SliceAttentionTrunk(nn.Module):
    """A lightweight physics-attention-style Trunk over learnable slices."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_slices: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_slices < 1:
            raise ValueError("num_slices must be >= 1")

        self.num_slices = int(num_slices)
        self.node_encoder = build_mlp(
            input_dim,
            [hidden_dim],
            hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=0.0,
            initializer=initializer,
            last_activation=True,
        )
        self.assignment = nn.Linear(hidden_dim, num_slices)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        act = ACTIVATIONS.get(activation, nn.SiLU)
        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.attn_norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    act(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())

        self.output_mlp = build_mlp(
            hidden_dim * 2,
            [hidden_dim],
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(node_features)
        values = self.value_proj(x)
        assignment = torch.softmax(self.assignment(x), dim=-1)
        denom = assignment.sum(dim=0, keepdim=True).T.clamp_min(1e-6)
        tokens = torch.matmul(assignment.T, values) / denom

        tokens = tokens.unsqueeze(0)
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms
        ):
            norm_tokens = attn_norm(tokens)
            attn_out, _ = attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
            tokens = tokens + attn_out
            tokens = tokens + ffn(ffn_norm(tokens))
        tokens = tokens.squeeze(0)

        context = torch.matmul(assignment, tokens)
        return self.output_mlp(torch.cat([x, context], dim=-1))


class TransolverLikeDeepONet(nn.Module):
    """Transolver-like slice-attention spatial Trunk with DeepONet fusion."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden: List[int] = [512, 512, 512, 512],
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_slices: int = 64,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.branch = build_mlp(
            branch_input_dim,
            branch_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.trunk = SliceAttentionTrunk(
            input_dim=trunk_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_slices=num_slices,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(params)
        trunk_out = self.trunk(node_features)
        return torch.matmul(branch_out, trunk_out.T) + self.bias

    def count_parameters(self) -> int:
        return _count_trainable(self)


class SineLayer(nn.Module):
    """SIREN-style sine layer used by the Geom-DeepONet source implementation."""

    def __init__(self, in_features: int, out_features: int, w0: float = 10.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = float(w0)
        bound = math.sqrt(6.0 / max(1, in_features)) / self.w0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SineMLP(nn.Module):
    """Small sine-activated MLP that accepts both 2D and 3D tensors."""

    def __init__(self, input_dim: int, hidden: Sequence[int], output_dim: int, w0: float = 10.0):
        super().__init__()
        dims = [int(input_dim), *[int(v) for v in hidden]]
        layers: list[nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.append(SineLayer(din, dout, w0=w0))
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptedGeomDeepONet(nn.Module):
    """Source-referenced Geom-DeepONet adaptation for fixed FEM nodes.

    This keeps the core Geom-DeepONet pattern from the public source: encode the
    global function/geometry branch and node-side geometry, multiply them
    elementwise, pool the mixed representation, then use a sine trunk refinement.
    The SDF/varying-geometry inputs from the original paper are replaced by the
    fixed benchmark node features.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        branch_pre_hidden: List[int] = [512, 512],
        branch_post_hidden: List[int] = [512, 512],
        trunk_pre_hidden: List[int] = [512],
        trunk_post_hidden: List[int] = [512, 512],
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
        siren_w0: float = 10.0,
    ):
        super().__init__()
        self.branch_pre = build_mlp(
            branch_input_dim,
            branch_pre_hidden,
            hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.trunk_pre = build_mlp(
            trunk_input_dim,
            trunk_pre_hidden,
            hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.branch_post = build_mlp(
            hidden_dim,
            branch_post_hidden,
            output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.trunk_post = SineMLP(hidden_dim, trunk_post_hidden, output_dim, w0=siren_w0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        branch_latent = self.branch_pre(params)
        node_latent = self.trunk_pre(node_features)
        mixed = branch_latent[:, None, :] * node_latent[None, :, :]
        pooled = mixed.mean(dim=1)
        coeffs = self.branch_post(pooled)
        basis = self.trunk_post(mixed)
        return torch.einsum("bk,bnk->bn", coeffs, basis) + self.bias

    def count_parameters(self) -> int:
        return _count_trainable(self)


class PhysicsAttentionIrregularMesh(nn.Module):
    """Physics-Attention module following the public Transolver irregular mesh core."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        slice_num: int = 64,
    ):
        super().__init__()
        if dim % heads != 0 and dim_head is None:
            raise ValueError("dim must be divisible by heads when dim_head is not provided")
        self.heads = int(heads)
        self.dim_head = int(dim_head if dim_head is not None else dim // heads)
        self.inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(1, self.heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, self.inner_dim)
        self.in_project_fx = nn.Linear(dim, self.inner_dim)
        self.in_project_slice = nn.Linear(self.dim_head, int(slice_num))
        nn.init.orthogonal_(self.in_project_slice.weight)
        nn.init.zeros_(self.in_project_slice.bias)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = x.shape
        fx_mid = self.in_project_fx(x).view(batch, n_nodes, self.heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).view(batch, n_nodes, self.heads, self.dim_head).permute(0, 2, 1, 3)

        weights = torch.softmax(self.in_project_slice(x_mid) / self.temperature.clamp_min(1e-3), dim=-1)
        norm = weights.sum(dim=2).clamp_min(1e-5)
        tokens = torch.einsum("bhnc,bhng->bhgc", fx_mid, weights) / norm[..., None]

        q = self.to_q(tokens)
        k = self.to_k(tokens)
        v = self.to_v(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out_tokens = torch.matmul(attn, v)

        out = torch.einsum("bhgc,bhng->bhnc", out_tokens, weights)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, n_nodes, self.inner_dim)
        return self.to_out(out)


class TransolverBlock(nn.Module):
    """Residual Transolver block with Physics-Attention and feed-forward update."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        mlp_ratio: int = 2,
        slice_num: int = 64,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class AdaptedTransolver(nn.Module):
    """Source-referenced Transolver adaptation for global SST operating inputs.

    The official Transolver predicts node fields directly with Physics-Attention
    blocks on irregular meshes. Here, the global 1422-dimensional operating input
    is encoded once and broadcast to every fixed FEM node before the Transolver
    blocks. The output is a direct per-node temperature field, not a DeepONet dot
    product.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden: List[int] = [512, 512],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_slices: int = 64,
        mlp_ratio: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.branch_encoder = build_mlp(
            branch_input_dim,
            branch_hidden,
            hidden_dim,
            activation="silu" if activation == "gelu" else activation,
            use_layer_norm=True,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.node_encoder = build_mlp(
            trunk_input_dim,
            [hidden_dim],
            hidden_dim,
            activation="silu" if activation == "gelu" else activation,
            use_layer_norm=True,
            dropout=0.0,
            initializer=initializer,
            last_activation=True,
        )
        self.preprocess = build_mlp(
            hidden_dim * 2,
            [hidden_dim * 2],
            hidden_dim,
            activation="silu" if activation == "gelu" else activation,
            use_layer_norm=True,
            dropout=dropout,
            initializer=initializer,
            last_activation=True,
        )
        self.placeholder = nn.Parameter(torch.zeros(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                    mlp_ratio=mlp_ratio,
                    slice_num=num_slices,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        batch = params.shape[0]
        branch = self.branch_encoder(params)[:, None, :].expand(-1, node_features.shape[0], -1)
        nodes = self.node_encoder(node_features)[None, :, :].expand(batch, -1, -1)
        x = self.preprocess(torch.cat([nodes, branch], dim=-1))
        x = x + self.placeholder[None, None, :]
        for block in self.blocks:
            x = block(x)
        return self.readout(x).squeeze(-1) + self.bias

    def count_parameters(self) -> int:
        return _count_trainable(self)


class SpectralConv3d(nn.Module):
    """3D Fourier layer from the nested Fourier-DeepONet/FNO family."""

    def __init__(self, in_channels: int, out_channels: int, modes: Sequence[int]):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(v) for v in modes)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, *self.modes, 2)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def compl_mul3d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    @staticmethod
    def complex_weight(weights: torch.Tensor, m1: int, m2: int, m3: int) -> torch.Tensor:
        return torch.view_as_complex(weights[:, :, :m1, :m2, :m3, :].contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        depth, height, width = x.shape[-3:]
        m1 = min(self.modes[0], depth)
        m2 = min(self.modes[1], height)
        m3 = min(self.modes[2], width // 2 + 1)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            depth,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.complex_weight(self.weights1, m1, m2, m3)
        )
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.complex_weight(self.weights2, m1, m2, m3)
        )
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.complex_weight(self.weights3, m1, m2, m3)
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.complex_weight(self.weights4, m1, m2, m3)
        )
        return torch.fft.irfftn(out_ft, s=(depth, height, width), dim=(-3, -2, -1))


class FNOGridDecoder3d(nn.Module):
    """Compact 3D FNO decoder used as the nested Fourier-DeepONet output merger."""

    def __init__(
        self,
        width: int = 16,
        modes: Sequence[int] = (4, 4, 4),
        num_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.convs = nn.ModuleList([SpectralConv3d(width, width, modes) for _ in range(num_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(num_layers)])
        self.activation = _activation(activation)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            _activation(activation),
            nn.Conv3d(width * 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, w in zip(self.convs, self.ws):
            x = self.activation(conv(x) + w(x))
            x = self.dropout(x)
        return self.proj(x)


class AdaptedFourierDeepONet(nn.Module):
    """Source-referenced nested Fourier-DeepONet adaptation.

    The public nested Fourier-DeepONet code uses a DeepONet/MIONet front end and
    a 3D FNO decoder as the output merger on regular reservoir grids. This
    adaptation preserves the FNO decoder, then samples the decoded regular latent
    field at the fixed irregular FEM node coordinates.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden: List[int] = [512, 512],
        grid_shape: Sequence[int] = (12, 12, 12),
        width: int = 16,
        modes: Sequence[int] = (4, 4, 4),
        num_fno_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        initializer: str = "glorot_uniform",
    ):
        super().__init__()
        self.grid_shape: Tuple[int, int, int] = tuple(int(v) for v in grid_shape)  # D, H, W
        if len(self.grid_shape) != 3:
            raise ValueError("grid_shape must have three entries")
        self.width = int(width)
        grid_size = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.branch_to_grid = build_mlp(
            branch_input_dim,
            branch_hidden,
            self.width * grid_size,
            activation="silu" if activation == "gelu" else activation,
            use_layer_norm=True,
            dropout=dropout,
            initializer=initializer,
            last_activation=False,
        )
        self.decoder = FNOGridDecoder3d(
            width=self.width,
            modes=modes,
            num_layers=num_fno_layers,
            activation=activation,
            dropout=dropout,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _query_grid(node_features: torch.Tensor) -> torch.Tensor:
        coords = node_features[:, :3]
        lo = coords.min(dim=0, keepdim=True).values
        hi = coords.max(dim=0, keepdim=True).values
        coords01 = (coords - lo) / (hi - lo).clamp_min(1e-6)
        return coords01.mul(2.0).sub(1.0)

    @staticmethod
    def _sample_trilinear(field: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Sample a `(B, 1, D, H, W)` field at fixed `[-1, 1]` xyz queries.

        This avoids `grid_sample` because 3D grid-sample backward is unavailable
        on the current MPS backend. Queries are fixed node coordinates, so
        differentiability with respect to coordinates is not needed.
        """
        batch, _, depth, height, width = field.shape
        qx = (query[:, 0] + 1.0) * 0.5 * (width - 1)
        qy = (query[:, 1] + 1.0) * 0.5 * (height - 1)
        qz = (query[:, 2] + 1.0) * 0.5 * (depth - 1)

        x0 = torch.floor(qx).long().clamp(0, width - 1)
        y0 = torch.floor(qy).long().clamp(0, height - 1)
        z0 = torch.floor(qz).long().clamp(0, depth - 1)
        x1 = (x0 + 1).clamp(0, width - 1)
        y1 = (y0 + 1).clamp(0, height - 1)
        z1 = (z0 + 1).clamp(0, depth - 1)

        wx = (qx - x0.to(qx.dtype)).clamp(0.0, 1.0)
        wy = (qy - y0.to(qy.dtype)).clamp(0.0, 1.0)
        wz = (qz - z0.to(qz.dtype)).clamp(0.0, 1.0)

        flat = field[:, 0].reshape(batch, depth * height * width)

        def gather(z_idx: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
            idx = (z_idx * height * width + y_idx * width + x_idx).to(field.device)
            return torch.gather(flat, 1, idx[None, :].expand(batch, -1))

        c000 = gather(z0, y0, x0)
        c001 = gather(z0, y0, x1)
        c010 = gather(z0, y1, x0)
        c011 = gather(z0, y1, x1)
        c100 = gather(z1, y0, x0)
        c101 = gather(z1, y0, x1)
        c110 = gather(z1, y1, x0)
        c111 = gather(z1, y1, x1)

        wx = wx[None, :]
        wy = wy[None, :]
        wz = wz[None, :]
        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        return c0 * (1 - wz) + c1 * wz

    def forward(self, params: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        batch = params.shape[0]
        latent = self.branch_to_grid(params)
        latent = latent.view(batch, self.width, *self.grid_shape)
        field = self.decoder(latent)
        query = self._query_grid(node_features).to(dtype=field.dtype, device=field.device)
        return self._sample_trilinear(field, query) + self.bias

    def count_parameters(self) -> int:
        return _count_trainable(self)
