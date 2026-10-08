# -*- coding: utf-8 -*-
"""Tests for the pure-PyTorch GCNLayer implementation."""

import torch

from sstonet.models.deeponet import GCNLayer


def test_gcn_layer_batched_matches_naive_loop():
    torch.manual_seed(0)
    layer = GCNLayer(in_dim=4, out_dim=8)

    bsz, n_nodes = 3, 17
    n_edges = 40

    x = torch.randn(bsz, n_nodes, 4)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

    out = layer(x, edge_index)

    # Naive reference: per-batch index_add on the linearly-transformed features.
    x_lin = layer.linear(x)
    norm = layer._get_norm(n_nodes, edge_index, None).to(dtype=x_lin.dtype)
    src, dst = edge_index
    ref = torch.zeros_like(x_lin)
    for b in range(bsz):
        msg = x_lin[b].index_select(0, src)
        msg.mul_(norm.unsqueeze(1))
        ref[b].index_add_(0, dst, msg)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_gcn_layer_unbatched_shape():
    layer = GCNLayer(in_dim=3, out_dim=5)
    x = torch.randn(11, 3)
    edge_index = torch.randint(0, 11, (2, 20), dtype=torch.long)
    out = layer(x, edge_index)
    assert out.shape == (11, 5)
