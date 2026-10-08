"""Trainer checkpoints must restore learned weights and normalization state."""

import numpy as np
import pytest
import torch

from sstonet.models import (
    ComponentAwareGNNDeepONet, DeepONet, GNNDeepONet, MIONet, MLP, MLPTrainer,
    PureGNN,
)
from sstonet.training import (
    ComponentAwareGNNTrainer, DeepONetTrainer, GNNDeepONetTrainer,
    MIONetTrainer,
)
from sstonet.training.pure_gnn_trainer import PureGNNTrainer


def make_trainer(kind):
    operator = dict(branch_hidden=[8], trunk_hidden=[8], output_dim=4, dropout=0.0)
    if kind == "mlp":
        return MLPTrainer(MLP(3, 6, hidden_dims=[8], dropout=0.0), device="cpu")
    if kind == "deeponet":
        return DeepONetTrainer(DeepONet(3, **operator), device="cpu")
    if kind == "mionet":
        dims = {"first": 1, "second": 2}
        return MIONetTrainer(MIONet(dims, **operator), input_dims=dims, device="cpu")
    if kind == "gnn":
        return GNNDeepONetTrainer(
            GNNDeepONet(3, num_gnn_layers=1, **operator), device="cpu")
    if kind == "component_gnn":
        return ComponentAwareGNNTrainer(ComponentAwareGNNDeepONet(
            3, k=2, cross_component_radius=10.0, use_edge_attr=True,
            num_gnn_layers=1, **operator), device="cpu")
    return PureGNNTrainer(PureGNN(3, hidden_dims=[8], num_gnn_layers=1), device="cpu")


@pytest.mark.parametrize("kind", ["mlp", "deeponet", "mionet", "gnn", "component_gnn", "pure_gnn"])
def test_checkpoint_restores_predictions_into_fresh_trainer(kind, tmp_path):
    rng = np.random.RandomState(2026)
    torch.manual_seed(2026)
    inputs = rng.randn(8, 3) + 5.0
    coords = rng.randn(6, 3) * 2.0
    temperatures = 300.0 + inputs[:, :1] + coords[None, :, 0]
    edges = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    components = np.array([0, 0, 0, 1, 1, 1])
    if kind == "mlp":
        geometry = ()
    elif kind in ("deeponet", "mionet"):
        geometry = (coords,)
    elif kind == "component_gnn":
        geometry = (coords, components)
    else:
        geometry = (coords, edges)

    trained = make_trainer(kind)
    trained.fit(inputs, *geometry, temperatures, epochs=2, batch_size=4, verbose=False)
    expected = trained.predict(inputs[2:5], *geometry)
    checkpoint = tmp_path / "model.pt"
    trained.save(checkpoint)

    torch.manual_seed(99)
    restored = make_trainer(kind)
    assert any(not torch.equal(a, b) for a, b in zip(
        trained.model.parameters(), restored.model.parameters()))
    restored.load(checkpoint)
    actual = restored.predict(inputs[2:5], *geometry)
    assert actual.shape == (3, 6)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
