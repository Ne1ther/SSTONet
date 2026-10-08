"""Train and predict on a tiny synthetic problem without saving any files.

This exercises the model interface, normalizers, graph construction, and Trunk
cache. The synthetic temperatures are not NX solutions or paper benchmark data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sstonet.inference import precompute_trunk
from sstonet.models import GNNDeepONet, ComponentAwareGNNDeepONet
from sstonet.training import GNNDeepONetTrainer, ComponentAwareGNNTrainer
from sstonet.utils import compute_metrics


def run(variant="fem", epochs=3):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    coords = rng.uniform(-1.0, 1.0, size=(24, 3)).astype(np.float32)
    comp_ids = np.repeat(np.arange(3), 8)
    X = rng.normal(size=(64, 6)).astype(np.float32)
    T = (
        290.0 + 3.0 * X[:, 0, None] + X[:, 1, None] * coords[None, :, 0]
        + 0.5 * X[:, 2, None] * coords[None, :, 1]
    ).astype(np.float32)
    model_args = dict(
        branch_input_dim=X.shape[1], branch_hidden=[32, 32],
        trunk_hidden=[32, 32], output_dim=16, num_gnn_layers=2,
    )
    if variant == "fem":
        # A bidirectional chain stands in for a supplied fixed mesh graph.
        src = np.arange(len(coords) - 1)
        dst = src + 1
        edge_index = np.array(
            [np.concatenate([src, dst]), np.concatenate([dst, src])],
            dtype=np.int64,
        )
        model = GNNDeepONet(**model_args)
        trainer = GNNDeepONetTrainer(model, device="cpu", use_edge_attr=True)
        trainer.fit(
            X[:48], coords, edge_index, T[:48],
            X_val=X[48:56], T_val=T[48:56],
            epochs=epochs, batch_size=8, verbose=False,
        )
        predictions = trainer.predict(X[56:], coords, edge_index)
        distances = np.linalg.norm(coords[edge_index[0]] - coords[edge_index[1]], axis=1)
    else:
        model = ComponentAwareGNNDeepONet(
            **model_args, k=3, cross_component_radius=0.5, use_edge_attr=True,
        )
        trainer = ComponentAwareGNNTrainer(model, device="cpu")
        trainer.fit(
            X[:48], coords, comp_ids, T[:48],
            X_val=X[48:56], T_val=T[48:56],
            epochs=epochs, batch_size=8, verbose=False,
        )
        predictions = trainer.predict(X[56:], coords, comp_ids)
        edge_index, distances = model.build_graph(coords, comp_ids)

    weights = np.exp(-distances ** 2 / (2.0 * distances.mean() ** 2))
    coords_t = torch.tensor(trainer.scaler_coords.transform(coords), dtype=torch.float32)
    inputs_t = torch.tensor(trainer.scaler_X.transform(X[56:]), dtype=torch.float32)
    edges_t = torch.tensor(edge_index, dtype=torch.long)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    model.eval()
    cache = precompute_trunk(model, coords_t, edges_t, weights_t)
    normalized_prediction = cache.predict(inputs_t).cpu().numpy()
    cached_predictions = trainer.scaler_T.inverse_transform(
        normalized_prediction.reshape(-1, 1)
    ).reshape(predictions.shape)
    np.testing.assert_allclose(cached_predictions, predictions, atol=0.002, rtol=0)
    print(f"Synthetic {variant.upper()} example: {predictions.shape[0]} cases, "
          f"{predictions.shape[1]} nodes; cache agrees within 0.002 K.")
    print(f"Synthetic test RMSE: {compute_metrics(predictions, T[56:])['RMSE']:.3f} K")
    print("No weights, data, or outputs were saved.")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["fem", "knn"], default="fem")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.variant, args.epochs)
