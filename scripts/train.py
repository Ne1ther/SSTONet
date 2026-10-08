#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified training entrypoint for SSTONet models."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Make the package importable when this script is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from sstonet.config import Config
from sstonet.data.loader import DataLoader
from sstonet.models import (
    DeepONet,
    MIONet,
    MLP,
    MLPTrainer,
    PODAnalyzer,
    PODDeepONet,
    PODMLP,
    GNNDeepONet,
    ComponentAwareGNNDeepONet,
    PureGNN,
    AdaptedFourierDeepONet,
    AdaptedGeomDeepONet,
    AdaptedTransolver,
)
from sstonet.training import (
    DeepONetTrainer,
    MIONetTrainer,
    GNNDeepONetTrainer,
    ComponentAwareGNNTrainer,
)
from sstonet.training.pure_gnn_trainer import PureGNNTrainer
from sstonet.utils import compute_metrics, print_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


MODEL_CHOICES = [
    "gnn_deeponet",
    "component_gnn_deeponet",
    "pure_gnn",
    "pod_deeponet",
    "deeponet",
    "adapted_geom_deeponet",
    "adapted_transolver",
    "mionet",
    "mlp",
    "pod_mlp",
    "adapted_fourier_deeponet",
]


def load_config(path: str) -> Tuple[Config, Dict[str, Any]]:
    """Load config as Config and raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML configuration must contain a mapping.")
    return Config.from_dict(raw), raw


def seed_everything(
    seed: int = 42, deterministic_algorithms: bool = False
) -> Dict[str, Any]:
    """Seed RNGs and enable the strongest practical deterministic mode.

    ``PYTHONHASHSEED`` only affects Python hashing when it is present before the
    interpreter starts. The returned record states whether that condition
    was satisfied; set it before launching Python for repeatable hashing.

    PyTorch's deterministic-algorithm guard is controlled explicitly by the
    runtime config. Ordinary MPS runs disable the guard while still recording
    all RNG seeds; deterministic diagnostic runs enable it in warning mode.
    """
    import random
    hash_seed_at_start = os.environ.get("PYTHONHASHSEED")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return {
        "python_random_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "pythonhashseed_requested": str(seed),
        "pythonhashseed_at_interpreter_start": hash_seed_at_start,
        "pythonhashseed_effective_at_interpreter_start": hash_seed_at_start == str(seed),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(deterministic_algorithms),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_bitwise_determinism_guaranteed": False,
        "limitation": (
            "PyTorch deterministic algorithms are requested with warn_only=True; "
            "MPS kernels may still be nondeterministic, so bitwise identity is not claimed."
            if deterministic_algorithms
            else "Ordinary MPS execution is used with deterministic algorithms disabled; "
            "all RNG seeds are recorded, but bitwise identity is not claimed."
        ),
    }


def _get_attr(obj: Any, name: str, default: Any) -> Any:
    """Safe getattr with fallback."""
    return getattr(obj, name, default)


def _normalize_optional_list(value: Any) -> Optional[list]:
    """Normalize optional list fields that might be empty or null."""
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 0:
        return None
    return value


def _normalize_train_val_split(train_split: float, val_split: float) -> Tuple[float, float]:
    """Normalize train/val splits to sum to 1.0."""
    total = train_split + val_split
    if total <= 0:
        raise ValueError("train_split + val_split must be > 0 when using a test set.")
    return train_split / total, val_split / total


def _split_indices(
    n_total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Split indices deterministically with the same logic as train_val_test_split."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be nonnegative.")
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    if n_train == 0 or n_val == 0 or (test_ratio > 0 and n_total - n_train - n_val == 0):
        raise ValueError("The sample count is too small for nonempty training, validation and test splits.")

    # Use a local RNG so fixing the data split does not overwrite the NumPy
    # state seeded for model/training randomness.
    perm = np.random.RandomState(seed).permutation(n_total)
    return {
        "train_idx": perm[:n_train],
        "val_idx": perm[n_train:n_train + n_val],
        "test_idx": perm[n_train + n_val:],
    }


def _split_hash(splits: Dict[str, np.ndarray]) -> str:
    """Return a stable SHA-256 over the exact zero-based split indices."""
    payload = {
        key: [int(value) for value in np.asarray(splits[key], dtype=np.int64)]
        for key in ("train_idx", "val_idx", "test_idx")
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_split_artifacts(
    splits: Dict[str, np.ndarray],
    output_dir: Path,
    *,
    split_seed: int,
    n_total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    """Persist exact split indices and a readable hash manifest."""
    split_hash = _split_hash(splits)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=np.asarray(splits["train_idx"], dtype=np.int64),
        val_idx=np.asarray(splits["val_idx"], dtype=np.int64),
        test_idx=np.asarray(splits["test_idx"], dtype=np.int64),
    )
    split_manifest = {
        "schema_version": 1,
        "index_base": 0,
        "n_total": int(n_total),
        "split_seed": int(split_seed),
        "ratios": {
            "train": float(train_ratio),
            "validation": float(val_ratio),
            "test": float(test_ratio),
        },
        "counts": {
            "train": int(len(splits["train_idx"])),
            "validation": int(len(splits["val_idx"])),
            "test": int(len(splits["test_idx"])),
        },
        "sha256": split_hash,
        "hash_encoding": (
            "SHA-256 of canonical JSON containing train_idx, val_idx, and "
            "test_idx as zero-based integer lists"
        ),
        "indices_path": "split_indices.npz",
    }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    return split_hash


def resolve_output_dir(
    config: Config,
    model_name: str,
    output_dir: Optional[str] = None
) -> Path:
    """Resolve output directory for logs and checkpoints."""
    if output_dir:
        base = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = ROOT / "results" / f"{model_name}_{timestamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_tensorboard(config: Config, output_dir: Path) -> Optional[SummaryWriter]:
    """Initialize TensorBoard writer if enabled."""
    enabled = _get_attr(config.training, "tensorboard", False)
    if not enabled:
        return None
    if SummaryWriter is None:
        print("TensorBoard not available; skipping logs.")
        return None
    log_dir = _get_attr(config.training, "log_dir", None)
    log_path = Path(log_dir) if log_dir else output_dir / "tensorboard"
    log_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_path))


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Save metrics to disk."""
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_history(history: Dict[str, list], output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _count_parameters(model: Any) -> int:
    """Count trainable parameters when possible."""
    if hasattr(model, "count_parameters"):
        return int(model.count_parameters())
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        return int(sum(p.numel() for p in model.model.parameters() if p.requires_grad))
    return 0


def build_node_features(
    coords: np.ndarray,
    comp_ids: Optional[np.ndarray] = None,
    use_component_features: bool = False,
    use_radius_features: bool = False,
    use_bbox_features: bool = False,
) -> np.ndarray:
    """Build node-side features for irregular-node baselines."""
    features = [np.asarray(coords, dtype=np.float32)]
    coords_arr = np.asarray(coords, dtype=np.float32)
    centered = coords_arr - coords_arr.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 1e-8:
        scale = 1.0

    if use_radius_features:
        radius = np.linalg.norm(centered, axis=1, keepdims=True) / scale
        features.append(radius.astype(np.float32))

    if use_bbox_features:
        bbox_min = coords_arr.min(axis=0, keepdims=True)
        bbox_max = coords_arr.max(axis=0, keepdims=True)
        bbox_span = np.maximum(bbox_max - bbox_min, 1e-6)
        bbox_coords = (coords_arr - bbox_min) / bbox_span
        features.append(bbox_coords.astype(np.float32))

    if use_component_features:
        if comp_ids is None:
            raise ValueError("comp_ids are required when use_component_features=true")
        comp = np.asarray(comp_ids, dtype=np.int64)
        n_components = int(comp.max()) + 1
        one_hot = np.eye(n_components, dtype=np.float32)[comp]
        features.append(one_hot)

    return np.concatenate(features, axis=1).astype(np.float32)


def add_interface_metrics(metrics: Dict[str, Any], T_pred: np.ndarray, T_true: np.ndarray, graph_dir: Path) -> None:
    """Add IDI and interface/non-interface metrics when a coupling mask exists."""
    coupling_mask_path = graph_dir / "thermal_coupling_mask.npy"
    if not coupling_mask_path.exists():
        return
    coupling_mask = np.load(coupling_mask_path).astype(bool)
    if coupling_mask.shape[0] != T_true.shape[1]:
        return
    interface_metrics = compute_metrics(T_pred[:, coupling_mask], T_true[:, coupling_mask])
    non_interface_metrics = compute_metrics(T_pred[:, ~coupling_mask], T_true[:, ~coupling_mask])
    for key, value in interface_metrics.items():
        metrics[f"interface_{key}"] = value
    for key, value in non_interface_metrics.items():
        metrics[f"non_interface_{key}"] = value
    metrics["IDI"] = float(interface_metrics["RelL2"] / (non_interface_metrics["RelL2"] + 1e-12))
    metrics["interface_nodes"] = int(coupling_mask.sum())
    metrics["non_interface_nodes"] = int((~coupling_mask).sum())


def _save_config_snapshot(config: Config, output_dir: Path) -> None:
    """Save all effective settings, including command-line overrides."""
    raw = {
        section: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(getattr(config, section)).items()
        }
        for section in ("data", "model", "training")
    }
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False)


def _resolve_repo_path(value: str) -> Path:
    """Resolve config paths against the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _log_history(writer: Optional[SummaryWriter], history: Dict[str, list]) -> None:
    """Write training history to TensorBoard."""
    if writer is None:
        return
    for idx, loss in enumerate(history.get("train_loss", []), start=1):
        writer.add_scalar("loss/train", loss, idx)
    for idx, loss in enumerate(history.get("val_loss", []), start=1):
        writer.add_scalar("loss/val", loss, idx)
    for idx, lr in enumerate(history.get("lr", []), start=1):
        writer.add_scalar("lr", lr, idx)


def run_training(
    model_name: str,
    config_path: str,
    output_dir: Optional[str] = None,
    epochs_override: Optional[int] = None,
    n_samples_override: Optional[int] = None,
    device_override: Optional[str] = None,
    verbose_override: Optional[bool] = None,
    data_dir_override: Optional[str] = None,
    graph_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run training and return metrics summary."""
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model: {model_name}")
    config, _ = load_config(config_path)
    if epochs_override is not None:
        config.training.epochs = int(epochs_override)
    if n_samples_override is not None:
        config.data.n_samples = int(n_samples_override)
    if device_override is not None:
        config.training.device = device_override
    if verbose_override is not None:
        config.training.verbose = bool(verbose_override)
    if data_dir_override is not None:
        config.data.root = Path(data_dir_override).expanduser().resolve()
        config.data.result_subdir = "."
    else:
        config.data.root = _resolve_repo_path(config.data.root)
    config.data.graph_dir = (
        Path(graph_dir_override).expanduser().resolve()
        if graph_dir_override is not None
        else _resolve_repo_path(_get_attr(config.data, "graph_dir", "data/graph"))
    )
    if _get_attr(config.data, "test_root", None):
        config.data.test_root = str(_resolve_repo_path(config.data.test_root))
    if int(config.training.epochs) <= 0:
        raise ValueError("epochs must be positive.")
    if not config.data.result_dir.is_dir():
        raise FileNotFoundError(
            f"NX export directory does not exist: {config.data.result_dir}. "
            "Place your NX CSV exports in data/nx_exports or pass --data-dir PATH."
        )
    if model_name in {"gnn_deeponet", "pure_gnn"}:
        edge_path = config.data.graph_dir / "edge_index.npy"
        if not edge_path.is_file():
            raise FileNotFoundError(
                f"FEM graph connectivity is missing: {edge_path}. "
                "Supply an edge_index.npy aligned to the exported temperature nodes with --graph-dir PATH."
            )
    seed = int(_get_attr(config.training, "seed", 42))
    split_seed = int(_get_attr(config.training, "split_seed", seed))
    deterministic_algorithms = bool(
        _get_attr(config.training, "deterministic_algorithms", False)
    )
    determinism = seed_everything(seed, deterministic_algorithms)

    output_path = resolve_output_dir(config, model_name, output_dir)
    _save_config_snapshot(config, output_path)

    data_cfg = config.data
    include_radiation = _get_attr(data_cfg, "include_radiation", True)
    radiation_pca_dim = _get_attr(data_cfg, "radiation_pca_dim", 0)
    loader = DataLoader(data_cfg.result_dir, n_samples=data_cfg.n_samples)
    indices = list(range(1, loader.n_samples + 1))

    train_split = _get_attr(config.training, "train_split", 0.7)
    val_split = _get_attr(config.training, "val_split", 0.15)
    test_split = _get_attr(config.training, "test_split", 0.15)

    test_root = _get_attr(data_cfg, "test_root", None)
    if isinstance(test_root, str) and test_root.strip() == "":
        test_root = None

    if test_root:
        train_ratio, val_ratio = _normalize_train_val_split(train_split, val_split)
        splits = _split_indices(
            len(indices),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=0.0,
            seed=split_seed,
        )
        train_ids = [indices[i] for i in splits["train_idx"]]
        pca_save_path = None
        if include_radiation and radiation_pca_dim > 0:
            pca_save_path = output_path / "radiation_pca.pkl"
        dataset = loader.load_dataset(
            indices=indices,
            include_radiation=include_radiation,
            radiation_pca_dim=radiation_pca_dim,
            pca_fit_indices=train_ids,
            pca_save_path=pca_save_path,
        )
        X = dataset["X"]
        T = dataset["T"]
        coords = dataset["coords"]
        comp_ids = dataset.get("comp_ids")  # Component IDs for the component-aware graph.

        X_train = X[splits["train_idx"]]
        T_train = T[splits["train_idx"]]
        X_val = X[splits["val_idx"]]
        T_val = T[splits["val_idx"]]

        test_root_path = Path(test_root)
        test_result_subdir = _get_attr(data_cfg, "test_result_subdir", data_cfg.result_subdir)
        test_n_samples = _get_attr(data_cfg, "test_n_samples", None)
        test_loader = DataLoader(test_root_path / test_result_subdir, n_samples=test_n_samples)
        test_dataset = test_loader.load_dataset(
            include_radiation=include_radiation,
            radiation_pca_dim=radiation_pca_dim,
            pca_model=loader.radiation_pca_model,
        )
        X_test = test_dataset["X"]
        T_test = test_dataset["T"]
        coords_test = test_dataset["coords"]
        comp_ids_test = test_dataset.get("comp_ids")  # Test-set component IDs.

        if X_test.shape[1] != X.shape[1]:
            raise ValueError(
                f"test input_dim {X_test.shape[1]} does not match training input_dim {X.shape[1]}"
            )
        if T_test.shape[1] != T.shape[1]:
            raise ValueError(
                f"test output_dim {T_test.shape[1]} does not match training output_dim {T.shape[1]}"
            )
    else:
        splits = _split_indices(
            len(indices),
            train_ratio=train_split,
            val_ratio=val_split,
            test_ratio=test_split,
            seed=split_seed,
        )
        train_ids = [indices[i] for i in splits["train_idx"]]
        pca_save_path = None
        if include_radiation and radiation_pca_dim > 0:
            pca_save_path = output_path / "radiation_pca.pkl"
        dataset = loader.load_dataset(
            indices=indices,
            include_radiation=include_radiation,
            radiation_pca_dim=radiation_pca_dim,
            pca_fit_indices=train_ids,
            pca_save_path=pca_save_path,
        )
        X = dataset["X"]
        T = dataset["T"]
        coords = dataset["coords"]
        comp_ids = dataset.get("comp_ids")  # Component IDs.

        X_train = X[splits["train_idx"]]
        T_train = T[splits["train_idx"]]
        X_val = X[splits["val_idx"]]
        T_val = T[splits["val_idx"]]
        X_test = X[splits["test_idx"]]
        T_test = T[splits["test_idx"]]
        coords_test = coords
        comp_ids_test = comp_ids  # Geometry and component IDs are shared across these cases.

    if test_root:
        split_train_ratio, split_val_ratio = _normalize_train_val_split(
            train_split, val_split
        )
        split_test_ratio = 0.0
    else:
        split_train_ratio = train_split
        split_val_ratio = val_split
        split_test_ratio = test_split
    split_hash = save_split_artifacts(
        splits,
        output_path,
        split_seed=split_seed,
        n_total=len(indices),
        train_ratio=split_train_ratio,
        val_ratio=split_val_ratio,
        test_ratio=split_test_ratio,
    )
    determinism.update(
        {
            "model_seed": seed,
            "split_seed": split_seed,
            "split_hash": split_hash,
            "device_requested": str(_get_attr(config.training, "device", None)),
        }
    )
    with open(output_path / "determinism.json", "w") as f:
        json.dump(determinism, f, indent=2)

    input_dim = X.shape[1]
    output_dim = T.shape[1]

    writer = setup_tensorboard(config, output_path)

    epochs = _get_attr(config.training, "epochs", 500)
    batch_size = _get_attr(config.training, "batch_size", 64)
    lr = _get_attr(config.training, "learning_rate", 1e-3)
    weight_decay = _get_attr(config.training, "weight_decay", 1e-4)
    grad_clip = _get_attr(config.training, "grad_clip", 1.0)
    device = _get_attr(config.training, "device", None)
    verbose = _get_attr(config.training, "verbose", True)

    metrics = {}
    train_start_time = time.time()

    if model_name == "mlp":
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=_get_attr(config.model, "hidden_dims", [256, 256, 256, 128]),
            bottleneck_dim=_get_attr(config.model, "bottleneck_dim", None),
            activation=_get_attr(config.model, "activation", "relu"),
            use_batch_norm=_get_attr(config.model, "use_batch_norm", False),
            dropout=_get_attr(config.model, "dropout", 0.1),
            initializer=_get_attr(config.model, "initializer", "he_uniform"),
        )
        trainer = MLPTrainer(model, lr=lr, weight_decay=weight_decay, device=device)
        history = trainer.fit(
            X_train,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test)

    elif model_name == "pod_mlp":
        model = PODMLP(
            n_modes=_get_attr(config.model, "n_modes", 64),
            hidden_layers=_get_attr(config.model, "hidden_layers", [256, 128]),
            dropout=_get_attr(config.model, "dropout", 0.1),
        )
        history = model.fit(
            X_train,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
            device=device,
        )
        T_pred = model.predict(X_test)

    elif model_name in {"deeponet", "pod_deeponet"}:
        branch_hidden = _get_attr(config.model, "branch_hidden", [256, 256, 256])
        trunk_hidden = _get_attr(config.model, "trunk_hidden", [256, 256, 256])
        trunk_hidden = _normalize_optional_list(trunk_hidden)
        trunk_input_dim = _get_attr(config.model, "trunk_input_dim", coords.shape[1])
        activation = _get_attr(config.model, "activation", "relu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", False)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")

        if model_name == "deeponet":
            model = DeepONet(
                branch_input_dim=input_dim,
                trunk_input_dim=trunk_input_dim,
                branch_hidden=branch_hidden,
                trunk_hidden=trunk_hidden or [256, 256, 256],
                output_dim=_get_attr(config.model, "output_dim", 128),
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
                initializer=initializer,
            )
        else:
            pod = PODAnalyzer().fit(T_train, n_modes=_get_attr(config.model, "n_modes", 64))
            model = PODDeepONet(
                pod_basis=pod.basis,
                branch_input_dim=input_dim,
                pod_mean=pod.mean,
                branch_hidden=branch_hidden,
                activation=activation,
                trunk_hidden=trunk_hidden,
                trunk_input_dim=trunk_input_dim,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
                initializer=initializer,
            )

        trainer = DeepONetTrainer(model, lr=lr, weight_decay=weight_decay, device=device)
        history = trainer.fit(
            X_train,
            coords,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test, coords_test)

    elif model_name == "mionet":
        input_dims = _get_attr(config.model, "input_dims", None)
        if input_dims is None:
            input_dims = MIONetTrainer.INPUT_DIMS
        if sum(input_dims.values()) != input_dim:
            raise ValueError(
                f"input_dims sum {sum(input_dims.values())} does not match input_dim {input_dim}"
            )

        branch_hidden = _get_attr(config.model, "branch_hidden", [256, 256])
        trunk_hidden = _get_attr(config.model, "trunk_hidden", [256, 256, 256])
        trunk_hidden = _normalize_optional_list(trunk_hidden)
        trunk_input_dim = _get_attr(config.model, "trunk_input_dim", coords.shape[1])
        merge = _get_attr(config.model, "merge", "product")
        activation = _get_attr(config.model, "activation", "relu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", False)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")

        model = MIONet(
            input_dims=input_dims,
            trunk_input_dim=trunk_input_dim,
            branch_hidden=branch_hidden,
            trunk_hidden=trunk_hidden or [256, 256, 256],
            output_dim=_get_attr(config.model, "output_dim", 128),
            merge=merge,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )

        trainer = MIONetTrainer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            input_dims=input_dims,
        )
        history = trainer.fit(
            X_train,
            coords,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test, coords_test)

    elif model_name in {
        "adapted_geom_deeponet",
        "adapted_transolver",
        "adapted_fourier_deeponet",
    }:
        use_component_features = _get_attr(config.model, "use_component_features", True)
        use_radius_features = _get_attr(config.model, "use_radius_features", True)
        use_bbox_features = _get_attr(config.model, "use_bbox_features", True)
        node_features = build_node_features(
            coords,
            comp_ids=comp_ids,
            use_component_features=use_component_features,
            use_radius_features=use_radius_features,
            use_bbox_features=use_bbox_features,
        )
        node_features_test = build_node_features(
            coords_test,
            comp_ids=comp_ids_test,
            use_component_features=use_component_features,
            use_radius_features=use_radius_features,
            use_bbox_features=use_bbox_features,
        )

        branch_hidden = _get_attr(config.model, "branch_hidden", [512, 512, 512, 512])
        output_basis_dim = _get_attr(config.model, "output_dim", 256)
        activation = _get_attr(config.model, "activation", "silu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", True)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")

        if model_name == "adapted_geom_deeponet":
            model = AdaptedGeomDeepONet(
                branch_input_dim=input_dim,
                trunk_input_dim=node_features.shape[1],
                hidden_dim=_get_attr(config.model, "hidden_dim", 256),
                output_dim=output_basis_dim,
                branch_pre_hidden=_get_attr(config.model, "branch_pre_hidden", [512, 512]),
                branch_post_hidden=_get_attr(config.model, "branch_post_hidden", [512, 512]),
                trunk_pre_hidden=_get_attr(config.model, "trunk_pre_hidden", [512]),
                trunk_post_hidden=_get_attr(config.model, "trunk_post_hidden", [512, 512]),
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
                initializer=initializer,
                siren_w0=_get_attr(config.model, "siren_w0", 10.0),
            )
            metrics["baseline_adapter"] = "source_referenced_geom_deeponet"
        elif model_name == "adapted_transolver":
            model = AdaptedTransolver(
                branch_input_dim=input_dim,
                trunk_input_dim=node_features.shape[1],
                branch_hidden=branch_hidden,
                hidden_dim=_get_attr(config.model, "hidden_dim", 256),
                num_layers=_get_attr(config.model, "num_layers", 4),
                num_heads=_get_attr(config.model, "num_heads", 8),
                num_slices=_get_attr(config.model, "num_slices", 64),
                mlp_ratio=_get_attr(config.model, "mlp_ratio", 2),
                activation=_get_attr(config.model, "activation", "gelu"),
                dropout=dropout,
                initializer=initializer,
            )
            metrics["baseline_adapter"] = "source_referenced_transolver"
        elif model_name == "adapted_fourier_deeponet":
            model = AdaptedFourierDeepONet(
                branch_input_dim=input_dim,
                trunk_input_dim=node_features.shape[1],
                branch_hidden=branch_hidden,
                grid_shape=_get_attr(config.model, "grid_shape", [12, 12, 12]),
                width=_get_attr(config.model, "width", 16),
                modes=_get_attr(config.model, "modes", [4, 4, 4]),
                num_fno_layers=_get_attr(config.model, "num_fno_layers", 4),
                activation=_get_attr(config.model, "activation", "gelu"),
                dropout=dropout,
                initializer=initializer,
            )
            metrics["baseline_adapter"] = "source_referenced_nested_fourier_deeponet"
        else:
            raise ValueError(f"Unknown adapted baseline model: {model_name}")

        trainer = DeepONetTrainer(model, lr=lr, weight_decay=weight_decay, device=device)
        history = trainer.fit(
            X_train,
            node_features,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test, node_features_test)
        metrics["node_feature_dim"] = int(node_features.shape[1])
        metrics["use_component_features"] = bool(use_component_features)
        metrics["use_radius_features"] = bool(use_radius_features)
        metrics["use_bbox_features"] = bool(use_bbox_features)

    elif model_name == "gnn_deeponet":
        branch_hidden = _get_attr(config.model, "branch_hidden", [512, 512, 512, 512])
        trunk_hidden = _get_attr(config.model, "trunk_hidden", [512, 512, 512, 512])
        trunk_input_dim = _get_attr(config.model, "trunk_input_dim", coords.shape[1])
        activation = _get_attr(config.model, "activation", "silu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", True)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")
        num_gnn_layers = _get_attr(config.model, "num_gnn_layers", 2)
        use_edge_attr = _get_attr(config.model, "use_edge_attr", False)

        # Load fixed FEM graph connectivity.
        graph_dir = Path(_get_attr(data_cfg, "graph_dir", "data/graph"))
        edge_index = np.load(graph_dir / "edge_index.npy")

        model = GNNDeepONet(
            branch_input_dim=input_dim,
            trunk_input_dim=trunk_input_dim,
            branch_hidden=branch_hidden,
            trunk_hidden=trunk_hidden,
            output_dim=_get_attr(config.model, "output_dim", 256),
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )

        trainer = GNNDeepONetTrainer(model, lr=lr, weight_decay=weight_decay, device=device, use_edge_attr=use_edge_attr)
        history = trainer.fit(
            X_train,
            coords,
            edge_index,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test, coords_test, edge_index)

    elif model_name == "component_gnn_deeponet":
        # Build the component-aware KNN graph model.
        branch_hidden = _get_attr(config.model, "branch_hidden", [512, 512, 512, 512])
        trunk_hidden = _get_attr(config.model, "trunk_hidden", [512, 512, 512, 512])
        trunk_input_dim = _get_attr(config.model, "trunk_input_dim", coords.shape[1])
        activation = _get_attr(config.model, "activation", "silu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", True)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")
        num_gnn_layers = _get_attr(config.model, "num_gnn_layers", 2)
        k = _get_attr(config.model, "k", 15)  # Within-component nearest neighbors.
        cross_component_radius = _get_attr(config.model, "cross_component_radius", 2.0)
        use_edge_attr = _get_attr(config.model, "use_edge_attr", False)

        if comp_ids is None:
            raise ValueError("comp_ids not found in dataset. Required for component_gnn_deeponet.")

        model = ComponentAwareGNNDeepONet(
            branch_input_dim=input_dim,
            trunk_input_dim=trunk_input_dim,
            branch_hidden=branch_hidden,
            trunk_hidden=trunk_hidden,
            output_dim=_get_attr(config.model, "output_dim", 256),
            num_gnn_layers=num_gnn_layers,
            k=k,
            cross_component_radius=cross_component_radius,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
            use_edge_attr=use_edge_attr,
        )

        trainer = ComponentAwareGNNTrainer(model, lr=lr, weight_decay=weight_decay, device=device)
        history = trainer.fit(
            X_train,
            coords,
            comp_ids,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        T_pred = trainer.predict(X_test, coords_test, comp_ids_test)

    elif model_name == "pure_gnn":
        # Full-field GNN baseline without Branch-Trunk factorization.
        hidden_dims = _get_attr(config.model, "hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = _get_attr(config.model, "trunk_hidden", [512, 512, 512, 512])
        activation = _get_attr(config.model, "activation", "silu")
        use_layer_norm = _get_attr(config.model, "use_layer_norm", True)
        dropout = _get_attr(config.model, "dropout", 0.0)
        initializer = _get_attr(config.model, "initializer", "glorot_uniform")
        num_gnn_layers = _get_attr(config.model, "num_gnn_layers", 3)
        use_edge_attr = _get_attr(config.model, "use_edge_attr", False)

        # Load fixed FEM graph connectivity.
        graph_dir = Path(_get_attr(data_cfg, "graph_dir", "data/graph"))
        edge_index = np.load(graph_dir / "edge_index.npy")

        model = PureGNN(
            param_dim=input_dim,
            coord_dim=coords.shape[1],
            hidden_dims=hidden_dims,
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            initializer=initializer,
        )

        trainer = PureGNNTrainer(
            model, lr=lr, weight_decay=weight_decay,
            device=device, use_edge_attr=use_edge_attr
        )
        history = trainer.fit(
            X_train,
            coords,
            edge_index,
            T_train,
            X_val=X_val,
            T_val=T_val,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_dir=str(output_path / "checkpoints"),
        )
        # Persist completed training before evaluation, so a test-time failure
        # never forces an otherwise completed 300-epoch run to be repeated.
        save_history(history, output_path)
        evaluation_batch_size = _get_attr(config.training, "evaluation_batch_size", 4)
        metrics["evaluation"] = trainer.validate_prediction(
            X_val, coords, edge_index, T_val, history,
            batch_size=evaluation_batch_size,
        )
        T_pred = trainer.predict(
            X_test, coords_test, edge_index, batch_size=evaluation_batch_size
        )
        metrics["evaluation"]["test_count"] = int(len(X_test))
        metrics["evaluation"]["torch_version"] = torch.__version__
        metrics["evaluation"]["device"] = str(trainer.device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    _log_history(writer, history)

    train_elapsed_time = time.time() - train_start_time
    full_metrics = compute_metrics(T_pred, T_test)
    graph_dir = Path(_get_attr(data_cfg, "graph_dir", "data/graph"))
    add_interface_metrics(full_metrics, T_pred, T_test, graph_dir)
    full_metrics.update(metrics)
    full_metrics["model_name"] = model_name
    full_metrics["model_class"] = model.__class__.__name__
    full_metrics["model_seed"] = seed
    full_metrics["split_seed"] = split_seed
    full_metrics["split_hash"] = split_hash
    full_metrics["pythonhashseed_effective_at_interpreter_start"] = determinism[
        "pythonhashseed_effective_at_interpreter_start"
    ]
    full_metrics["torch_deterministic_algorithms"] = determinism[
        "torch_deterministic_algorithms"
    ]
    full_metrics["mps_bitwise_determinism_guaranteed"] = False
    full_metrics["n_params"] = _count_parameters(model)
    full_metrics["train_time_seconds"] = round(train_elapsed_time, 2)
    full_metrics["output_dir"] = str(output_path)

    for key, value in full_metrics.items():
        if writer is not None and isinstance(value, (int, float)):
            writer.add_scalar(f"metrics/{key}", value, 0)

    if writer is not None:
        writer.flush()
        writer.close()

    save_metrics(full_metrics, output_path)
    save_history(history, output_path)

    print_metrics(model_name, full_metrics)

    return full_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSTONet models")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke tests")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, mps or cuda[:INDEX]")
    parser.add_argument("--data-dir", default=None, help="Directory containing NX CSV case exports")
    parser.add_argument("--graph-dir", default=None, help="Directory containing FEM edge_index.npy")
    parser.add_argument("--quiet", action="store_true", help="Disable per-epoch training logs")

    args = parser.parse_args()
    try:
        run_training(
            args.model,
            args.config,
            args.output_dir,
            epochs_override=args.epochs,
            n_samples_override=args.n_samples,
            device_override=args.device,
            verbose_override=False if args.quiet else None,
            data_dir_override=args.data_dir,
            graph_dir_override=args.graph_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
