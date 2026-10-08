# SSTONet

**Topology-aware operator learning for satellite steady-state temperature fields.**

Source-code companion to [Satellite Temperature Field Prediction under Varying Thermal Contact Conditions Using Topology-Aware Operator Learning](https://doi.org/10.1016/j.ast.2026.114015), published in *Aerospace Science and Technology*.

This repository provides the model implementations, principal training configurations, and self-contained tests. **Pretrained weights, simulation datasets, and generated experiment outputs are not included.** Training creates your own local checkpoints; those files are excluded from Git.

## Model overview

SSTONet predicts a nodal temperature field by combining two representations:

- An MLP **Branch** maps each operating input to field coefficients.
- A GNN **Trunk** maps fixed node coordinates and graph connectivity to a spatial basis.
- The inner product of the coefficients and the basis gives the temperature at each node.

**SSTONet-FEM** uses supplied finite-element adjacency. **SSTONet-KNN** builds neighbors within components and radius-based links between components. Graph weights describe geometric proximity; changing thermal-contact values are part of the operating input.

For a trained model and a fixed graph, the spatial basis can be prepared once and reused for later operating inputs. The `sstonet.inference` module implements this cache with checks for changes to the model, graph, device, and precision.

The paper studies a fixed 1U CubeSat with 14 components and 5,928 output nodes. Its temperature-network input combines 94 engineering entries with 1,328 incident solar-flux samples, for 1,422 entries. These dimensions describe the reference problem; the model constructors also accept other input dimensions and node counts.

## Repository layout

```text
.
├── sstonet/
│   ├── models/             # SSTONet and baseline model implementations
│   ├── training/           # Trainers, normalization, checkpoint save/load
│   ├── inference/          # Reusable graph-Trunk basis
│   ├── data/               # NX CSV loader and dataset splitting
│   ├── utils/              # Temperature-field error metrics
│   ├── config.py           # YAML configuration support
│   └── callbacks.py        # Training callbacks
├── configs/                # Eleven principal model configurations
├── scripts/
│   └── train.py            # Shared training entry point
├── examples/
│   └── quickstart.py       # Synthetic training and cached prediction
├── tests/                  # Self-contained model and trainer tests
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

Use Python 3.10 or newer in an isolated environment. For example:

```bash
git clone https://github.com/Ne1ther/SSTONet.git
cd SSTONet
conda create -n sstonet python=3.11
conda activate sstonet
python -m pip install -e ".[dev]"
```

Core dependencies are PyTorch, NumPy, pandas, SciPy, scikit-learn, matplotlib, PyYAML, and tqdm. TensorBoard is optional:

```bash
python -m pip install -e ".[tensorboard]"
```

The release was checked with Python 3.14.4 and PyTorch 2.11.0 on macOS. The self-contained checks use CPU execution. Training can select CPU, CUDA, or Apple MPS according to the available PyTorch backend.

## Quick start without data or weights

The example creates a small synthetic problem in memory, trains a compact graph model, and checks that cached prediction agrees with ordinary prediction:

```bash
python examples/quickstart.py --variant fem
python examples/quickstart.py --variant knn
```

It saves no files. The generated temperatures exercise the software interface and are not NX solutions or paper benchmark results.

## Train on your own data

The training entry point accepts the NX CSV format described below. Supply the export folder and, for FEM-based models, the matching graph folder:

```bash
python scripts/train.py \
  --model gnn_deeponet \
  --config configs/gnn_deeponet.yaml \
  --data-dir /path/to/nx_exports \
  --graph-dir /path/to/graph \
  --output-dir results/my_run \
  --device auto
```

For component-aware neighborhoods:

```bash
python scripts/train.py \
  --model component_gnn_deeponet \
  --config configs/component_gnn_deeponet.yaml \
  --data-dir /path/to/nx_exports \
  --device auto
```

`--device auto` selects an available backend. Use `--device cpu`, `--device cuda`, or `--device mps` to choose explicitly. `--epochs` and `--n-samples` override the YAML settings.

Each run records its effective configuration, exact train/validation/test indices, split hash, training history, error metrics, and locally generated checkpoints. The graph trainers save fitted normalizers with their checkpoints so later predictions can be returned in physical temperature units. Load only checkpoints you generated or trust, because the trainer checkpoint format includes serialized preprocessing objects.

### Principal configurations

| Model | Training identifier and configuration stem |
| --- | --- |
| SSTONet-FEM | `gnn_deeponet` |
| SSTONet-KNN | `component_gnn_deeponet` |
| Full-field GNN | `pure_gnn` |
| DeepONet | `deeponet` |
| POD-DeepONet | `pod_deeponet` |
| MIONet | `mionet` |
| MLP | `mlp` |
| POD-MLP | `pod_mlp` |
| Geometry-aware operator adaptation | `adapted_geom_deeponet` |
| Physics-attention adaptation | `adapted_transolver` |
| Fourier operator adaptation | `adapted_fourier_deeponet` |

For each identifier, the corresponding file is `configs/<identifier>.yaml`. The configurations retain the reference experiments' model and optimizer settings, with portable data paths and automatic device selection. Their standard split is 80% training, 10% validation, and 10% testing. The three adapted baselines are task-specific implementations of the respective model families.

### NX export format

The loader expects case numbers starting at 1. For each case `N`, the export folder contains:

```text
nx_exports/
├── Mixed_Coefficient_DOE_for_nx_automation_N_Solar_vector_log.csv
├── Mixed_Coefficient_DOE_for_nx_automation_N_Load_log.csv
├── Mixed_Coefficient_DOE_for_nx_automation_N_Optical_log.csv
├── Mixed_Coefficient_DOE_for_nx_automation_N_Thermal_coupl_log.csv
└── Mixed_Coefficient_DOE_for_nx_automation_N/
    └── NodeResults/
        ├── NodeResult_<component>_Temperature.csv
        └── Result_<component>_Radiation.csv
```

| Export | Required fields |
| --- | --- |
| Solar vector log | Columns `X`, `Y`, `Z` |
| Load log | Regional powers in the row `Value` |
| Optical log | Rows `Top_inf`, `Bot_inf`, `Top_sunabr`, `Bot_sunabr` |
| Thermal coupling log | Column `value` |
| Temperature CSV | `Temperature`, `X`, `Y`, `Z` |
| Radiation CSV | `Incident_Radiative_Flux_SUN(W/mm2)`, `X`, `Y`, `Z` |

Temperature CSV values are Celsius; the loader converts them to Kelvin. The `TEMPERATURE_COMPONENTS` and `RADIATION_COMPONENTS` lists in [loader.py](sstonet/data/loader.py) define the component concatenation order. Within each component, keep the node order consistent across cases, graph indices, and temperature targets.

The Branch input order is solar direction, incident solar flux, regional powers, emissivities, absorptivities, and contact/coupling values. In the reference problem, these groups have 3, 1,328, 20, 28, 28, and 15 entries, respectively.

A supplied FEM graph folder must contain `edge_index.npy`, an integer array of shape `(2, E)` whose indices match the concatenated temperature nodes. Optional `thermal_coupling_mask.npy` enables interface-region metrics. KNN topology is built from raw node coordinates and component labels. Its radius uses the coordinate units of the input geometry.

## Use the model API with array data

For datasets in another format, pass arrays directly to a trainer:

```python
from sstonet.models import GNNDeepONet
from sstonet.training import GNNDeepONetTrainer

model = GNNDeepONet(branch_input_dim=X_train.shape[1])
trainer = GNNDeepONetTrainer(model, device="cpu", use_edge_attr=True)
trainer.fit(
    X_train, coords, edge_index, T_train,
    X_val=X_val, T_val=T_val,
    epochs=300, batch_size=32,
)
T_pred = trainer.predict(X_test, coords, edge_index)
```

Here `X_train` has shape `(cases, inputs)`, `coords` has shape `(nodes, 3)`, `edge_index` has shape `(2, edges)`, and `T_train` has shape `(cases, nodes)`. Raw model outputs use the trainer's normalized temperature scale; `trainer.predict` reverses that normalization.

For cached inference, `precompute_trunk(model, coords_t, edge_index_t, edge_weight_t)` returns an object with a `predict` method. Use the same normalized coordinates, input normalizer, and geometric edge weights as training. Call `model.eval()` before preparing the cache. The [quick-start example](examples/quickstart.py) shows the complete normalization and reconstruction steps.

## Tests

```bash
python -m pytest -q
```

The suite uses synthetic data to check model outputs, graph operations, training, checkpoint round trips, best-validation-state restoration, and cache correctness/invalidation. It requires no NX installation, private dataset, or pretrained model.

## NXOpen companion

[**NXOpen_satellite**](https://github.com/Ne1ther/NXOpen_satellite) provides Siemens NX / Simcenter 3D automation for parameter updates, batch simulations, solve orchestration, and result export. It supports reference-data generation and is maintained separately from the neural models here.

## Citation and license

Please cite the [associated paper](https://doi.org/10.1016/j.ast.2026.114015) when using this code in research. Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

The source code is provided under the [MIT License](LICENSE).
