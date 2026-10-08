<div align="center">

<img src="assets/readme-banner.svg" alt="SSTONet: topology-aware thermal operator learning" width="100%">

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ast.2026.114015-26796D?style=flat-square)](https://doi.org/10.1016/j.ast.2026.114015)
[![Release](https://img.shields.io/badge/release-v0.1.0-405A55?style=flat-square)](https://github.com/Ne1ther/SSTONet/releases/tag/v0.1.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-687B75?style=flat-square)](#installation)
[![License](https://img.shields.io/badge/license-MIT-A5803E?style=flat-square)](LICENSE)

**Research code for satellite steady-state temperature-field prediction.**

[Paper](https://www.sciencedirect.com/science/article/pii/S1270963826023916) · [Quick start](#quick-start-without-data-or-weights) · [Training](#train-on-your-own-data) · [NXOpen companion](#nxopen-companion) · [Citation](#citation-and-license)

</div>

---

## The paper

**Satellite Temperature Field Prediction under Varying Thermal Contact Conditions Using Topology-Aware Operator Learning**

Chang Xu · Haibo Yang · Qianliang Wu · Zhanxiao Liu · Changsheng Zhou<br>
*Aerospace Science and Technology*, 2026, article 114015<br>
[Publisher page](https://www.sciencedirect.com/science/article/pii/S1270963826023916) · [DOI: 10.1016/j.ast.2026.114015](https://doi.org/10.1016/j.ast.2026.114015)

Repeated thermal assessment is expensive when satellite loads, surface properties, and thermal contacts change. SSTONet learns the mapping from these conditions to a complete steady-state temperature field, with particular attention to component and interface errors.

A multilayer perceptron (MLP) **Branch** encodes each operating case. A graph neural network (GNN) **Trunk** learns a spatial basis from fixed coordinates and connectivity. Their inner product reconstructs nodal temperatures. After training, the graph basis can be prepared once and reused across operating cases.

| Reference problem | Published benchmark |
| :--- | :--- |
| Satellite | Fixed 1U CubeSat, 14 components |
| Temperature field | 5,928 output nodes |
| Reference data | 5,000 Siemens NX SST cases |
| Train / validation / test | 4,000 / 500 / 500 cases |
| Network input | 94 engineering entries + 1,328 incident-flux samples |

### Published accuracy

| Model | Mean relative L2 error | RMSE |
| :--- | ---: | ---: |
| **SSTONet-FEM** | **0.6224%** | **2.2293 K** |
| SSTONet-KNN | 0.6675% | 2.3467 K |

These values describe the selected seed-42 models on 500 held-out cases with the benchmark flux inputs. The paper evaluates interpolation within the sampled parameter domain of the fixed assembly. The synthetic example below checks software operation; it does not reproduce these benchmark results.

## Model architecture

```mermaid
flowchart LR
    U["Operating inputs<br/>and incident flux"] --> B["MLP Branch<br/>case coefficients"]
    G["Fixed coordinates<br/>and connectivity"] --> T["Graph Trunk<br/>spatial encoding"]
    T --> S["Prepared<br/>spatial basis"]
    B --> R["Inner-product<br/>reconstruction"]
    S --> R
    R --> Y["Nodal<br/>temperatures"]
    classDef input fill:#F5F4EE,stroke:#879B8B,color:#173A36;
    classDef model fill:#E6EEE5,stroke:#26796D,color:#173A36;
    classDef output fill:#F5ECD8,stroke:#A5803E,color:#493E25;
    class U,G input;
    class B,T,S,R model;
    class Y output;
```

**SSTONet-FEM** uses supplied finite-element adjacency. **SSTONet-KNN** uses neighbors within components and radius-based links between components. Geometric edge weights guide spatial encoding; contact and coupling values are supplied through the operating input.

The `sstonet.inference` module provides the reusable Trunk cache. Its validity checks bind the spatial basis to the model, graph, device, and precision. See the [quick-start example](examples/quickstart.py) for normalization and cached prediction.

## Release scope

> [!NOTE]
> **v0.1.0 is a source-code release.** It includes model implementations, eleven principal configurations, training and inference utilities, synthetic examples, and self-contained tests. Pretrained weights, simulation datasets, generated outputs, and manuscript working materials are not distributed.

Training creates your own local checkpoints and fitted normalizers under the output directory. Data and generated artifacts are excluded from Git. Reference-data generation uses the separate [NXOpen_satellite](https://github.com/Ne1ther/NXOpen_satellite) toolkit; the neural models here run in a Python environment.

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
├── assets/                 # README banner
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

<details>
<summary><strong>All eleven model configurations</strong></summary>


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

</details>

<details>
<summary><strong>NX export format and node ordering</strong></summary>


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

</details>

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

If this code supports your research, please cite the paper:

```bibtex
@article{xu2026sstonet,
  title   = {Satellite Temperature Field Prediction under Varying Thermal
             Contact Conditions Using Topology-Aware Operator Learning},
  author  = {Xu, Chang and Yang, Haibo and Wu, Qianliang and Liu, Zhanxiao
             and Zhou, Changsheng},
  journal = {Aerospace Science and Technology},
  year    = {2026},
  pages   = {114015},
  doi     = {10.1016/j.ast.2026.114015}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff). The source code is provided under the [MIT License](LICENSE).
