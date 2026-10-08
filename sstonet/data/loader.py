# -*- coding: utf-8 -*-
"""Data loader for satellite thermal simulation data"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - optional dependency
    PCA = None

RADIATION_COMPONENTS = [
    "Board_back_sun", "Board_back_wind", "Board_face_aero",
    "Board_face_earth", "Board_face_sun", "Board_face_wind",
    "Fixed Frame", "Main Frame"
]

TEMPERATURE_COMPONENTS = [
    "Battery", "Battery Control", "Battery Frame", "Board_back_sun",
    "Board_back_wind", "Board_face_aero", "Board_face_earth",
    "Board_face_sun", "Board_face_wind", "Fixed Frame", "Main Frame",
    "On Board Computer", "S Band", "Telemetry Board"
]


class DataLoader:
    """Load exported NX thermal simulation inputs and temperature fields."""

    def __init__(self, result_dir: Path, n_samples: int = None):
        self.result_dir = result_dir
        self.n_samples = n_samples or self._count_samples()
        self.radiation_pca_model: Optional[Any] = None

    def _count_samples(self) -> int:
        count = 0
        for d in self.result_dir.iterdir():
            if d.is_dir() and d.name.startswith("Mixed_Coefficient_DOE_for_nx_automation_"):
                count += 1
        return count

    def _get_sample_dir(self, idx: int) -> Path:
        return self.result_dir / f"Mixed_Coefficient_DOE_for_nx_automation_{idx}"

    def load_solar_vector(self, idx: int) -> np.ndarray:
        path = self.result_dir / f"Mixed_Coefficient_DOE_for_nx_automation_{idx}_Solar_vector_log.csv"
        df = pd.read_csv(path)
        return df[['X', 'Y', 'Z']].values.flatten()

    def load_radiation(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample_dir = self._get_sample_dir(idx) / "NodeResults"
        flux_list, coord_list = [], []

        for comp in RADIATION_COMPONENTS:
            path = sample_dir / f"Result_{comp}_Radiation.csv"
            df = pd.read_csv(path)
            flux_list.append(df['Incident_Radiative_Flux_SUN(W/mm2)'].values)
            coord_list.append(df[['X', 'Y', 'Z']].values)

        return np.concatenate(flux_list), np.vstack(coord_list)

    def load_radiation_graph_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load radiation field together with coordinates and component ids."""
        sample_dir = self._get_sample_dir(idx) / "NodeResults"
        flux_list, coord_list, comp_id_list = [], [], []

        for comp_idx, comp in enumerate(RADIATION_COMPONENTS):
            path = sample_dir / f"Result_{comp}_Radiation.csv"
            df = pd.read_csv(path)
            flux = df['Incident_Radiative_Flux_SUN(W/mm2)'].values
            coords = df[['X', 'Y', 'Z']].values
            flux_list.append(flux)
            coord_list.append(coords)
            comp_id_list.append(np.full(len(df), comp_idx, dtype=np.int64))

        return (
            np.concatenate(flux_list),
            np.vstack(coord_list),
            np.concatenate(comp_id_list),
        )

    def load_heat_load(self, idx: int) -> np.ndarray:
        path = self.result_dir / f"Mixed_Coefficient_DOE_for_nx_automation_{idx}_Load_log.csv"
        df = pd.read_csv(path, index_col=0)
        return df.loc['Value'].values.astype(float)

    def load_optical(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self.result_dir / f"Mixed_Coefficient_DOE_for_nx_automation_{idx}_Optical_log.csv"
        df = pd.read_csv(path, index_col=0)

        inf_top = df.loc['Top_inf'].values.astype(float)
        inf_bot = df.loc['Bot_inf'].values
        sunabr_top = df.loc['Top_sunabr'].values.astype(float)
        sunabr_bot = df.loc['Bot_sunabr'].values

        inf_bot = pd.to_numeric(inf_bot, errors='coerce')
        sunabr_bot = pd.to_numeric(sunabr_bot, errors='coerce')
        inf_bot = np.where(np.isnan(inf_bot), inf_top, inf_bot)
        sunabr_bot = np.where(np.isnan(sunabr_bot), sunabr_top, sunabr_bot)

        inf = np.concatenate([inf_top, inf_bot])
        sunabr = np.concatenate([sunabr_top, sunabr_bot])

        return inf, sunabr

    def load_thermal_coupling(self, idx: int) -> np.ndarray:
        path = self.result_dir / f"Mixed_Coefficient_DOE_for_nx_automation_{idx}_Thermal_coupl_log.csv"
        df = pd.read_csv(path, index_col=0)
        return df['value'].values.astype(float)

    def load_temperature(self, idx: int, to_kelvin: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_dir = self._get_sample_dir(idx) / "NodeResults"
        temp_list, coord_list, comp_id_list = [], [], []

        for comp_idx, comp in enumerate(TEMPERATURE_COMPONENTS):
            path = sample_dir / f"NodeResult_{comp}_Temperature.csv"
            df = pd.read_csv(path)
            temp_list.append(df['Temperature'].values)
            coord_list.append(df[['X', 'Y', 'Z']].values)
            comp_id_list.append(np.full(len(df), comp_idx))

        temp = np.concatenate(temp_list)
        if to_kelvin:
            temp = temp + 273.15

        return temp, np.vstack(coord_list), np.concatenate(comp_id_list)

    def _load_base_inputs(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        solar = self.load_solar_vector(idx)
        load = self.load_heat_load(idx)
        inf, sunabr = self.load_optical(idx)
        coupling = self.load_thermal_coupling(idx)
        rest = np.concatenate([load, inf, sunabr, coupling])
        return solar, rest

    def load_all_inputs(
        self,
        idx: int,
        include_radiation: bool = True,
        radiation_pca_dim: int = 0,
        pca_model: Optional[Any] = None,
    ) -> np.ndarray:
        solar, rest = self._load_base_inputs(idx)

        features = [solar]
        if include_radiation:
            flux, _ = self.load_radiation(idx)
            if radiation_pca_dim > 0:
                if pca_model is None:
                    raise ValueError("pca_model must be provided when radiation_pca_dim > 0")
                flux = pca_model.transform(flux.reshape(1, -1)).squeeze(0)
            features.append(flux)
        features.append(rest)

        return np.concatenate(features)

    def load_dataset(
        self,
        indices: List[int] = None,
        include_radiation: bool = True,
        radiation_pca_dim: int = 0,
        pca_fit_indices: Optional[List[int]] = None,
        pca_model: Optional[Any] = None,
        pca_save_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        if indices is None:
            indices = list(range(1, self.n_samples + 1))

        X_list, T_list = [], []
        coords, comp_ids = None, None

        if include_radiation and radiation_pca_dim > 0:
            solar_list, rest_list, radiation_list = [], [], []
            for i, idx in enumerate(indices):
                if (i + 1) % 50 == 0:
                    print(f"Loading sample {i+1}/{len(indices)}...", flush=True)

                solar, rest = self._load_base_inputs(idx)
                solar_list.append(solar)
                rest_list.append(rest)

                flux, _ = self.load_radiation(idx)
                radiation_list.append(flux)

                temp, coord, comp_id = self.load_temperature(idx)
                T_list.append(temp)

                if coords is None:
                    coords = coord
                    comp_ids = comp_id

            radiation_matrix = np.vstack(radiation_list)
            if pca_model is None:
                if PCA is None:
                    raise ImportError("scikit-learn is required for radiation PCA.")
                if pca_fit_indices is None:
                    fit_mask = np.ones(len(indices), dtype=bool)
                else:
                    fit_set = set(pca_fit_indices)
                    fit_mask = np.array([idx in fit_set for idx in indices], dtype=bool)
                    if not np.any(fit_mask):
                        raise ValueError("pca_fit_indices does not overlap with provided indices.")

                pca_model = PCA(n_components=radiation_pca_dim)
                pca_model.fit(radiation_matrix[fit_mask])
                self.radiation_pca_model = pca_model
                if pca_save_path is not None:
                    save_path = Path(pca_save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "wb") as f:
                        pickle.dump(pca_model, f)
            else:
                self.radiation_pca_model = pca_model

            radiation_pca = pca_model.transform(radiation_matrix)
            for i in range(len(indices)):
                X_list.append(
                    np.concatenate([solar_list[i], radiation_pca[i], rest_list[i]])
                )
        else:
            for i, idx in enumerate(indices):
                if (i + 1) % 50 == 0:
                    print(f"Loading sample {i+1}/{len(indices)}...", flush=True)

                X_list.append(
                    self.load_all_inputs(
                        idx,
                        include_radiation=include_radiation,
                    )
                )
                temp, coord, comp_id = self.load_temperature(idx)
                T_list.append(temp)

                if coords is None:
                    coords = coord
                    comp_ids = comp_id

        return {
            'X': np.array(X_list),
            'T': np.array(T_list),
            'coords': coords,
            'comp_ids': comp_ids
        }


def train_val_test_split(X: np.ndarray, T: np.ndarray,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Split samples into training, validation, and test sets.

    Args:
        X: Input data with shape (n_samples, input_dim).
        T: Output data with shape (n_samples, output_dim).
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed.

    Returns:
        Dictionary containing the split arrays and indices.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Seed the random generator and shuffle sample indices.
    np.random.seed(seed)
    indices = np.random.permutation(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'X_train': X[train_idx],
        'T_train': T[train_idx],
        'X_val': X[val_idx],
        'T_val': T[val_idx],
        'X_test': X[test_idx],
        'T_test': T[test_idx],
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


def stratified_spatial_sample(
    coords: np.ndarray,
    sample_ratio: float = 0.2,
    grid_size: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide the 3D domain into cells and sample a fraction of each cell.

    Args:
        coords: Coordinate array with shape (n_points, 3).
        sample_ratio: Fraction of points to sample.
        grid_size: Number of grid divisions along each axis.
        seed: Random seed.

    Returns:
        train_point_idx: Indices of sampled training points.
        test_point_idx: Indices of all points used for evaluation.
    """
    np.random.seed(seed)
    n_points = len(coords)

    # Normalize coordinates to [0, grid_size).
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8) * (grid_size - 0.001)

    # Identify the grid cell containing each point.
    grid_idx = coords_norm.astype(int)
    cell_id = grid_idx[:, 0] * grid_size**2 + grid_idx[:, 1] * grid_size + grid_idx[:, 2]

    # Sample a fraction of the points in each cell.
    unique_cells = np.unique(cell_id)
    train_indices = []

    for cell in unique_cells:
        cell_points = np.where(cell_id == cell)[0]
        n_sample = max(1, int(len(cell_points) * sample_ratio))
        sampled = np.random.choice(cell_points, n_sample, replace=False)
        train_indices.extend(sampled)

    train_point_idx = np.array(sorted(train_indices))
    test_point_idx = np.arange(n_points)  # Evaluate on all points.

    return train_point_idx, test_point_idx


def stratified_component_sample(
    comp_ids: np.ndarray,
    sample_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a fraction of the points in each component.

    Args:
        comp_ids: Component IDs for each node, with shape (n_points,).
        sample_ratio: Fraction of points to sample.
        seed: Random seed.

    Returns:
        train_point_idx: Indices of sampled training points.
        test_point_idx: Indices of all points used for evaluation.
    """
    np.random.seed(seed)
    n_points = len(comp_ids)

    unique_comps = np.unique(comp_ids)
    train_indices = []

    for comp_id in unique_comps:
        comp_points = np.where(comp_ids == comp_id)[0]
        n_sample = max(1, int(len(comp_points) * sample_ratio))
        sampled = np.random.choice(comp_points, n_sample, replace=False)
        train_indices.extend(sampled)

    train_point_idx = np.array(sorted(train_indices))
    test_point_idx = np.arange(n_points)

    return train_point_idx, test_point_idx
