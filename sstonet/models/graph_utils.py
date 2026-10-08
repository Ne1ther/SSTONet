# -*- coding: utf-8 -*-
"""Graph construction utilities for component-aware GNN."""

import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Union


def build_component_aware_graph(
    coords: np.ndarray,
    comp_ids: np.ndarray,
    k: int = 15,
    cross_component_radius: float = 2.0,
    return_edge_attr: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Build component-aware graph with k-NN within components and radius-based cross-component edges.

    Args:
        coords: Node coordinates (N, 3)
        comp_ids: Component IDs for each node (N,)
        k: Number of neighbors for intra-component k-NN
        cross_component_radius: Distance threshold for cross-component edges
        return_edge_attr: If True, also return edge distances

    Returns:
        edge_index: Edge indices (2, E) in COO format
        edge_attr: Edge distances (E,), only if return_edge_attr=True
    """
    N = len(coords)
    unique_comps = np.unique(comp_ids)
    src_list, dst_list = [], []

    # Intra-component k-NN edges (bidirectional)
    for comp in unique_comps:
        mask = comp_ids == comp
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        comp_coords = coords[indices]
        tree = KDTree(comp_coords)
        k_actual = min(k + 1, len(indices))  # +1 because query includes self
        _, neighbors = tree.query(comp_coords, k=k_actual)
        for i, local_neighbors in enumerate(neighbors):
            global_i = indices[i]
            for local_j in local_neighbors:
                if local_j != i:  # skip self-loop
                    global_j = indices[local_j]
                    # Add bidirectional edges for symmetric GCN
                    src_list.extend([global_i, global_j])
                    dst_list.extend([global_j, global_i])

    # Cross-component edges via radius search
    if len(unique_comps) > 1:
        full_tree = KDTree(coords)
        pairs = full_tree.query_pairs(r=cross_component_radius, output_type='ndarray')
        for i, j in pairs:
            if comp_ids[i] != comp_ids[j]:
                # Add bidirectional edges
                src_list.extend([i, j])
                dst_list.extend([j, i])

    # Remove duplicate edges
    edge_set = set(zip(src_list, dst_list))
    src_list, dst_list = zip(*edge_set) if edge_set else ([], [])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    if return_edge_attr:
        # Compute edge distances
        src, dst = edge_index[0], edge_index[1]
        edge_attr = np.linalg.norm(coords[src] - coords[dst], axis=1).astype(np.float32)
        return edge_index, edge_attr

    return edge_index
