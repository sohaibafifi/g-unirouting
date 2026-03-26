from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from encoder.encoder_probe import _quantile_states


CORE_CONCEPT_NAMES: List[str] = [
    "instance_compactness_state",
    "load_concentration_state",
    "distance_budget_pressure_state",
]

CONCEPT_DISPLAY_NAMES: Dict[str, str] = {
    "instance_compactness_state": "compactness",
    "spatial_clustering_state": "clustering",
    "outlier_presence_state": "outliers",
    "load_concentration_state": "load concentration",
    "linehaul_backhaul_balance_state": "lh/bh balance",
    "capacity_pressure_prior_state": "capacity prior",
    "tw_density_state": "tw density",
    "tw_width_profile_state": "tw width",
    "distance_budget_pressure_state": "distance budget",
    "combined_constraint_tension_state": "combined tension",
}

CONCEPT_CLASS_ORDERS: Dict[str, List[str]] = {
    "instance_compactness_state": ["compact", "medium", "spread"],
    "spatial_clustering_state": ["diffuse", "clustered", "highly_clustered"],
    "outlier_presence_state": ["no_remote_outlier", "has_remote_outlier"],
    "load_concentration_state": ["balanced_load", "mixed_load", "concentrated_load"],
    "linehaul_backhaul_balance_state": [
        "linehaul_only",
        "linehaul_dominant",
        "balanced_mixed",
        "backhaul_dominant",
    ],
    "capacity_pressure_prior_state": ["low_pressure", "medium_pressure", "high_pressure"],
    "tw_density_state": ["no_tw", "sparse_tw", "medium_tw", "dense_tw"],
    "tw_width_profile_state": ["no_tw", "tight_windows", "mixed_windows", "wide_windows"],
    "distance_budget_pressure_state": ["no_limit", "tight_budget", "medium_budget", "loose_budget"],
    "combined_constraint_tension_state": ["low_tension", "medium_tension", "high_tension"],
}


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-6)


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * total) - (n + 1.0) / n)


def _best_spatial_clustering_score(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    unique_points = np.unique(points, axis=0)
    max_clusters = min(4, unique_points.shape[0] - 1)
    if max_clusters < 2:
        return 0.0

    best_score = float("-inf")
    for n_clusters in range(2, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        labels = model.fit_predict(points)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(points, labels, metric="euclidean"))
        best_score = max(best_score, score)
    return 0.0 if best_score == float("-inf") else best_score


def _remote_outlier_score(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        return 1.0
    centroid = np.mean(points, axis=0, keepdims=True)
    distances = np.linalg.norm(points - centroid, axis=1)
    median_distance = float(np.median(distances))
    if median_distance <= 1e-9:
        return 1.0
    return float(np.max(distances) / median_distance)


def _binary_high_tail_state(
    values: np.ndarray,
    positive_label: str,
    negative_label: str,
    quantile: float = 0.75,
) -> List[str]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    threshold = float(np.quantile(values, quantile))
    if np.allclose(values, values[0]):
        return [negative_label for _ in range(values.shape[0])]
    positive_mask = values > threshold
    if not np.any(positive_mask):
        positive_mask = values >= threshold
    if np.all(positive_mask):
        threshold = float(np.quantile(values, 0.5))
        positive_mask = values > threshold
        if not np.any(positive_mask):
            positive_mask = values >= threshold
    return [
        positive_label if bool(flag) else negative_label
        for flag in positive_mask.tolist()
    ]


def _minmax(values: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
    normalized = np.zeros(values.shape[0], dtype=np.float64)
    if active_mask is None:
        active_mask = np.ones(values.shape[0], dtype=bool)
    indices = np.flatnonzero(active_mask)
    if indices.size == 0:
        return normalized
    active_values = values[indices]
    low = float(np.min(active_values))
    high = float(np.max(active_values))
    if np.isclose(low, high):
        normalized[indices] = 0.5
        return normalized
    normalized[indices] = (active_values - low) / (high - low)
    return normalized


def _summarize_raw(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _mean_active_or_zero(values: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=np.float64)
    for idx in range(values.shape[0]):
        row_mask = active_mask[idx]
        if np.any(row_mask):
            out[idx] = float(np.mean(values[idx][row_mask]))
    return out


def compute_concept_bank(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    metadata: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    node_np = _to_numpy(node_features)
    global_np = _to_numpy(global_features)

    customers_xy = node_np[:, 1:, :2]
    depot_xy = node_np[:, 0, :2]
    demand_linehaul = node_np[:, 1:, 2]
    demand_backhaul = node_np[:, 1:, 3]
    total_customer_demand = demand_linehaul + demand_backhaul
    service_time = node_np[:, 1:, 6]
    tw_start = node_np[:, 1:, 4]
    tw_end = node_np[:, 1:, 5]
    tw_active_mask = np.isfinite(tw_end)

    depot_to_customer = np.linalg.norm(customers_xy - depot_xy[:, None, :], axis=-1)
    mean_depot_distance = depot_to_customer.mean(axis=1)
    max_depot_distance = depot_to_customer.max(axis=1)
    spatial_clustering = np.asarray(
        [_best_spatial_clustering_score(points) for points in customers_xy], dtype=np.float64
    )
    remote_outlier_score = np.asarray(
        [_remote_outlier_score(points) for points in customers_xy], dtype=np.float64
    )

    total_linehaul = demand_linehaul.sum(axis=1)
    total_backhaul = demand_backhaul.sum(axis=1)
    total_demand = total_customer_demand.sum(axis=1)
    backhaul_share = _safe_ratio(total_backhaul, total_demand)
    load_concentration = np.asarray(
        [_gini(sample) for sample in total_customer_demand], dtype=np.float64
    )

    vehicle_capacity = global_np[:, 0]
    capacity_pressure = _safe_ratio(total_demand, vehicle_capacity)

    tw_density = tw_active_mask.mean(axis=1)
    tw_width = np.where(tw_active_mask, tw_end - tw_start, np.nan)
    tw_flexibility = _mean_active_or_zero(
        tw_width / np.maximum(service_time, 1e-6),
        tw_active_mask,
    )

    distance_limit = global_np[:, 3]
    has_distance_limit = np.isfinite(distance_limit)
    distance_budget_ratio = np.full(distance_limit.shape[0], np.nan, dtype=np.float64)
    distance_budget_ratio[has_distance_limit] = (
        distance_limit[has_distance_limit]
        / np.maximum(2.0 * max_depot_distance[has_distance_limit], 1e-6)
    )

    has_time_windows = np.asarray(
        [bool(meta["flags"]["time_windows"]) for meta in metadata], dtype=bool
    )

    concept_states: Dict[str, List[str]] = {}
    concept_raw_values: Dict[str, np.ndarray] = {
        "instance_compactness_state": mean_depot_distance,
        "spatial_clustering_state": spatial_clustering,
        "outlier_presence_state": remote_outlier_score,
        "load_concentration_state": load_concentration,
        "capacity_pressure_prior_state": capacity_pressure,
        "tw_density_state": tw_density,
        "tw_width_profile_state": np.nan_to_num(tw_flexibility, nan=0.0, posinf=0.0, neginf=0.0),
        "distance_budget_pressure_state": np.nan_to_num(
            distance_budget_ratio, nan=0.0, posinf=0.0, neginf=0.0
        ),
    }

    concept_states["instance_compactness_state"] = _quantile_states(
        values=mean_depot_distance,
        active_mask=np.ones_like(mean_depot_distance, dtype=bool),
        labels=["compact", "medium", "spread"],
        off_label="medium",
    )
    concept_states["spatial_clustering_state"] = _quantile_states(
        values=spatial_clustering,
        active_mask=np.ones_like(spatial_clustering, dtype=bool),
        labels=["diffuse", "clustered", "highly_clustered"],
        off_label="clustered",
    )
    concept_states["outlier_presence_state"] = _binary_high_tail_state(
        values=remote_outlier_score,
        positive_label="has_remote_outlier",
        negative_label="no_remote_outlier",
        quantile=0.75,
    )
    concept_states["load_concentration_state"] = _quantile_states(
        values=load_concentration,
        active_mask=np.ones_like(load_concentration, dtype=bool),
        labels=["balanced_load", "mixed_load", "concentrated_load"],
        off_label="mixed_load",
    )

    balance_states: List[str] = []
    for linehaul, backhaul in zip(total_linehaul.tolist(), total_backhaul.tolist()):
        if backhaul <= 1e-6:
            balance_states.append("linehaul_only")
            continue
        total = max(linehaul + backhaul, 1e-6)
        share = backhaul / total
        if share < 0.33:
            balance_states.append("linehaul_dominant")
        elif share > 0.67:
            balance_states.append("backhaul_dominant")
        else:
            balance_states.append("balanced_mixed")
    concept_states["linehaul_backhaul_balance_state"] = balance_states
    concept_raw_values["linehaul_backhaul_balance_state"] = backhaul_share

    concept_states["capacity_pressure_prior_state"] = _quantile_states(
        values=capacity_pressure,
        active_mask=np.ones_like(capacity_pressure, dtype=bool),
        labels=["low_pressure", "medium_pressure", "high_pressure"],
        off_label="medium_pressure",
    )
    concept_states["tw_density_state"] = _quantile_states(
        values=tw_density,
        active_mask=has_time_windows,
        labels=["sparse_tw", "medium_tw", "dense_tw"],
        off_label="no_tw",
    )
    concept_states["tw_width_profile_state"] = _quantile_states(
        values=np.nan_to_num(tw_flexibility, nan=0.0, posinf=0.0, neginf=0.0),
        active_mask=has_time_windows,
        labels=["tight_windows", "mixed_windows", "wide_windows"],
        off_label="no_tw",
    )
    concept_states["distance_budget_pressure_state"] = _quantile_states(
        values=np.nan_to_num(distance_budget_ratio, nan=0.0, posinf=0.0, neginf=0.0),
        active_mask=has_distance_limit,
        labels=["tight_budget", "medium_budget", "loose_budget"],
        off_label="no_limit",
    )

    capacity_component = _minmax(capacity_pressure)
    tw_component = _minmax(
        1.0 / np.maximum(np.nan_to_num(tw_flexibility, nan=0.0), 1e-6),
        active_mask=has_time_windows,
    )
    distance_component = _minmax(
        1.0 / np.maximum(np.nan_to_num(distance_budget_ratio, nan=0.0), 1e-6),
        active_mask=has_distance_limit,
    )
    combined_tension = (capacity_component + tw_component + distance_component) / 3.0
    concept_raw_values["combined_constraint_tension_state"] = combined_tension
    concept_states["combined_constraint_tension_state"] = _quantile_states(
        values=combined_tension,
        active_mask=np.ones_like(combined_tension, dtype=bool),
        labels=["low_tension", "medium_tension", "high_tension"],
        off_label="medium_tension",
    )

    core_signatures = [
        "+".join(concept_states[name][idx] for name in CORE_CONCEPT_NAMES)
        for idx in range(node_np.shape[0])
    ]

    return {
        "concept_states": concept_states,
        "concept_raw_values": concept_raw_values,
        "concept_display_names": CONCEPT_DISPLAY_NAMES,
        "concept_class_orders": CONCEPT_CLASS_ORDERS,
        "core_concept_names": CORE_CONCEPT_NAMES,
        "core_concept_signatures": core_signatures,
        "raw_value_summary": {
            name: _summarize_raw(values)
            for name, values in sorted(concept_raw_values.items())
        },
    }
