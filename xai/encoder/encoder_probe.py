from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from engine.decode_ops import variant_metadata_from_inputs
from engine.model_io import ModelLoader


def _quantile_states(
    values: np.ndarray,
    active_mask: np.ndarray,
    labels: Sequence[str],
    off_label: str,
) -> List[str]:
    states = np.full(values.shape[0], off_label, dtype=object)
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        return states.astype(str).tolist()

    active_values = values[active_idx]
    if np.allclose(active_values, active_values[0]):
        states[active_idx] = labels[min(len(labels) // 2, len(labels) - 1)]
        return states.astype(str).tolist()

    boundaries = np.quantile(
        active_values,
        np.linspace(0.0, 1.0, len(labels) + 1)[1:-1],
    )
    boundaries = np.unique(boundaries)
    bin_ids = np.digitize(active_values, boundaries, right=False)
    for local_idx, global_idx in enumerate(active_idx):
        states[global_idx] = labels[min(int(bin_ids[local_idx]), len(labels) - 1)]
    return states.astype(str).tolist()


def _encoder_node_embeddings(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _pool_node_embeddings(node_embeddings: torch.Tensor, mode: str) -> np.ndarray:
    if mode == "mean":
        pooled = node_embeddings.mean(dim=1)
    elif mode == "meanstd":
        pooled = torch.cat(
            [
                node_embeddings.mean(dim=1),
                node_embeddings.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
    elif mode == "depot_meanstd":
        depot = node_embeddings[:, 0, :]
        customers = node_embeddings[:, 1:, :]
        pooled = torch.cat(
            [
                depot,
                customers.mean(dim=1),
                customers.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")
    return pooled.detach().cpu().numpy()


def _effective_rank(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    singular_values = singular_values[singular_values > 0]
    if singular_values.size == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))


def _stable_rank(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    max_sv = float(np.max(singular_values))
    if max_sv <= 0:
        return 0.0
    return float(np.sum(singular_values ** 2) / (max_sv ** 2))


def _mean_pdist(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    tensor = torch.from_numpy(matrix.astype(np.float32, copy=False))
    distances = torch.pdist(tensor, p=2)
    if distances.numel() == 0:
        return 0.0
    return float(distances.mean().item())


def _matrix_richness(node_embeddings: np.ndarray) -> Dict[str, float]:
    effective_ranks: List[float] = []
    stable_ranks: List[float] = []
    mean_pairwise_distances: List[float] = []
    centered_energy: List[float] = []

    for sample in node_embeddings:
        centered = sample - sample.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
        effective_ranks.append(_effective_rank(singular_values))
        stable_ranks.append(_stable_rank(singular_values))
        mean_pairwise_distances.append(_mean_pdist(centered))
        centered_energy.append(float(np.linalg.norm(centered, ord="fro") / max(sample.shape[0], 1)))

    return {
        "effective_rank_mean": float(np.mean(effective_ranks)),
        "effective_rank_std": float(np.std(effective_ranks)),
        "stable_rank_mean": float(np.mean(stable_ranks)),
        "stable_rank_std": float(np.std(stable_ranks)),
        "mean_pairwise_node_distance": float(np.mean(mean_pairwise_distances)),
        "centered_frobenius_per_node": float(np.mean(centered_energy)),
    }


def _label_silhouette(features: np.ndarray, labels: Sequence[str], seed: int) -> float:
    unique = sorted(set(labels))
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    sample_size = min(len(labels), 1000)
    return float(
        silhouette_score(
            scaled,
            labels,
            metric="euclidean",
            sample_size=sample_size,
            random_state=seed,
        )
    )


def _centroid_margin(features: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    classes = sorted(set(labels))
    if len(classes) < 2:
        return {
            "between_centroid_distance": float("nan"),
            "within_centroid_distance": float("nan"),
            "margin_ratio": float("nan"),
        }

    centroids: Dict[str, np.ndarray] = {}
    within_distances: List[float] = []
    for class_name in classes:
        mask = np.array([label == class_name for label in labels], dtype=bool)
        class_features = features[mask]
        centroid = class_features.mean(axis=0)
        centroids[class_name] = centroid
        within_distances.extend(np.linalg.norm(class_features - centroid, axis=1).tolist())

    between_distances: List[float] = []
    for idx, class_name in enumerate(classes):
        for other_name in classes[idx + 1:]:
            between_distances.append(
                float(np.linalg.norm(centroids[class_name] - centroids[other_name]))
            )

    within_mean = float(np.mean(within_distances)) if within_distances else float("nan")
    between_mean = float(np.mean(between_distances)) if between_distances else float("nan")
    return {
        "between_centroid_distance": between_mean,
        "within_centroid_distance": within_mean,
        "margin_ratio": (
            float(between_mean / within_mean)
            if np.isfinite(within_mean) and within_mean > 0
            else float("nan")
        ),
    }


def _kmeans_metrics(features: np.ndarray, k: int, seed: int) -> Dict[str, float]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    cluster_labels = model.fit_predict(scaled)
    metrics: Dict[str, float] = {
        "k": float(k),
        "inertia": float(model.inertia_),
    }
    if len(set(cluster_labels)) >= 2:
        sample_size = min(len(cluster_labels), 1000)
        metrics["silhouette"] = float(
            silhouette_score(
                scaled,
                cluster_labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=seed,
            )
        )
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(scaled, cluster_labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(scaled, cluster_labels))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")
    return metrics


def _label_aligned_kmeans(
    features: np.ndarray, labels: Sequence[str], seed: int
) -> Dict[str, float]:
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=len(unique_labels), n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(scaled)
    return {
        "k": float(len(unique_labels)),
        "inertia": float(kmeans.inertia_),
        "adjusted_rand": float(adjusted_rand_score(labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(labels, cluster_labels)),
        "homogeneity": float(homogeneity_score(labels, cluster_labels)),
        "completeness": float(completeness_score(labels, cluster_labels)),
        "v_measure": float(v_measure_score(labels, cluster_labels)),
    }


def _cross_validated_probe(
    features: np.ndarray,
    labels: Sequence[str],
    seed: int,
    binary: bool,
) -> Dict[str, float]:
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(list(labels))
    class_counts = Counter(encoded)
    if len(class_counts) < 2:
        return {}

    min_class_count = min(class_counts.values())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return {}

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if binary else None,
            random_state=seed,
        ),
    )

    predictions = cross_val_predict(probe, features, encoded, cv=splitter, method="predict")
    result: Dict[str, float] = {
        "accuracy": float(accuracy_score(encoded, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(encoded, predictions)),
        "macro_f1": float(f1_score(encoded, predictions, average="macro")),
    }
    if binary:
        probabilities = cross_val_predict(
            probe, features, encoded, cv=splitter, method="predict_proba"
        )[:, 1]
        result["roc_auc"] = float(roc_auc_score(encoded, probabilities))
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantitative probe of encoder latent spaces focused on constraint separation."
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument(
        "--pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="logs/xai/encoder_probe.json")
    parser.add_argument("--artifacts-output", default=None)
    return parser


def compute_probe_bundle(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ModelLoader.preflight_check()
    loader = ModelLoader(args)
    config, _, checkpoint_path = loader.resolve()
    model = loader.load_model()
    problem = config.get_problem()

    dataset = problem.dataset(
        graph_size=config.graph_size,
        num_samples=args.num_samples,
        variant="mtvrp",
        device=config.device,
    )
    node_features = dataset.node_features.to(config.device)
    global_features = dataset.global_features.to(config.device)
    metadata = variant_metadata_from_inputs(node_features, global_features)

    with torch.inference_mode():
        encoder_output = model.encoder((node_features, global_features))
        node_embeddings = _encoder_node_embeddings(encoder_output)

    feature_matrix = _pool_node_embeddings(node_embeddings, args.pooling)
    node_matrix = node_embeddings.detach().cpu().numpy()

    total_linehaul = node_features[:, 1:, 2].sum(dim=1).detach().cpu().numpy()
    total_backhaul = node_features[:, 1:, 3].sum(dim=1).detach().cpu().numpy()
    total_demand = total_linehaul + total_backhaul
    vehicle_capacity = global_features[:, 0].detach().cpu().numpy()
    load_ratio = total_demand / np.maximum(vehicle_capacity, 1e-6)

    customers = node_features[:, 1:, :2]
    depot = node_features[:, 0:1, :2]
    max_depot_distance = (
        torch.cdist(customers, depot).amax(dim=1).squeeze(-1).detach().cpu().numpy()
    )
    distance_limit = global_features[:, 3].detach().cpu().numpy()
    finite_limit_mask = np.isfinite(distance_limit)
    distance_tightness = np.full_like(distance_limit, np.nan, dtype=np.float64)
    distance_tightness[finite_limit_mask] = (
        distance_limit[finite_limit_mask]
        / np.maximum(2.0 * max_depot_distance[finite_limit_mask], 1e-6)
    )

    tw_length = (
        (node_features[:, 1:, 5] - node_features[:, 1:, 4]).mean(dim=1).detach().cpu().numpy()
    )
    service_time = node_features[:, 1:, 6].mean(dim=1).detach().cpu().numpy()
    temporal_tightness = tw_length / np.maximum(service_time, 1e-6)

    constraint_signatures: List[str] = []
    primitive_constraint_flags: Dict[str, List[str]] = defaultdict(list)
    grouped_constraint_flags: Dict[str, List[str]] = defaultdict(list)
    family_states: Dict[str, List[str]] = defaultdict(list)
    for meta in metadata:
        active_constraints = sorted(str(item) for item in meta["active_constraints"])
        constraint_signatures.append("+".join(active_constraints))
        for flag_name, flag_value in meta["flags"].items():
            primitive_constraint_flags[flag_name].append("on" if flag_value else "off")

        grouped_constraint_flags["route_structure"].append(
            "on"
            if bool(meta["flags"]["open_route"])
            or bool(meta["flags"]["backhaul"])
            or bool(meta["flags"]["mixed_backhaul"])
            else "off"
        )
        grouped_constraint_flags["space_distance"].append(
            "on" if bool(meta["flags"]["distance_limit"]) else "off"
        )
        grouped_constraint_flags["time_windows_service"].append(
            "on" if bool(meta["flags"]["time_windows"]) else "off"
        )

        family_states["route_openness_state"].append(
            "open_route" if bool(meta["flags"]["open_route"]) else "closed_route"
        )

        flow_state = "linehaul_only"
        if bool(meta["flags"]["mixed_backhaul"]) and bool(meta["flags"]["backhaul"]):
            flow_state = "mixed_backhaul"
        elif bool(meta["flags"]["backhaul"]):
            flow_state = "backhaul"
        family_states["flow_structure_state"].append(flow_state)

    family_states["geometry_state"] = _quantile_states(
        values=max_depot_distance,
        active_mask=np.ones_like(max_depot_distance, dtype=bool),
        labels=["compact_geometry", "medium_geometry", "spread_geometry"],
        off_label="medium_geometry",
    )

    family_states["capacity_demands_state"] = _quantile_states(
        values=load_ratio,
        active_mask=np.ones_like(load_ratio, dtype=bool),
        labels=["load_low", "load_medium", "load_high"],
        off_label="load_low",
    )
    family_states["distance_limit_state"] = _quantile_states(
        values=distance_tightness,
        active_mask=finite_limit_mask,
        labels=["tight_limit", "medium_limit", "large_limit"],
        off_label="no_limit",
    )
    family_states["time_windows_service_state"] = _quantile_states(
        values=temporal_tightness,
        active_mask=np.array(
            [bool(meta["flags"]["time_windows"]) for meta in metadata], dtype=bool
        ),
        labels=["tight_tw", "medium_tw", "large_tw"],
        off_label="no_tw",
    )

    unique_constraint_signatures = sorted(set(constraint_signatures))

    clustering_k_values = list(
        range(
            2,
            min(
                max(args.max_k, 2),
                feature_matrix.shape[0] - 1,
            )
            + 1,
        )
    )
    clustering_sweep = [
        _kmeans_metrics(feature_matrix, k=k_value, seed=args.seed)
        for k_value in clustering_k_values
    ]

    best_silhouette = max(
        (row for row in clustering_sweep if np.isfinite(row.get("silhouette", np.nan))),
        key=lambda row: row["silhouette"],
        default=None,
    )

    primitive_flag_probes: Dict[str, Dict[str, Any]] = {}
    for flag_name, labels in sorted(primitive_constraint_flags.items()):
        primitive_flag_probes[flag_name] = {
            "silhouette": _label_silhouette(feature_matrix, labels, seed=args.seed),
            "centroid_margin": _centroid_margin(feature_matrix, labels),
            "linear_probe": _cross_validated_probe(
                feature_matrix, labels, seed=args.seed, binary=True
            ),
            "class_balance": dict(Counter(labels)),
        }

    grouped_flag_probes: Dict[str, Dict[str, Any]] = {}
    for flag_name, labels in sorted(grouped_constraint_flags.items()):
        grouped_flag_probes[flag_name] = {
            "silhouette": _label_silhouette(feature_matrix, labels, seed=args.seed),
            "centroid_margin": _centroid_margin(feature_matrix, labels),
            "linear_probe": _cross_validated_probe(
                feature_matrix, labels, seed=args.seed, binary=True
            ),
            "class_balance": dict(Counter(labels)),
        }

    family_state_probes: Dict[str, Dict[str, Any]] = {}
    for family_name, labels in sorted(family_states.items()):
        is_binary = len(set(labels)) == 2
        family_state_probes[family_name] = {
            "silhouette": _label_silhouette(feature_matrix, labels, seed=args.seed),
            "centroid_margin": _centroid_margin(feature_matrix, labels),
            "linear_probe": _cross_validated_probe(
                feature_matrix, labels, seed=args.seed, binary=is_binary
            ),
            "class_balance": dict(Counter(labels)),
        }

    report = {
        "config": {
            "problem": str(config.problem),
            "graph_size": int(config.graph_size),
            "encoder_name": str(config.encoder.__name__),
            "decoder_name": str(config.decoder.__name__),
            "config_repr": repr(config),
            "checkpoint_path": str(checkpoint_path),
            "pooling": args.pooling,
            "num_samples": int(args.num_samples),
            "sampling_mode": "mtvrp_mixture",
        },
        "dataset": {
            "num_instances": int(feature_matrix.shape[0]),
            "num_constraint_signatures": int(len(unique_constraint_signatures)),
            "constraint_signature_counts": dict(Counter(constraint_signatures)),
        },
        "matrix_richness": _matrix_richness(node_matrix),
        "constraint_signature_separation": {
            "label_silhouette": _label_silhouette(
                feature_matrix, constraint_signatures, seed=args.seed
            ),
            "centroid_margin": _centroid_margin(feature_matrix, constraint_signatures),
            "linear_probe": _cross_validated_probe(
                feature_matrix, constraint_signatures, seed=args.seed, binary=False
            ),
            "kmeans_aligned": _label_aligned_kmeans(
                feature_matrix, constraint_signatures, seed=args.seed
            ),
        },
        "constraint_family_state_separation": family_state_probes,
        "constraint_group_separation": grouped_flag_probes,
        "constraint_flag_separation": primitive_flag_probes,
        "clustering_sweep": clustering_sweep,
        "best_k_by_silhouette": best_silhouette,
    }
    artifacts = {
        "pooled_features": feature_matrix.astype(np.float32, copy=False),
        "constraint_signatures": np.asarray(constraint_signatures),
        "primitive_flags": {
            name: np.asarray(values)
            for name, values in sorted(primitive_constraint_flags.items())
        },
        "grouped_flags": {
            name: np.asarray(values)
            for name, values in sorted(grouped_constraint_flags.items())
        },
        "family_states": {
            name: np.asarray(values)
            for name, values in sorted(family_states.items())
        },
    }
    return report, artifacts


def compute_probe_report(args: argparse.Namespace) -> Dict[str, Any]:
    report, _ = compute_probe_bundle(args)
    return report


def write_probe_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_probe_artifacts(artifacts: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: Dict[str, Any] = {
        "pooled_features": artifacts["pooled_features"],
        "constraint_signatures": artifacts["constraint_signatures"],
    }
    for name, values in sorted((artifacts.get("primitive_flags") or {}).items()):
        flattened[f"primitive__{name}"] = values
    for name, values in sorted((artifacts.get("grouped_flags") or {}).items()):
        flattened[f"grouped__{name}"] = values
    for name, values in sorted((artifacts.get("family_states") or {}).items()):
        flattened[f"family__{name}"] = values
    np.savez_compressed(path, **flattened)
    return path


def main() -> None:
    args = _build_parser().parse_args()
    report, artifacts = compute_probe_bundle(args)

    if args.artifacts_output:
        artifact_path = write_probe_artifacts(artifacts, args.artifacts_output)
        report["artifacts"] = {
            "pooled_features_path": str(artifact_path),
        }

    output_path = write_probe_report(report, args.output)

    print(
        f"Encoder probe saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} pooling={report['config']['pooling']} "
        f"instances={report['dataset']['num_instances']} "
        f"constraint_signatures={report['dataset']['num_constraint_signatures']}"
    )
    best_k = report["best_k_by_silhouette"]
    if best_k is not None:
        print(
            "Best k by silhouette: "
            f"k={int(best_k['k'])} "
            f"silhouette={best_k['silhouette']:.4f} "
            f"inertia={best_k['inertia']:.4f}"
        )
    print(
        "Constraint-signature linear probe: "
        f"{json.dumps(_json_ready(report['constraint_signature_separation']['linear_probe']), ensure_ascii=False)}"
    )


if __name__ == "__main__":
    main()
