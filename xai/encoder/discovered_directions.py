from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

from encoder.concept_bank import CONCEPT_DISPLAY_NAMES
from encoder.concept_probe import compute_concept_probe_bundle
from encoder.encoder_probe import _json_ready

PRIMITIVE_CONSTRAINT_DISPLAY_NAMES: Dict[str, str] = {
    "open_route": "open route",
    "backhaul": "backhaul",
    "mixed_backhaul": "mixed backhaul",
    "distance_limit": "distance limit",
    "time_windows": "time windows",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Study discovered latent directions in encoder representations by extracting "
            "principal components and correlating them with the existing concept bank."
        )
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument(
        "--output",
        default="logs/xai/encoder/graph/discovered_directions/probe.json",
    )
    parser.add_argument("--artifacts-output", default=None)
    return parser


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_float(value: Any) -> float | None:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _fit_pca(features: np.ndarray, seed: int, num_components: int) -> Tuple[PCA, np.ndarray]:
    n_components = max(
        2,
        min(
            int(num_components),
            int(features.shape[0]),
            int(features.shape[1]),
        ),
    )
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(scaled)
    return pca, scores


def _fit_ica(
    features: np.ndarray,
    seed: int,
    num_components: int,
) -> Tuple[FastICA, np.ndarray]:
    n_components = max(
        2,
        min(
            int(num_components),
            int(features.shape[0]),
            int(features.shape[1]),
        ),
    )
    scaled = StandardScaler().fit_transform(features)
    ica = FastICA(
        n_components=n_components,
        random_state=seed,
        whiten="unit-variance",
        max_iter=2000,
        tol=1e-4,
    )
    scores = ica.fit_transform(scaled)
    return ica, scores


def _summarize_method(
    method_name: str,
    scores: np.ndarray,
    concept_names: List[str],
    concept_raw_values: Dict[str, np.ndarray],
    top_components: int,
    component_vectors: np.ndarray,
    explained_variance_ratio: np.ndarray | None = None,
    display_names: Dict[str, str] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    display_names = dict(CONCEPT_DISPLAY_NAMES if display_names is None else display_names)
    correlation_matrix = np.full(
        (scores.shape[1], len(concept_names)),
        np.nan,
        dtype=np.float64,
    )
    for component_idx in range(scores.shape[1]):
        component_scores = scores[:, component_idx]
        for concept_idx, concept_name in enumerate(concept_names):
            correlation_matrix[component_idx, concept_idx] = _pearson_corr(
                component_scores,
                concept_raw_values[concept_name],
            )

    component_summaries: List[Dict[str, Any]] = []
    best_abs_per_component: List[float] = []
    for component_idx in range(scores.shape[1]):
        row = correlation_matrix[component_idx]
        finite = np.isfinite(row)
        if np.any(finite):
            ordered = sorted(
                (
                    (concept_names[idx], float(row[idx]), float(abs(row[idx])))
                    for idx in range(len(concept_names))
                    if np.isfinite(row[idx])
                ),
                key=lambda item: item[2],
                reverse=True,
            )
            best_name, best_corr, best_abs = ordered[0]
            top_concepts = [
                {
                    "concept_name": name,
                    "display_name": display_names.get(name, name),
                    "correlation": corr,
                    "abs_correlation": abs_corr,
                }
                for name, corr, abs_corr in ordered[:3]
            ]
        else:
            best_name, best_corr, best_abs = "", float("nan"), float("nan")
            top_concepts = []
        best_abs_per_component.append(best_abs)
        component_summaries.append(
            {
                "component_index": int(component_idx + 1),
                "explained_variance_ratio": (
                    float(explained_variance_ratio[component_idx])
                    if explained_variance_ratio is not None
                    else None
                ),
                "cumulative_explained_variance_ratio": (
                    float(np.cumsum(explained_variance_ratio)[component_idx])
                    if explained_variance_ratio is not None
                    else None
                ),
                "best_concept_name": best_name,
                "best_concept_display": display_names.get(best_name, best_name),
                "best_correlation": best_corr,
                "best_abs_correlation": best_abs,
                "top_concepts": top_concepts,
            }
        )

    concept_alignment: Dict[str, Dict[str, Any]] = {}
    best_concept_name = None
    best_concept_abs_corr = float("nan")
    best_concept_component = None
    for concept_idx, concept_name in enumerate(concept_names):
        col = correlation_matrix[:, concept_idx]
        finite = np.isfinite(col)
        if not np.any(finite):
            concept_alignment[concept_name] = {
                "best_component_index": None,
                "best_correlation": None,
                "best_abs_correlation": None,
                "display_name": display_names.get(concept_name, concept_name),
            }
            continue
        valid_indices = np.flatnonzero(finite)
        best_local_idx = max(valid_indices.tolist(), key=lambda idx: abs(float(col[idx])))
        best_corr = float(col[best_local_idx])
        best_abs_corr = float(abs(best_corr))
        concept_alignment[concept_name] = {
            "best_component_index": int(best_local_idx + 1),
            "best_correlation": best_corr,
            "best_abs_correlation": best_abs_corr,
            "display_name": display_names.get(concept_name, concept_name),
        }
        if not np.isfinite(best_concept_abs_corr) or best_abs_corr > best_concept_abs_corr:
            best_concept_abs_corr = best_abs_corr
            best_concept_name = concept_name
            best_concept_component = int(best_local_idx + 1)

    top_component_abs = [value for value in best_abs_per_component[:top_components] if np.isfinite(value)]
    all_component_abs = [value for value in best_abs_per_component if np.isfinite(value)]
    cumulative = np.cumsum(explained_variance_ratio) if explained_variance_ratio is not None else None
    summary = {
        "mean_best_abs_correlation_top_components": (
            float(np.mean(top_component_abs)) if top_component_abs else None
        ),
        "best_abs_correlation_overall": (
            float(np.max(all_component_abs)) if all_component_abs else None
        ),
        "num_components_abs_correlation_ge_0_3": int(sum(value >= 0.3 for value in all_component_abs)),
        "num_components_abs_correlation_ge_0_5": int(sum(value >= 0.5 for value in all_component_abs)),
        "top1_explained_variance_ratio": (
            float(explained_variance_ratio[0])
            if explained_variance_ratio is not None and explained_variance_ratio.size >= 1
            else None
        ),
        "top3_cumulative_explained_variance_ratio": (
            float(cumulative[min(2, cumulative.size - 1)])
            if cumulative is not None and cumulative.size >= 1
            else None
        ),
        "top5_cumulative_explained_variance_ratio": (
            float(cumulative[min(4, cumulative.size - 1)])
            if cumulative is not None and cumulative.size >= 1
            else None
        ),
        "strongest_concept_name": best_concept_name,
        "strongest_concept_display": (
            display_names.get(best_concept_name, best_concept_name)
            if best_concept_name
            else None
        ),
        "strongest_concept_component_index": best_concept_component,
    }

    payload = {
        "method": method_name,
        "num_components": int(scores.shape[1]),
        "top_components_used_for_summary": int(top_components),
        "summary": summary,
        "top_components": component_summaries[:top_components],
        "concept_alignment": concept_alignment,
    }
    artifacts = {
        "component_scores": scores.astype(np.float32, copy=False),
        "component_vectors": np.asarray(component_vectors, dtype=np.float32),
        "correlation_matrix": correlation_matrix.astype(np.float32, copy=False),
        "explained_variance_ratio": (
            np.asarray(explained_variance_ratio, dtype=np.float32)
            if explained_variance_ratio is not None
            else np.asarray([], dtype=np.float32)
        ),
        "cumulative_explained_variance_ratio": (
            np.cumsum(np.asarray(explained_variance_ratio, dtype=np.float32))
            if explained_variance_ratio is not None
            else np.asarray([], dtype=np.float32)
        ),
    }
    return payload, artifacts


def compute_discovered_direction_bundle(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    concept_args = SimpleNamespace(**vars(args))
    if not hasattr(concept_args, "max_k"):
        concept_args.max_k = 12
    concept_report, concept_artifacts = compute_concept_probe_bundle(concept_args)
    features = np.asarray(concept_artifacts["pooled_features"], dtype=np.float32)
    concept_raw_values = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in (concept_artifacts.get("concept_raw_values") or {}).items()
    }
    concept_names = sorted(concept_raw_values.keys())
    primitive_constraint_raw_values = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in (concept_artifacts.get("primitive_raw_values") or {}).items()
    }
    primitive_constraint_names = sorted(primitive_constraint_raw_values.keys())
    known_bank_names = primitive_constraint_names + concept_names
    known_bank_display_names = {
        **{name: PRIMITIVE_CONSTRAINT_DISPLAY_NAMES.get(name, name) for name in primitive_constraint_names},
        **{name: CONCEPT_DISPLAY_NAMES.get(name, name) for name in concept_names},
    }
    known_bank_raw_values = {
        **primitive_constraint_raw_values,
        **concept_raw_values,
    }
    top_components = min(int(args.top_components), int(args.num_components))

    pca, pca_scores = _fit_pca(
        features,
        seed=int(args.seed),
        num_components=int(args.num_components),
    )
    pca_payload, pca_artifacts = _summarize_method(
        method_name="pca",
        scores=pca_scores,
        concept_names=concept_names,
        concept_raw_values=concept_raw_values,
        top_components=min(top_components, pca_scores.shape[1]),
        component_vectors=np.asarray(pca.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )
    pca_constraint_payload, pca_constraint_artifacts = _summarize_method(
        method_name="pca",
        scores=pca_scores,
        concept_names=primitive_constraint_names,
        concept_raw_values=primitive_constraint_raw_values,
        top_components=min(top_components, pca_scores.shape[1]),
        component_vectors=np.asarray(pca.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        display_names=PRIMITIVE_CONSTRAINT_DISPLAY_NAMES,
    )
    pca_known_payload, pca_known_artifacts = _summarize_method(
        method_name="pca",
        scores=pca_scores,
        concept_names=known_bank_names,
        concept_raw_values=known_bank_raw_values,
        top_components=min(top_components, pca_scores.shape[1]),
        component_vectors=np.asarray(pca.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        display_names=known_bank_display_names,
    )

    ica, ica_scores = _fit_ica(
        features,
        seed=int(args.seed),
        num_components=int(args.num_components),
    )
    ica_payload, ica_artifacts = _summarize_method(
        method_name="ica",
        scores=ica_scores,
        concept_names=concept_names,
        concept_raw_values=concept_raw_values,
        top_components=min(top_components, ica_scores.shape[1]),
        component_vectors=np.asarray(ica.components_, dtype=np.float32),
        explained_variance_ratio=None,
    )
    ica_constraint_payload, ica_constraint_artifacts = _summarize_method(
        method_name="ica",
        scores=ica_scores,
        concept_names=primitive_constraint_names,
        concept_raw_values=primitive_constraint_raw_values,
        top_components=min(top_components, ica_scores.shape[1]),
        component_vectors=np.asarray(ica.components_, dtype=np.float32),
        explained_variance_ratio=None,
        display_names=PRIMITIVE_CONSTRAINT_DISPLAY_NAMES,
    )
    ica_known_payload, ica_known_artifacts = _summarize_method(
        method_name="ica",
        scores=ica_scores,
        concept_names=known_bank_names,
        concept_raw_values=known_bank_raw_values,
        top_components=min(top_components, ica_scores.shape[1]),
        component_vectors=np.asarray(ica.components_, dtype=np.float32),
        explained_variance_ratio=None,
        display_names=known_bank_display_names,
    )

    report = {
        "config": dict(concept_report["config"]),
        "dataset": dict(concept_report["dataset"]),
        "matrix_richness": dict(concept_report["matrix_richness"]),
        "concept_bank_reference": {
            "display_names": dict(concept_report["concept_bank"]["display_names"]),
            "raw_value_summary": dict(concept_report["concept_bank"]["raw_value_summary"]),
            "core_concepts": list(concept_report["concept_bank"]["core_concepts"]),
        },
        "constraint_reference": {
            "display_names": dict(PRIMITIVE_CONSTRAINT_DISPLAY_NAMES),
            "names": list(primitive_constraint_names),
        },
        "known_bank_reference": {
            "constraint_names": list(primitive_constraint_names),
            "concept_names": list(concept_names),
            "display_names": dict(known_bank_display_names),
        },
        "discovered_directions": pca_payload,
        "constraint_reference_alignment": pca_constraint_payload,
        "known_bank_alignment": pca_known_payload,
        "alternative_methods": {
            "ica": ica_payload,
            "constraint_reference_alignment": ica_constraint_payload,
            "known_bank_alignment": ica_known_payload,
        },
    }

    artifacts = {
        "pooled_features": features.astype(np.float32, copy=False),
        "component_scores": pca_artifacts["component_scores"],
        "component_vectors": pca_artifacts["component_vectors"],
        "explained_variance_ratio": pca_artifacts["explained_variance_ratio"],
        "cumulative_explained_variance_ratio": pca_artifacts["cumulative_explained_variance_ratio"],
        "correlation_matrix": pca_artifacts["correlation_matrix"],
        "ica_component_scores": ica_artifacts["component_scores"],
        "ica_component_vectors": ica_artifacts["component_vectors"],
        "ica_explained_variance_ratio": ica_artifacts["explained_variance_ratio"],
        "ica_cumulative_explained_variance_ratio": ica_artifacts["cumulative_explained_variance_ratio"],
        "ica_correlation_matrix": ica_artifacts["correlation_matrix"],
        "concept_names": np.asarray(concept_names),
        "concept_raw_values": concept_raw_values,
        "constraint_names": np.asarray(primitive_constraint_names),
        "constraint_raw_values": primitive_constraint_raw_values,
        "constraint_correlation_matrix": pca_constraint_artifacts["correlation_matrix"],
        "ica_constraint_correlation_matrix": ica_constraint_artifacts["correlation_matrix"],
        "known_bank_names": np.asarray(known_bank_names),
        "known_bank_types": np.asarray(
            ["constraint"] * len(primitive_constraint_names) + ["concept"] * len(concept_names)
        ),
        "known_bank_correlation_matrix": pca_known_artifacts["correlation_matrix"],
        "ica_known_bank_correlation_matrix": ica_known_artifacts["correlation_matrix"],
    }
    return report, artifacts


def write_discovered_direction_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_discovered_direction_artifacts(artifacts: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: Dict[str, Any] = {
        "pooled_features": artifacts["pooled_features"],
        "component_scores": artifacts["component_scores"],
        "component_vectors": artifacts["component_vectors"],
        "explained_variance_ratio": artifacts["explained_variance_ratio"],
        "cumulative_explained_variance_ratio": artifacts["cumulative_explained_variance_ratio"],
        "correlation_matrix": artifacts["correlation_matrix"],
        "ica_component_scores": artifacts["ica_component_scores"],
        "ica_component_vectors": artifacts["ica_component_vectors"],
        "ica_explained_variance_ratio": artifacts["ica_explained_variance_ratio"],
        "ica_cumulative_explained_variance_ratio": artifacts["ica_cumulative_explained_variance_ratio"],
        "ica_correlation_matrix": artifacts["ica_correlation_matrix"],
        "concept_names": artifacts["concept_names"],
        "constraint_names": artifacts["constraint_names"],
        "constraint_correlation_matrix": artifacts["constraint_correlation_matrix"],
        "ica_constraint_correlation_matrix": artifacts["ica_constraint_correlation_matrix"],
        "known_bank_names": artifacts["known_bank_names"],
        "known_bank_types": artifacts["known_bank_types"],
        "known_bank_correlation_matrix": artifacts["known_bank_correlation_matrix"],
        "ica_known_bank_correlation_matrix": artifacts["ica_known_bank_correlation_matrix"],
    }
    for name, values in sorted((artifacts.get("concept_raw_values") or {}).items()):
        flattened[f"concept_value__{name}"] = np.asarray(values, dtype=np.float32)
    for name, values in sorted((artifacts.get("constraint_raw_values") or {}).items()):
        flattened[f"constraint_value__{name}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(path, **flattened)
    return path


def main() -> None:
    args = _build_parser().parse_args()
    report, artifacts = compute_discovered_direction_bundle(args)

    if args.artifacts_output:
        artifact_path = write_discovered_direction_artifacts(artifacts, args.artifacts_output)
        report["artifacts"] = {"discovered_directions_path": str(artifact_path)}

    output_path = write_discovered_direction_report(report, args.output)
    summary = report["discovered_directions"]["summary"]
    print(
        f"Discovered-direction probe saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} pooling={report['config']['pooling']} "
        f"instances={report['dataset']['num_instances']} "
        f"mean_abs_corr={_safe_float(summary.get('mean_best_abs_correlation_top_components'))}"
    )


if __name__ == "__main__":
    main()
