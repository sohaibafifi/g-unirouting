from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from encoder.concept_bank import _best_spatial_clustering_score, _gini, _remote_outlier_score
from encoder.discovered_directions import _pearson_corr
from engine.decode_ops import variant_metadata_from_inputs
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect graph-level discovered directions that remain weakly explained by the "
            "known bank (constraints + concepts), and produce interpretation dossiers."
        )
    )
    parser.add_argument(
        "--input-json",
        default="logs/xai/encoder/graph/discovered_directions/comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/xai/encoder/graph/discovered_directions/unexplained_dossiers",
    )
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument("--known-threshold", type=float, default=0.3)
    parser.add_argument("--top-instances", type=int, default=5)
    return parser


def _safe_float(value: Any) -> float | None:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _fmt(value: Any, ndigits: int = 4) -> str:
    cast = _safe_float(value)
    if cast is None:
        return "-"
    return f"{cast:.{ndigits}f}"


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


def _finite_mean(values: np.ndarray) -> float | None:
    cast = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(cast)
    if not np.any(finite):
        return None
    return float(np.mean(cast[finite]))


def _angular_entropy(points: np.ndarray, depot: np.ndarray, bins: int = 8) -> float:
    rel = points - depot[None, :]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _tw_overlap_ratio(starts: np.ndarray, ends: np.ndarray, active_mask: np.ndarray) -> float:
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size < 2:
        return 0.0
    starts = starts[active_idx]
    ends = ends[active_idx]
    overlaps: List[float] = []
    for i in range(active_idx.size):
        for j in range(i + 1, active_idx.size):
            left = max(float(starts[i]), float(starts[j]))
            right = min(float(ends[i]), float(ends[j]))
            overlap = max(0.0, right - left)
            union = max(float(max(ends[i], ends[j]) - min(starts[i], starts[j])), 1e-6)
            overlaps.append(float(overlap / union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def _pairwise_mean_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    tensor = torch.from_numpy(points.astype(np.float32, copy=False))
    distances = torch.pdist(tensor, p=2)
    return float(distances.mean().item()) if distances.numel() else 0.0


def _auxiliary_descriptors(config_id: int, num_samples: int) -> Dict[str, np.ndarray]:
    config = Config.all()[config_id]
    torch.manual_seed(1234)
    np.random.seed(1234)
    dataset = config.get_problem().dataset(
        graph_size=config.graph_size,
        num_samples=num_samples,
        variant="mtvrp",
        device=config.device,
    )
    node_features = dataset.node_features.detach().cpu()
    global_features = dataset.global_features.detach().cpu()
    metadata = variant_metadata_from_inputs(node_features, global_features)

    node_np = node_features.numpy()
    global_np = global_features.numpy()
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
    std_x = np.std(customers_xy[:, :, 0], axis=1)
    std_y = np.std(customers_xy[:, :, 1], axis=1)

    linehaul_total = demand_linehaul.sum(axis=1)
    backhaul_total = demand_backhaul.sum(axis=1)
    total_demand = total_customer_demand.sum(axis=1)
    top1_demand_share = np.max(total_customer_demand, axis=1) / np.maximum(total_demand, 1e-6)
    active_constraint_count = np.asarray(
        [len(meta.get("active_constraints", [])) for meta in metadata],
        dtype=np.float32,
    )

    descriptors: Dict[str, np.ndarray] = {
        "max_depot_distance": depot_to_customer.max(axis=1).astype(np.float32),
        "depot_distance_std": depot_to_customer.std(axis=1).astype(np.float32),
        "radial_cv": (
            depot_to_customer.std(axis=1) / np.maximum(depot_to_customer.mean(axis=1), 1e-6)
        ).astype(np.float32),
        "pairwise_customer_distance_mean": np.asarray(
            [_pairwise_mean_distance(points) for points in customers_xy],
            dtype=np.float32,
        ),
        "angular_entropy_8": np.asarray(
            [_angular_entropy(points, depot) for points, depot in zip(customers_xy, depot_xy)],
            dtype=np.float32,
        ),
        "anisotropy_ratio": (
            np.maximum(std_x, std_y) / np.maximum(np.minimum(std_x, std_y), 1e-6)
        ).astype(np.float32),
        "linehaul_total": linehaul_total.astype(np.float32),
        "backhaul_total": backhaul_total.astype(np.float32),
        "top1_demand_share": top1_demand_share.astype(np.float32),
        "service_time_mean": service_time.mean(axis=1).astype(np.float32),
        "tw_overlap_ratio": np.asarray(
            [
                _tw_overlap_ratio(starts, ends, active)
                for starts, ends, active in zip(tw_start, tw_end, tw_active_mask)
            ],
            dtype=np.float32,
        ),
        "tw_start_mean_active": np.asarray(
            [
                float(np.mean(starts[active])) if np.any(active) else 0.0
                for starts, active in zip(tw_start, tw_active_mask)
            ],
            dtype=np.float32,
        ),
        "tw_end_mean_active": np.asarray(
            [
                float(np.mean(ends[active])) if np.any(active) else 0.0
                for ends, active in zip(tw_end, tw_active_mask)
            ],
            dtype=np.float32,
        ),
        "depot_time_limit": global_np[:, 4].astype(np.float32),
        "active_constraint_count": active_constraint_count.astype(np.float32),
        "remote_outlier_ratio": np.asarray(
            [_remote_outlier_score(points) for points in customers_xy],
            dtype=np.float32,
        ),
        "demand_gini_aux": np.asarray(
            [_gini(sample) for sample in total_customer_demand],
            dtype=np.float32,
        ),
        "spatial_clustering_aux": np.asarray(
            [_best_spatial_clustering_score(points) for points in customers_xy],
            dtype=np.float32,
        ),
    }
    return descriptors


def _load_artifacts(report: Dict[str, Any]) -> Dict[str, Any]:
    artifact_path = report.get("artifacts", {}).get("discovered_directions_path")
    if not artifact_path:
        raise FileNotFoundError("Missing discovered-direction artifact path in report.")
    loaded = np.load(Path(str(artifact_path)), allow_pickle=False)
    return {
        "component_scores": loaded["component_scores"].astype(np.float32),
        "ica_component_scores": loaded["ica_component_scores"].astype(np.float32),
        "known_bank_names": loaded["known_bank_names"].astype(str),
        "known_bank_types": loaded["known_bank_types"].astype(str),
        "known_bank_correlation_matrix": loaded["known_bank_correlation_matrix"].astype(np.float32),
        "ica_known_bank_correlation_matrix": loaded["ica_known_bank_correlation_matrix"].astype(np.float32),
    }


def _top_correlations(
    component_scores: np.ndarray,
    descriptors: Dict[str, np.ndarray],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, values in sorted(descriptors.items()):
        values = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(component_scores) & np.isfinite(values)
        if int(mask.sum()) < 3:
            continue
        corr = _pearson_corr(component_scores[mask], values[mask])
        if _safe_float(corr) is None:
            continue
        rows.append(
            {
                "descriptor": name,
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
            }
        )
    rows.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return rows[:limit]


def _top_bottom_gap(
    component_scores: np.ndarray,
    descriptors: Dict[str, np.ndarray],
    count: int,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    order = np.argsort(component_scores)
    low_idx = order[:count]
    high_idx = order[-count:]
    rows: List[Dict[str, Any]] = []
    for name, values in sorted(descriptors.items()):
        values = np.asarray(values, dtype=np.float32)
        high_mean = _finite_mean(values[high_idx])
        low_mean = _finite_mean(values[low_idx])
        if high_mean is None or low_mean is None:
            continue
        gap = float(high_mean - low_mean)
        rows.append({"descriptor": name, "top_minus_bottom": gap, "abs_gap": abs(gap)})
    rows.sort(key=lambda item: item["abs_gap"], reverse=True)
    return rows[:limit]


def _instance_slice(
    scores: np.ndarray,
    metadata: List[Dict[str, Any]],
    descriptors: Dict[str, np.ndarray],
    indices: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    key_descriptors = [
        "max_depot_distance",
        "angular_entropy_8",
        "top1_demand_share",
        "tw_overlap_ratio",
        "active_constraint_count",
    ]
    for idx in indices.tolist():
        item = {
            "instance_index": int(idx),
            "score": float(scores[idx]),
            "active_constraints": list(metadata[idx].get("active_constraints", [])),
            "flags": dict(metadata[idx].get("flags", {})),
            "descriptors": {
                name: _safe_float(np.asarray(descriptors[name], dtype=np.float32)[idx])
                for name in key_descriptors
                if name in descriptors
            },
        }
        rows.append(item)
    return rows


def _recreate_metadata(config_id: int, num_samples: int) -> List[Dict[str, Any]]:
    config = Config.all()[config_id]
    torch.manual_seed(1234)
    np.random.seed(1234)
    dataset = config.get_problem().dataset(
        graph_size=config.graph_size,
        num_samples=num_samples,
        variant="mtvrp",
        device=config.device,
    )
    return list(variant_metadata_from_inputs(dataset.node_features, dataset.global_features))


def _report_by_model(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(report["config"]["config_repr"]): report for report in (payload.get("reports") or [])}


def _build_dossier_for_report(
    row: Dict[str, Any],
    report: Dict[str, Any],
    top_components: int,
    known_threshold: float,
    top_instances: int,
) -> Dict[str, Any]:
    config_id = int(row["config_id"])
    num_samples = int(report["config"]["num_samples"])
    artifacts = _load_artifacts(report)
    descriptors = _auxiliary_descriptors(config_id, num_samples)
    metadata = _recreate_metadata(config_id, num_samples)

    output: Dict[str, Any] = {
        "config_id": config_id,
        "model": str(row["model"]),
        "threshold": float(known_threshold),
        "top_components_considered": int(top_components),
        "methods": {},
    }

    for method, score_key, corr_key in (
        ("pca", "component_scores", "known_bank_correlation_matrix"),
        ("ica", "ica_component_scores", "ica_known_bank_correlation_matrix"),
    ):
        scores = np.asarray(artifacts[score_key], dtype=np.float32)
        corr = np.asarray(artifacts[corr_key], dtype=np.float32)
        known_names = artifacts["known_bank_names"].astype(str).tolist()
        known_types = artifacts["known_bank_types"].astype(str).tolist()
        method_rows: List[Dict[str, Any]] = []
        n_use = min(int(top_components), int(scores.shape[1]))

        for component_idx in range(n_use):
            row_corr = corr[component_idx]
            finite = np.isfinite(row_corr)
            if np.any(finite):
                best_idx = max(np.flatnonzero(finite).tolist(), key=lambda idx: abs(float(row_corr[idx])))
                best_abs = float(abs(row_corr[best_idx]))
                best_name = known_names[best_idx]
                best_type = known_types[best_idx]
                best_corr = float(row_corr[best_idx])
            else:
                best_idx = -1
                best_abs = float("nan")
                best_name = ""
                best_type = ""
                best_corr = float("nan")

            if _safe_float(best_abs) is None or best_abs >= float(known_threshold):
                continue

            component_scores = scores[:, component_idx]
            order = np.argsort(component_scores)
            low_idx = order[: top_instances]
            high_idx = order[-top_instances:]
            method_rows.append(
                {
                    "component_index": int(component_idx + 1),
                    "best_known_name": best_name,
                    "best_known_type": best_type,
                    "best_known_correlation": best_corr,
                    "best_known_abs_correlation": best_abs,
                    "top_auxiliary_correlations": _top_correlations(component_scores, descriptors),
                    "top_bottom_descriptor_gaps": _top_bottom_gap(component_scores, descriptors, top_instances),
                    "lowest_instances": _instance_slice(component_scores, metadata, descriptors, low_idx),
                    "highest_instances": _instance_slice(component_scores, metadata, descriptors, high_idx[::-1]),
                }
            )

        output["methods"][method] = {
            "candidate_components": method_rows,
            "num_candidates": int(len(method_rows)),
        }
    return output


def _render_markdown(dossiers: List[Dict[str, Any]], top_components: int, threshold: float, top_instances: int) -> str:
    lines = [
        "# Unexplained Discovered Directions",
        "",
        "## Purpose",
        "",
        "This report inspects graph-level discovered directions that remain weakly explained by the current known bank.",
        "The known bank now includes both primitive constraints and graph-level concepts.",
        "",
        "## Setup",
        "",
        f"- `top_components_considered`: `{top_components}`",
        f"- `known_threshold`: `{threshold}`",
        f"- `top_instances`: `{top_instances}`",
        "",
    ]
    for dossier in dossiers:
        lines.extend(
            [
                f"## {dossier['model']}",
                "",
                f"- `config_id`: `{dossier['config_id']}`",
            ]
        )
        for method_name, method_payload in dossier["methods"].items():
            lines.extend(
                [
                    "",
                    f"### {method_name.upper()}",
                    "",
                    f"- `num_candidates`: `{method_payload['num_candidates']}`",
                ]
            )
            if not method_payload["candidate_components"]:
                lines.append("- No candidate component below the explanation threshold.")
                continue
            for component in method_payload["candidate_components"]:
                lines.extend(
                    [
                        "",
                        f"#### {method_name.upper()}{component['component_index']}",
                        "",
                        f"- Best known match: `{component['best_known_name']}` "
                        f"(`{component['best_known_type']}`, corr={_fmt(component['best_known_correlation'])}, abs={_fmt(component['best_known_abs_correlation'])})",
                        "- Top auxiliary correlations:",
                    ]
                )
                for item in component["top_auxiliary_correlations"]:
                    lines.append(
                        f"  - `{item['descriptor']}`: corr={_fmt(item['correlation'])}"
                    )
                lines.append("- Strongest top-bottom descriptor gaps:")
                for item in component["top_bottom_descriptor_gaps"]:
                    lines.append(
                        f"  - `{item['descriptor']}`: top-bottom={_fmt(item['top_minus_bottom'])}"
                    )
                lines.append("- Highest-score instances:")
                for item in component["highest_instances"]:
                    lines.append(
                        f"  - `idx={item['instance_index']}` score={_fmt(item['score'])} "
                        f"constraints={','.join(item['active_constraints']) or 'base_vrp'} "
                        f"desc={json.dumps(_json_ready(item['descriptors']), ensure_ascii=False)}"
                    )
                lines.append("- Lowest-score instances:")
                for item in component["lowest_instances"]:
                    lines.append(
                        f"  - `idx={item['instance_index']}` score={_fmt(item['score'])} "
                        f"constraints={','.join(item['active_constraints']) or 'base_vrp'} "
                        f"desc={json.dumps(_json_ready(item['descriptors']), ensure_ascii=False)}"
                    )
        lines.append("")
    return "\n".join(lines)


def _render_readme() -> str:
    lines = [
        "# Unexplained Direction Dossiers",
        "",
        "## Purpose",
        "",
        "This folder inspects discovered graph-level directions that are still weakly explained by the current known bank.",
        "The known bank includes both primitive constraints and graph-level concepts.",
        "",
        "A component appears here only if its best absolute correlation with the known bank stays below the chosen threshold.",
        "",
        "## Files",
        "",
        "- `component_dossiers.json`: machine-readable version",
        "- `component_dossiers.md`: human-readable analysis",
        "",
        "## How to read one component dossier",
        "",
        "- `Best known match`",
        "  - nearest item in the current known bank",
        "  - if the absolute correlation is still low, the component remains only weakly explained",
        "",
        "- `Top auxiliary correlations`",
        "  - exploratory correlations with additional descriptors not included in the official known bank",
        "  - use them as hypotheses, not as proof",
        "",
        "- `Strongest top-bottom descriptor gaps`",
        "  - compare the highest-score instances and lowest-score instances on this component",
        "  - large gaps often reveal what the axis is separating in practice",
        "",
        "- `Highest-score instances` / `Lowest-score instances`",
        "  - anchor examples for manual interpretation",
        "  - check recurring active constraints, geometry, demand concentration, and TW patterns",
        "",
        "## Recommended reading order",
        "",
        "1. Check whether the best known match is already moderately strong.",
        "2. Look for repeated auxiliary descriptors across the top correlations and top-bottom gaps.",
        "3. Compare the highest-score and lowest-score instances.",
        "4. Formulate a tentative interpretation.",
        "5. Validate it later with a targeted intervention if it looks promising.",
        "",
        "## Important caveat",
        "",
        "A weakly explained component is not automatically a new concept.",
        "It may still be:",
        "",
        "- a missing constraint descriptor",
        "- a missing structural concept",
        "- a mixture of several effects",
        "- an unstable decomposition artifact",
        "",
        "The right claim is:",
        "",
        "- this direction is not yet well explained by the current known bank",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows", []) or []
    reports_by_model = _report_by_model(payload)
    dossiers: List[Dict[str, Any]] = []

    for row in rows:
        report = reports_by_model[str(row["model"])]
        dossiers.append(
            _build_dossier_for_report(
                row=row,
                report=report,
                top_components=int(args.top_components),
                known_threshold=float(args.known_threshold),
                top_instances=int(args.top_instances),
            )
        )

    json_path = output_dir / "component_dossiers.json"
    md_path = output_dir / "component_dossiers.md"
    readme_path = output_dir / "README.md"
    json_path.write_text(json.dumps(_json_ready(dossiers), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        _render_markdown(
            dossiers,
            top_components=int(args.top_components),
            threshold=float(args.known_threshold),
            top_instances=int(args.top_instances),
        )
        + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(_render_readme() + "\n", encoding="utf-8")

    print(f"Wrote unexplained-direction dossiers to {output_dir}")
    print(f"- {json_path}")
    print(f"- {md_path}")
    print(f"- {readme_path}")


if __name__ == "__main__":
    main()
