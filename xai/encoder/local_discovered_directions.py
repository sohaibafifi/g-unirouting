from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from encoder.discovered_directions import (
    _fit_ica,
    _fit_pca,
    _safe_float,
    _summarize_method,
)
from encoder.edge_probe import compute_edge_probe_bundle
from encoder.encoder_probe import _json_ready
from encoder.node_probe import compute_node_probe_bundle


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def _level_specs(level: str) -> Dict[str, Any]:
    if level == "node":
        return {
            "compute_bundle": compute_node_probe_bundle,
            "feature_key": "customer_features",
            "raw_key": "node_concept_raw_values",
            "states_key": "node_concept_states",
            "bank_key": "node_concept_bank",
            "default_output": "logs/xai/encoder/node/discovered_directions/probe.json",
            "kind_label": "node",
            "sample_count_key": "num_probe_node_examples",
            "max_examples_arg": "max_probe_nodes",
        }
    if level == "edge":
        return {
            "compute_bundle": compute_edge_probe_bundle,
            "feature_key": "edge_features",
            "raw_key": "edge_concept_raw_values",
            "states_key": "edge_concept_states",
            "bank_key": "edge_concept_bank",
            "default_output": "logs/xai/encoder/edge/discovered_directions/probe.json",
            "kind_label": "edge",
            "sample_count_key": "num_probe_edges",
            "max_examples_arg": "max_probe_edges",
        }
    raise ValueError(f"Unsupported level: {level}")


def _build_parser(level: str) -> argparse.ArgumentParser:
    spec = _level_specs(level)
    parser = argparse.ArgumentParser(
        description=(
            f"Study discovered latent directions in encoder {level}-level representations "
            f"by extracting PCA/ICA axes and correlating them with the existing {level} concept bank."
        )
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=256 if level == "node" else 128)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--inference-batch-size", type=int, default=64 if level == "node" else 32)
    parser.add_argument(f"--{spec['max_examples_arg'].replace('_', '-')}", type=int, default=20000)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument("--output", default=spec["default_output"])
    parser.add_argument("--artifacts-output", default=None)
    return parser


def _expand_concept_values(
    concept_states: Dict[str, np.ndarray],
    concept_raw_values: Dict[str, np.ndarray],
    base_display_names: Dict[str, str],
    class_orders: Dict[str, List[str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    expanded_values: Dict[str, np.ndarray] = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in sorted(concept_raw_values.items())
    }
    expanded_display_names: Dict[str, str] = {
        str(name): str(base_display_names.get(name, name))
        for name in expanded_values.keys()
    }

    for concept_name, labels in sorted(concept_states.items()):
        label_array = np.asarray(labels).astype(str)
        ordered_labels = list(class_orders.get(concept_name, [])) or sorted(set(label_array.tolist()))
        base_display = str(base_display_names.get(concept_name, concept_name))
        for label in ordered_labels:
            concept_key = f"{concept_name}::{label}"
            expanded_values[concept_key] = (label_array == str(label)).astype(np.float32)
            expanded_display_names[concept_key] = f"{base_display} = {_pretty_label(str(label))}"

    return expanded_values, expanded_display_names


def compute_local_discovered_direction_bundle(
    args: argparse.Namespace,
    level: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    spec = _level_specs(level)
    probe_args = SimpleNamespace(**vars(args))
    probe_report, probe_artifacts = spec["compute_bundle"](probe_args)

    features = np.asarray(probe_artifacts[spec["feature_key"]], dtype=np.float32)
    bank = dict(probe_report[spec["bank_key"]])
    concept_raw_values, concept_display_names = _expand_concept_values(
        concept_states={
            str(name): np.asarray(values)
            for name, values in (probe_artifacts.get(spec["states_key"]) or {}).items()
        },
        concept_raw_values={
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in (probe_artifacts.get(spec["raw_key"]) or {}).items()
        },
        base_display_names=dict(bank.get("display_names", {})),
        class_orders=dict(bank.get("class_orders", {})),
    )
    concept_names = sorted(concept_raw_values.keys())
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
        display_names=concept_display_names,
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
        display_names=concept_display_names,
    )

    report = {
        "level": str(level),
        "config": dict(probe_report["config"]),
        "dataset": dict(probe_report["dataset"]),
        "matrix_richness": dict(probe_report["matrix_richness"]),
        "concept_bank_reference": {
            "display_names": dict(concept_display_names),
            "base_display_names": dict(bank.get("display_names", {})),
            "raw_value_summary": dict(bank.get("raw_value_summary", {})),
            "core_concepts": list(bank.get("core_concepts", [])),
        },
        "discovered_directions": pca_payload,
        "alternative_methods": {
            "ica": ica_payload,
        },
    }

    artifacts = {
        "feature_matrix": features.astype(np.float32, copy=False),
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
    }
    return report, artifacts


def write_local_discovered_direction_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_local_discovered_direction_artifacts(artifacts: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: Dict[str, Any] = {
        "feature_matrix": artifacts["feature_matrix"],
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
    }
    for name, values in sorted((artifacts.get("concept_raw_values") or {}).items()):
        flattened[f"concept_value__{name}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(path, **flattened)
    return path


def _main(level: str) -> None:
    args = _build_parser(level).parse_args()
    report, artifacts = compute_local_discovered_direction_bundle(args, level=level)

    if args.artifacts_output:
        artifact_path = write_local_discovered_direction_artifacts(artifacts, args.artifacts_output)
        report["artifacts"] = {"discovered_directions_path": str(artifact_path)}

    output_path = write_local_discovered_direction_report(report, args.output)
    summary = report["discovered_directions"]["summary"]
    count_key = _level_specs(level)["sample_count_key"]
    print(
        f"{level.capitalize()} discovered directions saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} decode={report['config']['decode_mode']} "
        f"{level}s={report['dataset'][count_key]} "
        f"mean_abs_corr={_safe_float(summary.get('mean_best_abs_correlation_top_components'))}"
    )


def main_for_level(level: str) -> None:
    _main(level)


if __name__ == "__main__":
    raise SystemExit("Use node_discovered_directions.py or edge_discovered_directions.py")
