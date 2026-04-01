from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from encoder.concept_bank import compute_concept_bank
from encoder.encoder_probe import (
    _centroid_margin,
    _cross_validated_probe,
    _encoder_node_embeddings,
    _json_ready,
    _kmeans_metrics,
    _label_aligned_kmeans,
    _label_silhouette,
    _matrix_richness,
    _pool_node_embeddings,
)
from engine.decode_ops import variant_metadata_from_inputs
from engine.model_io import ModelLoader


def _collapse_rare_labels(labels: list[str], min_count: int = 2) -> list[str]:
    counts = Counter(labels)
    if not counts:
        return labels
    return [label if counts[label] >= min_count else "other_rare" for label in labels]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantitative probe of encoder latent spaces focused on instance-level concepts."
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
    parser.add_argument("--output", default="logs/xai/encoder/graph/concepts/probe.json")
    parser.add_argument("--artifacts-output", default=None)
    return parser


def compute_concept_probe_bundle(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    concept_bank = compute_concept_bank(node_features, global_features, metadata)
    concept_states = concept_bank["concept_states"]
    primitive_flags: Dict[str, list[str]] = {}
    primitive_raw_values: Dict[str, np.ndarray] = {}
    for flag_name in sorted(metadata[0]["flags"].keys() if metadata else []):
        labels = ["on" if bool(meta["flags"][flag_name]) else "off" for meta in metadata]
        primitive_flags[flag_name] = labels
        primitive_raw_values[flag_name] = np.asarray(
            [1.0 if label == "on" else 0.0 for label in labels],
            dtype=np.float32,
        )
    core_signatures = concept_bank["core_concept_signatures"]
    probe_core_signatures = _collapse_rare_labels(core_signatures, min_count=2)
    unique_signatures = sorted(set(core_signatures))
    unique_probe_signatures = sorted(set(probe_core_signatures))

    clustering_k_values = list(
        range(
            2,
            min(max(args.max_k, 2), feature_matrix.shape[0] - 1) + 1,
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

    concept_probes: Dict[str, Dict[str, Any]] = {}
    concept_f1_values = []
    for concept_name, labels in sorted(concept_states.items()):
        is_binary = len(set(labels)) == 2
        linear_probe = _cross_validated_probe(
            feature_matrix,
            labels,
            seed=args.seed,
            binary=is_binary,
        )
        macro_f1 = linear_probe.get("macro_f1")
        if macro_f1 is not None:
            concept_f1_values.append(float(macro_f1))
        concept_probes[concept_name] = {
            "silhouette": _label_silhouette(feature_matrix, labels, seed=args.seed),
            "centroid_margin": _centroid_margin(feature_matrix, labels),
            "linear_probe": linear_probe,
            "class_balance": dict(Counter(labels)),
            "display_name": concept_bank["concept_display_names"].get(concept_name, concept_name),
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
            "num_core_concept_signatures": int(len(unique_signatures)),
            "core_concept_signature_counts": dict(Counter(core_signatures)),
            "num_probe_core_concept_signatures": int(len(unique_probe_signatures)),
            "probe_core_concept_signature_counts": dict(Counter(probe_core_signatures)),
        },
        "matrix_richness": _matrix_richness(node_matrix),
        "concept_bank": {
            "core_concepts": list(concept_bank["core_concept_names"]),
            "display_names": dict(concept_bank["concept_display_names"]),
            "class_orders": dict(concept_bank["concept_class_orders"]),
            "raw_value_summary": dict(concept_bank["raw_value_summary"]),
            "concept_macro_f1_mean": (
                float(np.mean(concept_f1_values)) if concept_f1_values else None
            ),
            "concept_macro_f1_std": (
                float(np.std(concept_f1_values)) if concept_f1_values else None
            ),
        },
        "concept_signature_separation": {
            "label_silhouette": _label_silhouette(
                feature_matrix,
                probe_core_signatures,
                seed=args.seed,
            ),
            "centroid_margin": _centroid_margin(feature_matrix, probe_core_signatures),
            "linear_probe": _cross_validated_probe(
                feature_matrix,
                probe_core_signatures,
                seed=args.seed,
                binary=False,
            ),
            "kmeans_aligned": _label_aligned_kmeans(
                feature_matrix,
                probe_core_signatures,
                seed=args.seed,
            ),
        },
        "concept_state_separation": concept_probes,
        "clustering_sweep": clustering_sweep,
        "best_k_by_silhouette": best_silhouette,
    }
    artifacts = {
        "pooled_features": feature_matrix.astype(np.float32, copy=False),
        "core_concept_signatures": np.asarray(core_signatures),
        "probe_core_concept_signatures": np.asarray(probe_core_signatures),
        "concept_states": {
            name: np.asarray(values)
            for name, values in sorted(concept_states.items())
        },
        "concept_raw_values": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in sorted(concept_bank["concept_raw_values"].items())
        },
        "primitive_flags": {
            name: np.asarray(values)
            for name, values in sorted(primitive_flags.items())
        },
        "primitive_raw_values": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in sorted(primitive_raw_values.items())
        },
    }
    return report, artifacts


def compute_concept_probe_report(args: argparse.Namespace) -> Dict[str, Any]:
    report, _ = compute_concept_probe_bundle(args)
    return report


def write_concept_probe_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_concept_probe_artifacts(artifacts: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: Dict[str, Any] = {
        "pooled_features": artifacts["pooled_features"],
        "core_concept_signatures": artifacts["core_concept_signatures"],
        "probe_core_concept_signatures": artifacts["probe_core_concept_signatures"],
    }
    for name, values in sorted((artifacts.get("concept_states") or {}).items()):
        flattened[f"concept_state__{name}"] = values
    for name, values in sorted((artifacts.get("concept_raw_values") or {}).items()):
        flattened[f"concept_value__{name}"] = values
    for name, values in sorted((artifacts.get("primitive_flags") or {}).items()):
        flattened[f"primitive__{name}"] = values
    for name, values in sorted((artifacts.get("primitive_raw_values") or {}).items()):
        flattened[f"primitive_value__{name}"] = values
    np.savez_compressed(path, **flattened)
    return path


def main() -> None:
    args = _build_parser().parse_args()
    report, artifacts = compute_concept_probe_bundle(args)

    if args.artifacts_output:
        artifact_path = write_concept_probe_artifacts(artifacts, args.artifacts_output)
        report["artifacts"] = {
            "pooled_features_path": str(artifact_path),
        }

    output_path = write_concept_probe_report(report, args.output)

    print(
        f"Concept probe saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} pooling={report['config']['pooling']} "
        f"instances={report['dataset']['num_instances']} "
        f"core_concept_signatures={report['dataset']['num_core_concept_signatures']}"
    )
    best_k = report["best_k_by_silhouette"]
    if best_k is not None:
        print(
            "Best k by silhouette: "
            f"k={int(best_k['k'])}, silhouette={best_k['silhouette']:.4f}"
        )


if __name__ == "__main__":
    main()
