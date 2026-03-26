from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from encoder.encoder_probe import (
    _centroid_margin,
    _cross_validated_probe,
    _encoder_node_embeddings,
    _json_ready,
    _kmeans_metrics,
    _label_aligned_kmeans,
    _label_silhouette,
    _matrix_richness,
)
from encoder.node_concept_bank import compute_node_concept_bank
from engine.decode_ops import variant_metadata_from_inputs
from engine.model_io import ModelLoader


def _collapse_rare_labels(labels: List[str], min_count: int = 4) -> List[str]:
    counts = Counter(labels)
    if not counts:
        return labels
    return [label if counts[label] >= min_count else "other_rare" for label in labels]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantitative node-level probe of encoder latent spaces using solution-derived concepts."
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-probe-nodes", type=int, default=20000)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="logs/xai/encoder/node_probes/probe.json")
    parser.add_argument("--artifacts-output", default=None)
    return parser


def _forward_with_cached_encoder(
    model: Any,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    decode_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = list(model.encoder((node_features, global_features)))
    node_embeddings = _encoder_node_embeddings(encoded)
    global_embeddings = encoded[1]
    if len(encoded) == 4:
        _, routes, _ = model.decoder(
            (node_features, global_features),
            encoded[0],
            global_embeddings,
            decode_mode=decode_mode,
            edge_index=encoded[2],
            edge_attn_scores=encoded[3],
        )
    else:
        _, routes, _ = model.decoder(
            (node_features, global_features),
            node_embeddings,
            global_embeddings,
            decode_mode=decode_mode,
        )
    return node_embeddings, routes


def _pad_route_batches(route_batches: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(batch.shape[1]) for batch in route_batches)
    padded: List[torch.Tensor] = []
    for batch in route_batches:
        if int(batch.shape[1]) == max_len:
            padded.append(batch)
            continue
        pad_width = max_len - int(batch.shape[1])
        padding = torch.zeros(
            (batch.shape[0], pad_width),
            dtype=batch.dtype,
            device=batch.device,
        )
        padded.append(torch.cat([batch, padding], dim=1))
    return torch.cat(padded, dim=0)


def _batched_encoder_and_routes(
    model: Any,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    decode_mode: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_batches: List[torch.Tensor] = []
    route_batches: List[torch.Tensor] = []
    total = int(node_features.shape[0])
    for start in range(0, total, max(batch_size, 1)):
        end = min(total, start + max(batch_size, 1))
        batch_nodes = node_features[start:end]
        batch_globals = global_features[start:end]
        with torch.inference_mode():
            node_embeddings, routes = _forward_with_cached_encoder(
                model=model,
                node_features=batch_nodes,
                global_features=batch_globals,
                decode_mode=decode_mode,
            )
        node_batches.append(node_embeddings.detach().cpu())
        route_batches.append(routes.detach().cpu())
    return torch.cat(node_batches, dim=0), _pad_route_batches(route_batches)


def _subsample_indices(total: int, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or total <= limit:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=limit, replace=False).astype(np.int64))


def compute_node_probe_bundle(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    node_embeddings, routes = _batched_encoder_and_routes(
        model=model,
        node_features=node_features,
        global_features=global_features,
        decode_mode=args.decode_mode,
        batch_size=args.inference_batch_size,
    )
    node_matrix = node_embeddings.detach().cpu().numpy()
    customer_feature_matrix = (
        node_embeddings[:, 1:, :].reshape(-1, node_embeddings.shape[-1]).detach().cpu().numpy()
    )

    node_concept_bank = compute_node_concept_bank(
        node_features=node_features.detach().cpu(),
        routes=routes,
        metadata=metadata,
    )
    concept_states = node_concept_bank["concept_states"]
    concept_raw_values = node_concept_bank["concept_raw_values"]
    core_signatures = node_concept_bank["core_concept_signatures"]
    probe_core_signatures = _collapse_rare_labels(core_signatures, min_count=4)

    probe_indices = _subsample_indices(
        total=customer_feature_matrix.shape[0],
        limit=int(args.max_probe_nodes),
        seed=args.seed,
    )
    probe_features = customer_feature_matrix[probe_indices]
    probe_core_signatures = [probe_core_signatures[idx] for idx in probe_indices.tolist()]
    sampled_concept_states = {
        name: [values[idx] for idx in probe_indices.tolist()]
        for name, values in sorted(concept_states.items())
    }
    sampled_concept_raw_values = {
        name: np.asarray(values, dtype=np.float32)[probe_indices]
        for name, values in sorted(concept_raw_values.items())
    }

    clustering_k_values = list(
        range(
            2,
            min(max(args.max_k, 2), probe_features.shape[0] - 1) + 1,
        )
    )
    clustering_sweep = [
        _kmeans_metrics(probe_features, k=k_value, seed=args.seed)
        for k_value in clustering_k_values
    ]
    best_silhouette = max(
        (row for row in clustering_sweep if np.isfinite(row.get("silhouette", np.nan))),
        key=lambda row: row["silhouette"],
        default=None,
    )

    concept_probes: Dict[str, Dict[str, Any]] = {}
    concept_f1_values = []
    for concept_name, labels in sorted(sampled_concept_states.items()):
        is_binary = len(set(labels)) == 2
        linear_probe = _cross_validated_probe(
            probe_features,
            labels,
            seed=args.seed,
            binary=is_binary,
        )
        macro_f1 = linear_probe.get("macro_f1")
        if macro_f1 is not None:
            concept_f1_values.append(float(macro_f1))
        concept_probes[concept_name] = {
            "silhouette": _label_silhouette(probe_features, labels, seed=args.seed),
            "centroid_margin": _centroid_margin(probe_features, labels),
            "linear_probe": linear_probe,
            "class_balance": dict(Counter(labels)),
            "display_name": node_concept_bank["concept_display_names"].get(concept_name, concept_name),
        }

    report = {
        "config": {
            "problem": str(config.problem),
            "graph_size": int(config.graph_size),
            "encoder_name": str(config.encoder.__name__),
            "decoder_name": str(config.decoder.__name__),
            "config_repr": repr(config),
            "checkpoint_path": str(checkpoint_path),
            "num_samples": int(args.num_samples),
            "decode_mode": str(args.decode_mode),
            "inference_batch_size": int(args.inference_batch_size),
            "max_probe_nodes": int(args.max_probe_nodes),
            "sampling_mode": "mtvrp_mixture",
        },
        "dataset": {
            "num_instances": int(node_embeddings.shape[0]),
            "num_customers_per_instance": int(node_embeddings.shape[1] - 1),
            "num_node_examples": int(customer_feature_matrix.shape[0]),
            "num_probe_node_examples": int(probe_features.shape[0]),
            "num_core_node_signatures": int(len(set(core_signatures))),
            "core_node_signature_counts": dict(Counter(core_signatures)),
            "num_probe_core_node_signatures": int(len(set(probe_core_signatures))),
            "probe_core_node_signature_counts": dict(Counter(probe_core_signatures)),
        },
        "matrix_richness": _matrix_richness(node_matrix),
        "node_concept_bank": {
            "core_concepts": list(node_concept_bank["core_concept_names"]),
            "display_names": dict(node_concept_bank["concept_display_names"]),
            "class_orders": dict(node_concept_bank["concept_class_orders"]),
            "raw_value_summary": dict(node_concept_bank["raw_value_summary"]),
            "concept_macro_f1_mean": (
                float(np.mean(concept_f1_values)) if concept_f1_values else None
            ),
            "concept_macro_f1_std": (
                float(np.std(concept_f1_values)) if concept_f1_values else None
            ),
        },
        "node_concept_signature_separation": {
            "label_silhouette": _label_silhouette(
                probe_features,
                probe_core_signatures,
                seed=args.seed,
            ),
            "centroid_margin": _centroid_margin(probe_features, probe_core_signatures),
            "linear_probe": _cross_validated_probe(
                probe_features,
                probe_core_signatures,
                seed=args.seed,
                binary=False,
            ),
            "kmeans_aligned": _label_aligned_kmeans(
                probe_features,
                probe_core_signatures,
                seed=args.seed,
            ),
        },
        "node_concept_state_separation": concept_probes,
        "clustering_sweep": clustering_sweep,
        "best_k_by_silhouette": best_silhouette,
    }
    artifacts = {
        "customer_features": probe_features.astype(np.float32, copy=False),
        "probe_indices": probe_indices.astype(np.int64, copy=False),
        "core_node_signatures": np.asarray(core_signatures),
        "probe_core_node_signatures": np.asarray(probe_core_signatures),
        "node_concept_states": {
            name: np.asarray(values)
            for name, values in sorted(sampled_concept_states.items())
        },
        "node_concept_raw_values": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in sorted(sampled_concept_raw_values.items())
        },
        "instance_indices": node_concept_bank["instance_indices"][probe_indices],
        "customer_indices": node_concept_bank["customer_indices"][probe_indices],
    }
    return report, artifacts


def compute_node_probe_report(args: argparse.Namespace) -> Dict[str, Any]:
    report, _ = compute_node_probe_bundle(args)
    return report


def write_node_probe_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_node_probe_artifacts(artifacts: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: Dict[str, Any] = {
        "customer_features": artifacts["customer_features"],
        "probe_indices": artifacts["probe_indices"],
        "core_node_signatures": artifacts["core_node_signatures"],
        "probe_core_node_signatures": artifacts["probe_core_node_signatures"],
        "instance_indices": artifacts["instance_indices"],
        "customer_indices": artifacts["customer_indices"],
    }
    for name, values in sorted((artifacts.get("node_concept_states") or {}).items()):
        flattened[f"node_concept_state__{name}"] = values
    for name, values in sorted((artifacts.get("node_concept_raw_values") or {}).items()):
        flattened[f"node_concept_value__{name}"] = values
    np.savez_compressed(path, **flattened)
    return path


def main() -> None:
    args = _build_parser().parse_args()
    report, artifacts = compute_node_probe_bundle(args)

    if args.artifacts_output:
        artifact_path = write_node_probe_artifacts(artifacts, args.artifacts_output)
        report["artifacts"] = {
            "customer_features_path": str(artifact_path),
        }

    output_path = write_node_probe_report(report, args.output)

    print(
        f"Node probe saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} decode={report['config']['decode_mode']} "
        f"instances={report['dataset']['num_instances']} "
        f"nodes={report['dataset']['num_probe_node_examples']}"
    )
    best_k = report["best_k_by_silhouette"]
    if best_k is not None:
        print(
            "Best k by silhouette: "
            f"k={int(best_k['k'])}, silhouette={best_k['silhouette']:.4f}"
        )


if __name__ == "__main__":
    main()
