from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

from encoder.concept_bank import compute_concept_bank
from encoder.discovered_directions import _pearson_corr, _summarize_method
from encoder.encoder_probe import _encoder_node_embeddings, _json_ready, _pool_node_embeddings
from engine.decode_ops import variant_metadata_from_inputs
from engine.model_io import ModelLoader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate discovered latent directions by applying controlled interventions "
            "to graph-level concepts and checking whether PCA/ICA component scores move as expected."
        )
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
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
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/probe.json",
    )
    return parser


def _fit_transform_pca(features: np.ndarray, seed: int, num_components: int) -> Tuple[StandardScaler, PCA, np.ndarray]:
    n_components = max(
        2,
        min(
            int(num_components),
            int(features.shape[0]),
            int(features.shape[1]),
        ),
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = PCA(n_components=n_components, random_state=seed)
    scores = model.fit_transform(scaled)
    return scaler, model, scores


def _fit_transform_ica(features: np.ndarray, seed: int, num_components: int) -> Tuple[StandardScaler, FastICA, np.ndarray]:
    n_components = max(
        2,
        min(
            int(num_components),
            int(features.shape[0]),
            int(features.shape[1]),
        ),
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = FastICA(
        n_components=n_components,
        random_state=seed,
        whiten="unit-variance",
        max_iter=2000,
        tol=1e-4,
    )
    scores = model.fit_transform(scaled)
    return scaler, model, scores


def _encode_features(
    model: Any,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    pooling: str,
) -> np.ndarray:
    with torch.inference_mode():
        encoder_output = model.encoder((node_features, global_features))
        node_embeddings = _encoder_node_embeddings(encoder_output)
    return _pool_node_embeddings(node_embeddings, pooling)


def _metadata(node_features: torch.Tensor, global_features: torch.Tensor) -> List[Dict[str, Any]]:
    return list(variant_metadata_from_inputs(node_features, global_features))


def _apply_geometry_spread(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_node = node_features.clone()
    depot = new_node[:, 0:1, :2]
    customers = new_node[:, 1:, :2]
    scale = 1.20
    new_node[:, 1:, :2] = depot + (customers - depot) * scale
    valid_mask = torch.ones(node_features.shape[0], dtype=torch.bool, device=node_features.device)
    return new_node, global_features.clone(), valid_mask


def _redistribute_concentration(values: torch.Tensor, gamma: float = 1.7) -> torch.Tensor:
    total = values.sum(dim=1, keepdim=True)
    positive = torch.clamp(values, min=0.0)
    weights = torch.pow(positive + 1e-6, gamma)
    weight_sum = weights.sum(dim=1, keepdim=True)
    safe_weights = torch.where(weight_sum > 0, weights / torch.clamp(weight_sum, min=1e-6), torch.zeros_like(weights))
    return safe_weights * total


def _apply_demand_concentration(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_node = node_features.clone()
    new_node[:, 1:, 2] = _redistribute_concentration(new_node[:, 1:, 2])
    new_node[:, 1:, 3] = _redistribute_concentration(new_node[:, 1:, 3])
    valid_mask = torch.ones(node_features.shape[0], dtype=torch.bool, device=node_features.device)
    return new_node, global_features.clone(), valid_mask


def _apply_tighten_time_windows(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_node = node_features.clone()
    start = new_node[:, 1:, 4]
    end = new_node[:, 1:, 5]
    service = torch.clamp(new_node[:, 1:, 6], min=1e-3)
    active = torch.isfinite(end)
    width = torch.clamp(end - start, min=service)
    midpoint = 0.5 * (start + end)
    new_width = torch.maximum(width * 0.60, service * 1.10)
    new_start = midpoint - 0.5 * new_width
    new_end = midpoint + 0.5 * new_width
    new_node[:, 1:, 4] = torch.where(active, new_start, start)
    new_node[:, 1:, 5] = torch.where(active, new_end, end)
    valid_mask = active.any(dim=1)
    return new_node, global_features.clone(), valid_mask


def _apply_tighten_distance_limit(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_global = global_features.clone()
    limit = new_global[:, 3]
    active = torch.isfinite(limit)
    tightened = torch.clamp(limit * 0.75, min=1e-3)
    new_global[:, 3] = torch.where(active, tightened, limit)
    return node_features.clone(), new_global, active


INTERVENTIONS: List[Dict[str, Any]] = [
    {
        "name": "geometry_spread",
        "target_concept": "instance_compactness_state",
        "intended_direction": "increase",
        "apply": _apply_geometry_spread,
    },
    {
        "name": "demand_concentration",
        "target_concept": "load_concentration_state",
        "intended_direction": "increase",
        "apply": _apply_demand_concentration,
    },
    {
        "name": "tighten_time_windows",
        "target_concept": "tw_width_profile_state",
        "intended_direction": "decrease",
        "apply": _apply_tighten_time_windows,
    },
    {
        "name": "tighten_distance_limit",
        "target_concept": "distance_budget_pressure_state",
        "intended_direction": "decrease",
        "apply": _apply_tighten_distance_limit,
    },
]


def _evaluate_method_under_intervention(
    payload: Dict[str, Any],
    scaler: StandardScaler,
    model: Any,
    original_features: np.ndarray,
    counterfactual_features: np.ndarray,
    target_concept: str,
    concept_delta: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    concept_info = (payload.get("concept_alignment") or {}).get(target_concept, {}) or {}
    component_index = concept_info.get("best_component_index", None)
    concept_corr = concept_info.get("best_correlation", None)
    if component_index in (None, "") or concept_corr in (None, ""):
        return {
            "selected_component_index": None,
            "selected_component_correlation": None,
            "valid_instances": int(valid_mask.sum()),
            "directional_success_rate": None,
            "aligned_delta_correlation": None,
            "mean_component_delta": None,
            "mean_abs_component_delta": None,
        }

    component_index = int(component_index) - 1
    concept_corr = float(concept_corr)
    if not np.isfinite(concept_corr):
        return {
            "selected_component_index": component_index + 1,
            "selected_component_correlation": None,
            "valid_instances": int(valid_mask.sum()),
            "directional_success_rate": None,
            "aligned_delta_correlation": None,
            "mean_component_delta": None,
            "mean_abs_component_delta": None,
        }

    original_scores = model.transform(scaler.transform(original_features))
    counterfactual_scores = model.transform(scaler.transform(counterfactual_features))
    component_delta = counterfactual_scores[:, component_index] - original_scores[:, component_index]

    eps = 1e-8
    valid = (
        valid_mask
        & np.isfinite(concept_delta)
        & np.isfinite(component_delta)
        & (np.abs(concept_delta) > eps)
    )
    if not np.any(valid):
        return {
            "selected_component_index": component_index + 1,
            "selected_component_correlation": concept_corr,
            "valid_instances": 0,
            "directional_success_rate": None,
            "aligned_delta_correlation": None,
            "mean_component_delta": None,
            "mean_abs_component_delta": None,
        }

    expected_signed_delta = concept_delta[valid] * np.sign(concept_corr)
    observed_delta = component_delta[valid]
    success = observed_delta * expected_signed_delta > eps
    aligned_delta_correlation = _pearson_corr(expected_signed_delta, observed_delta)

    return {
        "selected_component_index": component_index + 1,
        "selected_component_correlation": concept_corr,
        "valid_instances": int(valid.sum()),
        "directional_success_rate": float(np.mean(success.astype(np.float32))),
        "aligned_delta_correlation": aligned_delta_correlation,
        "mean_component_delta": float(np.mean(observed_delta)),
        "mean_abs_component_delta": float(np.mean(np.abs(observed_delta))),
    }


def compute_intervention_validation_bundle(args: argparse.Namespace) -> Dict[str, Any]:
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
    metadata = _metadata(node_features, global_features)

    original_features = _encode_features(model, node_features, global_features, args.pooling)
    original_concepts = compute_concept_bank(node_features, global_features, metadata)

    pca_scaler, pca_model, pca_scores = _fit_transform_pca(
        original_features,
        seed=int(args.seed),
        num_components=int(args.num_components),
    )
    pca_payload, _ = _summarize_method(
        method_name="pca",
        scores=pca_scores,
        concept_names=sorted(original_concepts["concept_raw_values"].keys()),
        concept_raw_values={
            name: np.asarray(values, dtype=np.float32)
            for name, values in original_concepts["concept_raw_values"].items()
        },
        top_components=min(int(args.top_components), pca_scores.shape[1]),
        component_vectors=np.asarray(pca_model.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca_model.explained_variance_ratio_, dtype=np.float64),
    )
    ica_scaler, ica_model, ica_scores = _fit_transform_ica(
        original_features,
        seed=int(args.seed),
        num_components=int(args.num_components),
    )
    ica_payload, _ = _summarize_method(
        method_name="ica",
        scores=ica_scores,
        concept_names=sorted(original_concepts["concept_raw_values"].keys()),
        concept_raw_values={
            name: np.asarray(values, dtype=np.float32)
            for name, values in original_concepts["concept_raw_values"].items()
        },
        top_components=min(int(args.top_components), ica_scores.shape[1]),
        component_vectors=np.asarray(ica_model.components_, dtype=np.float32),
        explained_variance_ratio=None,
    )

    interventions_report: Dict[str, Any] = {}
    for spec in INTERVENTIONS:
        cf_node, cf_global, valid_mask_tensor = spec["apply"](node_features, global_features)
        cf_metadata = _metadata(cf_node, cf_global)
        cf_features = _encode_features(model, cf_node, cf_global, args.pooling)
        cf_concepts = compute_concept_bank(cf_node, cf_global, cf_metadata)

        target = str(spec["target_concept"])
        original_value = np.asarray(original_concepts["concept_raw_values"][target], dtype=np.float32)
        counterfactual_value = np.asarray(cf_concepts["concept_raw_values"][target], dtype=np.float32)
        concept_delta = counterfactual_value - original_value
        intended_sign = 1.0 if spec["intended_direction"] == "increase" else -1.0
        valid_mask = valid_mask_tensor.detach().cpu().numpy().astype(bool)
        eps = 1e-8
        concept_valid = valid_mask & np.isfinite(concept_delta) & (np.abs(concept_delta) > eps)
        concept_success = concept_valid & (concept_delta * intended_sign > eps)

        interventions_report[str(spec["name"])] = {
            "target_concept": target,
            "intended_direction": str(spec["intended_direction"]),
            "valid_instances": int(concept_valid.sum()),
            "concept_success_rate": (
                float(np.mean(concept_success.astype(np.float32)[concept_valid]))
                if np.any(concept_valid)
                else None
            ),
            "mean_target_concept_delta": (
                float(np.mean(concept_delta[concept_valid])) if np.any(concept_valid) else None
            ),
            "methods": {
                "pca": _evaluate_method_under_intervention(
                    pca_payload,
                    pca_scaler,
                    pca_model,
                    original_features,
                    cf_features,
                    target,
                    concept_delta,
                    concept_valid,
                ),
                "ica": _evaluate_method_under_intervention(
                    ica_payload,
                    ica_scaler,
                    ica_model,
                    original_features,
                    cf_features,
                    target,
                    concept_delta,
                    concept_valid,
                ),
            },
        }

    return {
        "config": {
            "problem": str(config.problem),
            "graph_size": int(config.graph_size),
            "encoder_name": str(config.encoder.__name__),
            "decoder_name": str(config.decoder.__name__),
            "config_repr": repr(config),
            "checkpoint_path": str(checkpoint_path),
            "pooling": args.pooling,
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "num_components": int(args.num_components),
            "top_components": int(args.top_components),
        },
        "matrix_richness": {
            "num_latent_features": int(original_features.shape[1]),
        },
        "base_discovered_directions": {
            "pca": pca_payload,
            "ica": ica_payload,
        },
        "interventions": interventions_report,
    }


def write_intervention_validation_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = _build_parser().parse_args()
    report = compute_intervention_validation_bundle(args)
    output_path = write_intervention_validation_report(report, args.output)
    print(
        f"Intervention validation saved to {output_path}\n"
        f"encoder={report['config']['encoder_name']} pooling={report['config']['pooling']} "
        f"instances={report['config']['num_samples']}"
    )


if __name__ == "__main__":
    main()
