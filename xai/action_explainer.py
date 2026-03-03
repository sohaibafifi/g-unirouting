from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import torch

if TYPE_CHECKING:
    from mavrp.configs.config import Config
    from mavrp.env.models import TransformerModel


FEATURE_SPECS: List[Dict[str, Any]] = [
    {"name": "locs", "source": "node", "index": slice(0, 2)},
    {"name": "demand_linehaul", "source": "node", "index": 2},
    {"name": "demand_backhaul", "source": "node", "index": 3},
    {"name": "time_windows", "source": "node", "index": slice(4, 6)},
    {"name": "service_time", "source": "node", "index": 6},
    {"name": "vehicle_capacity", "source": "global", "index": 0},
    {"name": "open_route", "source": "global", "index": 1},
    {"name": "mixed_backhaul", "source": "global", "index": 2},
    {"name": "distance_limit", "source": "global", "index": 3},
    {"name": "depot_tw_end", "source": "global", "index": 4},
    {"name": "has_backhaul", "source": "global", "index": 5},
]

FEATURE_SPEC_BY_NAME = {spec["name"]: spec for spec in FEATURE_SPECS}

CONSTRAINT_GROUP_RULES: List[Tuple[str, set[str]]] = [
    ("space_distance", {"locs", "distance_limit"}),
    ("time_windows_service", {"time_windows", "service_time", "depot_tw_end"}),
    ("capacity_demands", {"demand_linehaul", "demand_backhaul", "vehicle_capacity"}),
    ("route_structure", {"open_route", "mixed_backhaul", "has_backhaul"}),
]

COUNTERFACTUAL_SPECS: List[Dict[str, Any]] = [
    {
        "id": "time_window_end",
        "feature": "time_windows",
        "source": "node",
        "raw_index": 5,
        "component": 1,
        "direction": "increase",
    },
    {
        "id": "service_time",
        "feature": "service_time",
        "source": "node",
        "raw_index": 6,
        "direction": "decrease",
    },
    {
        "id": "demand_linehaul",
        "feature": "demand_linehaul",
        "source": "node",
        "raw_index": 2,
        "direction": "decrease",
    },
    {
        "id": "demand_backhaul",
        "feature": "demand_backhaul",
        "source": "node",
        "raw_index": 3,
        "direction": "decrease",
    },
    {
        "id": "vehicle_capacity",
        "feature": "vehicle_capacity",
        "source": "global",
        "raw_index": 0,
        "direction": "increase",
    },
    {
        "id": "distance_limit",
        "feature": "distance_limit",
        "source": "global",
        "raw_index": 3,
        "direction": "increase",
    },
    {
        "id": "depot_tw_end",
        "feature": "depot_tw_end",
        "source": "global",
        "raw_index": 4,
        "direction": "increase",
    },
]


@dataclass
class DecodeCache:
    node_embeddings: torch.Tensor
    global_embeddings: torch.Tensor
    q_global: torch.Tensor | int
    attn_matrix: Optional[torch.Tensor]


@dataclass
class DecodeState:
    current_node: torch.Tensor
    not_served: torch.Tensor
    leave_time: torch.Tensor
    deliveries: torch.Tensor
    pickups: torch.Tensor
    distance: torch.Tensor
    total_distance: torch.Tensor
    is_depot: torch.Tensor


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or not bool(mask.any().item()):
        return 0.0
    return float(values[mask].mean().item())


def _safe_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-_.")
    return text.lower() or "unknown"


def _feature_to_constraint_group(feature_name: str) -> str:
    for group_name, members in CONSTRAINT_GROUP_RULES:
        if feature_name in members:
            return group_name
    return "other"


def _top_feature_payload(
    feature_scores: Dict[str, float], top_n: int = 3
) -> List[Dict[str, float]]:
    if not feature_scores:
        return []
    sanitized = {}
    for key, value in feature_scores.items():
        sanitized[key] = max(float(value), 0.0) if math.isfinite(float(value)) else 0.0
    total = sum(sanitized.values())
    items = sorted(sanitized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    payload = []
    for name, score in items:
        share = (score / total) if total > 0 else 0.0
        payload.append({"feature": name, "score": float(score), "share": float(share)})
    return payload


def _aggregate_constraint_scores(feature_scores: Dict[str, float]) -> Dict[str, float]:
    grouped: Dict[str, float] = defaultdict(float)
    for feature_name, score in feature_scores.items():
        if not math.isfinite(float(score)):
            continue
        grouped[_feature_to_constraint_group(feature_name)] += max(float(score), 0.0)
    return dict(grouped)


def _top_constraint_payload(
    constraint_scores: Dict[str, float], top_n: Optional[int] = 3
) -> List[Dict[str, float]]:
    if not constraint_scores:
        return []
    total = sum(max(float(v), 0.0) for v in constraint_scores.values())
    items = sorted(
        constraint_scores.items(), key=lambda x: max(float(x[1]), 0.0), reverse=True
    )
    if top_n is not None:
        items = items[:top_n]
    payload: List[Dict[str, float]] = []
    for name, score in items:
        score_pos = max(float(score), 0.0)
        share = (score_pos / total) if total > 0 else 0.0
        payload.append(
            {
                "constraint": name,
                "score": float(score_pos),
                "share": float(share),
            }
        )
    return payload


def _top_nodes_and_scores(
    node_scores: torch.Tensor, candidate_mask: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if node_scores.shape != candidate_mask.shape:
        raise ValueError("node_scores and candidate_mask must have the same shape")
    k_eff = min(k, node_scores.size(-1))
    masked_scores = torch.where(
        candidate_mask,
        node_scores,
        torch.full_like(node_scores, float("-inf")),
    )
    top_vals, top_idx = torch.topk(masked_scores, k_eff, dim=-1)
    valid_mask = torch.isfinite(top_vals)
    top_vals = torch.where(valid_mask, top_vals, torch.full_like(top_vals, float("nan")))
    top_idx = torch.where(valid_mask, top_idx, torch.zeros_like(top_idx))
    return top_idx, top_vals, valid_mask, k_eff


def _normalize_node_scores(node_scores: torch.Tensor) -> torch.Tensor:
    scores = node_scores.clone()
    if scores.size(-1) <= 1:
        return scores.zero_()
    customer_scores = scores[:, 1:]
    max_scores = customer_scores.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    scores[:, 1:] = customer_scores / max_scores
    scores[:, 0] = 0.0
    return scores


def _select_best_alternative(
    logits: torch.Tensor, logprobs: torch.Tensor, chosen_action: torch.Tensor, action_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = action_mask.bool().clone()
    masked.scatter_(1, chosen_action.unsqueeze(-1), False)

    alt_logprobs = logprobs.masked_fill(~masked, float("-inf"))
    alt_logprob, alt_action = alt_logprobs.max(dim=-1)
    has_alt = torch.isfinite(alt_logprob)

    alt_action_safe = torch.where(has_alt, alt_action, torch.zeros_like(alt_action))
    alt_logit = logits.gather(-1, alt_action_safe.unsqueeze(-1)).squeeze(-1)
    alt_logit = torch.where(has_alt, alt_logit, torch.zeros_like(alt_logit))
    alt_logprob = torch.where(has_alt, alt_logprob, torch.zeros_like(alt_logprob))
    return alt_action_safe, has_alt, alt_logit, alt_logprob


def _parse_topk_nodes(raw: str) -> List[int]:
    if isinstance(raw, str):
        tokens = str(raw).strip().strip("[]")
        values = [int(tok.strip()) for tok in tokens.split(",") if tok.strip()]
        values = sorted({v for v in values if v > 0})
        return values or [1, 3, 5]
    return [1, 3, 5]


def _parse_attr_features(raw: str | None) -> List[str]:
    if raw is None or raw.lower() == "auto":
        selected = [spec["name"] for spec in FEATURE_SPECS]
    else:
        selected = [token.strip() for token in raw.split(",") if token.strip()]
        selected = [name for name in selected if name in FEATURE_SPEC_BY_NAME]
    if "locs" not in selected:
        selected = ["locs"] + selected
    return selected


def _parse_feasibility_weight(
    raw_weight: float | None, recourse_enabled: bool
) -> float:
    if raw_weight is None:
        return 1.0 if recourse_enabled else 0.0
    return max(float(raw_weight), 0.0)


def _parse_feasibility_top_m(raw_top_m: int) -> int:
    return max(int(raw_top_m), 0)


def _parse_feasibility_cost_weight(raw_cost_weight: float) -> float:
    return max(float(raw_cost_weight), 0.0)


def _is_recourse_decoder(model: "TransformerModel") -> bool:
    return model.decoder.__class__.__name__ == "RecourseDecoder"


def _variant_metadata_from_inputs(
    node_features: torch.Tensor, global_features: torch.Tensor
) -> List[Dict[str, Any]]:
    open_route = global_features[:, 1] > 0.5
    mixed_backhaul = global_features[:, 2] > 0.5
    has_backhaul = global_features[:, 5] > 0.5
    has_limit = torch.isfinite(global_features[:, 3])
    has_tw = torch.isfinite(node_features[:, 1:, 5]).any(dim=1)

    out: List[Dict[str, Any]] = []
    for i in range(node_features.size(0)):
        is_open = bool(open_route[i].item())
        is_mixed = bool(mixed_backhaul[i].item())
        is_backhaul = bool(has_backhaul[i].item())
        is_limit = bool(has_limit[i].item())
        is_tw = bool(has_tw[i].item())
        code = (
            f"{'o' if is_open else ''}vrp"
            f"{'m' if is_mixed and is_backhaul else ''}"
            f"{'b' if is_backhaul else ''}"
            f"{'l' if is_limit else ''}"
            f"{'tw' if is_tw else ''}"
        )
        active_constraints: List[str] = []
        if is_open:
            active_constraints.append("open_route")
        if is_backhaul:
            active_constraints.append("mixed_backhaul" if is_mixed else "backhaul")
        if is_limit:
            active_constraints.append("distance_limit")
        if is_tw:
            active_constraints.append("time_windows")
        if not active_constraints:
            active_constraints = ["base_vrp"]
        out.append(
            {
                "code": code,
                "flags": {
                    "open_route": is_open,
                    "backhaul": is_backhaul,
                    "mixed_backhaul": is_mixed,
                    "distance_limit": is_limit,
                    "time_windows": is_tw,
                },
                "active_constraints": active_constraints,
            }
        )
    return out


def _init_instance_traces(
    node_features: torch.Tensor, num_store: int, variant_meta: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    locs = node_features[:num_store, :, :2].detach().cpu().tolist()
    return [
        {
            "instance_index": int(i),
            "instance_variant_code": variant_meta[i]["code"],
            "instance_variant_flags": variant_meta[i]["flags"],
            "instance_active_constraints": variant_meta[i]["active_constraints"],
            "locs": locs[i],
            "actions": [],
            "done_before": [],
            "done_after": [],
            "top_nodes": [],
            "top_scores": [],
            "top_nodes_decision": [],
            "top_scores_decision": [],
            "top_nodes_feasibility": [],
            "top_scores_feasibility": [],
            "top_features": [],
            "top_constraints": [],
            "contrastive_alt_action": [],
            "contrastive_alt_source": [],
            "contrastive_alt_feasible": [],
            "contrastive_alt_recourse": [],
            "contrastive_logit_gap": [],
            "contrastive_logprob_gap": [],
            "contrastive_top_constraints": [],
            "chosen_feasible": [],
            "recourse_triggered": [],
            "recourse_cost_est": [],
            "counterfactuals": [],
        }
        for i in range(num_store)
    ]


def _loc_distance(locs: Sequence[Sequence[float]], src: int, dst: int) -> float:
    if not (0 <= src < len(locs) and 0 <= dst < len(locs)):
        return float("nan")
    src_xy = locs[src]
    dst_xy = locs[dst]
    if len(src_xy) < 2 or len(dst_xy) < 2:
        return float("nan")
    return float(math.hypot(float(src_xy[0]) - float(dst_xy[0]), float(src_xy[1]) - float(dst_xy[1])))


def _first_true_index(flags: Sequence[bool]) -> Optional[int]:
    for idx, flag in enumerate(flags):
        if bool(flag):
            return int(idx)
    return None


def _count_true_bursts(flags: Sequence[bool]) -> int:
    bursts = 0
    prev = False
    for flag in flags:
        curr = bool(flag)
        if curr and not prev:
            bursts += 1
        prev = curr
    return int(bursts)


def _mean_constraint_share_per_step(
    payloads: Sequence[Any],
) -> Dict[str, float]:
    if not payloads:
        return {}
    known_groups = [name for name, _ in CONSTRAINT_GROUP_RULES] + ["other"]
    per_group: Dict[str, List[float]] = {name: [] for name in known_groups}
    for payload in payloads:
        step_map = {name: 0.0 for name in known_groups}
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("constraint", "")).strip() or "other"
                try:
                    share = max(float(item.get("share", 0.0)), 0.0)
                except (TypeError, ValueError):
                    continue
                target = name if name in step_map else "other"
                step_map[target] += share
        for name in known_groups:
            per_group[name].append(step_map[name])
    return {
        name: _safe_mean(values) for name, values in per_group.items() if values
    }


def _dominant_constraint_name(shares: Dict[str, float]) -> str:
    if not shares:
        return "none"
    best_name, best_value = max(
        shares.items(), key=lambda kv: (float(kv[1]), str(kv[0]))
    )
    return str(best_name) if float(best_value) > 0 else "none"


def _summarize_trajectory(instance_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not instance_traces:
        return {}

    stored_steps: List[float] = []
    customer_actions_per_trace: List[float] = []
    depot_returns_per_trace: List[float] = []
    depot_return_share_per_trace: List[float] = []
    first_depot_step: List[float] = []
    first_depot_step_norm: List[float] = []
    action_distance_mean: List[float] = []
    customer_hop_distance_mean: List[float] = []
    recourse_events_per_trace: List[float] = []
    recourse_burst_count: List[float] = []
    first_recourse_step: List[float] = []
    first_recourse_step_norm: List[float] = []
    early_constraint_terms: Dict[str, List[float]] = defaultdict(list)
    late_constraint_terms: Dict[str, List[float]] = defaultdict(list)

    for trace in instance_traces:
        actions = [int(v) for v in (trace.get("actions", []) or [])]
        if not actions:
            continue
        locs = trace.get("locs", []) or []
        top_constraints = trace.get("top_constraints", []) or []
        recourse_flags = [bool(v) for v in (trace.get("recourse_triggered", []) or [])]
        step_count = len(actions)
        stored_steps.append(float(step_count))

        customer_count = sum(1 for action in actions if action > 0)
        depot_count = sum(1 for action in actions if action == 0)
        customer_actions_per_trace.append(float(customer_count))
        depot_returns_per_trace.append(float(depot_count))
        depot_return_share_per_trace.append(
            float(depot_count / step_count) if step_count > 0 else 0.0
        )

        first_depot = next((idx for idx, action in enumerate(actions) if action == 0), None)
        if first_depot is not None:
            first_depot_step.append(float(first_depot))
            first_depot_step_norm.append(float(first_depot / max(step_count, 1)))

        per_action_distances: List[float] = []
        per_customer_hops: List[float] = []
        current = 0
        for action in actions:
            hop = _loc_distance(locs, current, action)
            if math.isfinite(hop):
                per_action_distances.append(hop)
                if action > 0 and current > 0:
                    per_customer_hops.append(hop)
            current = action
        action_distance_mean.append(_safe_mean(per_action_distances))
        customer_hop_distance_mean.append(_safe_mean(per_customer_hops))

        recourse_events_per_trace.append(float(sum(1 for flag in recourse_flags if flag)))
        recourse_burst_count.append(float(_count_true_bursts(recourse_flags)))
        first_recourse = _first_true_index(recourse_flags)
        if first_recourse is not None:
            first_recourse_step.append(float(first_recourse))
            first_recourse_step_norm.append(float(first_recourse / max(step_count, 1)))

        split_idx = max(1, step_count // 2)
        early_share = _mean_constraint_share_per_step(top_constraints[:split_idx])
        late_share = _mean_constraint_share_per_step(top_constraints[split_idx:])
        if not late_share and early_share:
            late_share = dict(early_share)
        for name, value in early_share.items():
            early_constraint_terms[name].append(float(value))
        for name, value in late_share.items():
            late_constraint_terms[name].append(float(value))

    early_constraint_share = {
        name: _safe_mean(values)
        for name, values in sorted(early_constraint_terms.items())
    }
    late_constraint_share = {
        name: _safe_mean(values)
        for name, values in sorted(late_constraint_terms.items())
    }
    early_top = _dominant_constraint_name(early_constraint_share)
    late_top = _dominant_constraint_name(late_constraint_share)

    return {
        "basis": "stored_instance_traces",
        "num_traces": int(len([trace for trace in instance_traces if trace.get("actions")])),
        "stored_step_mean": _safe_mean(stored_steps),
        "mean_customer_actions_per_instance": _safe_mean(customer_actions_per_trace),
        "mean_depot_returns_per_instance": _safe_mean(depot_returns_per_trace),
        "mean_depot_return_share": _safe_mean(depot_return_share_per_trace),
        "mean_first_depot_return_step": _safe_mean(first_depot_step),
        "mean_first_depot_return_step_norm": _safe_mean(first_depot_step_norm),
        "mean_action_distance": _safe_mean(action_distance_mean),
        "mean_customer_hop_distance": _safe_mean(customer_hop_distance_mean),
        "mean_recourse_events_per_instance": _safe_mean(recourse_events_per_trace),
        "mean_recourse_burst_count": _safe_mean(recourse_burst_count),
        "mean_first_recourse_step": _safe_mean(first_recourse_step),
        "mean_first_recourse_step_norm": _safe_mean(first_recourse_step_norm),
        "early_constraint_share": early_constraint_share,
        "late_constraint_share": late_constraint_share,
        "early_top_constraint": early_top,
        "late_top_constraint": late_top,
        "dominant_constraint_shift": f"{early_top}->{late_top}",
    }


def _resolve_config(args: argparse.Namespace) -> Tuple[Config, Optional[int], Path]:
    from mavrp.configs.config import Config

    configs = Config.all()
    selected_idx: Optional[int] = None

    if args.config_id is not None:
        if args.config_id < 0 or args.config_id >= len(configs):
            raise IndexError(f"config-id out of range: {args.config_id}")
        config = configs[args.config_id]
        selected_idx = int(args.config_id)
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        folder_name = ckpt_path.parent.name
        graph_size = ckpt_path.parent.parent.name if len(ckpt_path.parents) >= 2 else None
        problem = ckpt_path.parent.parent.parent.name if len(ckpt_path.parents) >= 3 else None
        matches = [
            (i, conf)
            for i, conf in enumerate(configs)
            if repr(conf) == folder_name
            and (graph_size is None or str(conf.graph_size) == str(graph_size))
            and (problem is None or conf.problem == problem)
        ]
        if not matches:
            raise ValueError(
                "Unable to infer config from checkpoint path. Pass --config-id explicitly."
            )
        selected_idx, config = matches[0]
    else:
        raise ValueError("Pass either --config-id or --checkpoint")

    if args.graph_size is not None:
        config.graph_size = int(args.graph_size)
    if args.problem is not None:
        config.problem = str(args.problem)
    if args.device is not None:
        config.device = str(args.device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        ckpt_path = Path(config.working_dir) / config.problem / str(config.graph_size) / repr(config) / "baseline.pt"
        ckpt_path = ckpt_path.resolve()

    return config, selected_idx, ckpt_path


def _load_model(config: "Config", checkpoint_path: Path) -> "TransformerModel":
    from mavrp.env.models import TransformerModel

    model = TransformerModel(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".ckpt":
        model.load_from_ckpt(str(checkpoint_path), baseline=False)
    else:
        state = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(state, strict=True, assign=True)
    model = model.to(config.device)
    model.eval()
    return model


def _randomize_model_weights(model: "TransformerModel") -> None:
    def _reset(module: torch.nn.Module) -> None:
        if module is model:
            return
        reset_fn = getattr(module, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    model.apply(_reset)
    model.eval()


def _prepare_inputs(
    config: "Config", num_instances: int
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    dataset = config.get_problem().dataset(
        graph_size=config.graph_size,
        num_samples=num_instances,
        device=config.device,
    )
    node_features = dataset.node_features.to(config.device)
    global_features = dataset.global_features.to(config.device)
    variant_meta = _variant_metadata_from_inputs(node_features, global_features)
    return node_features, global_features, variant_meta


def _encode_inputs(
    model: "TransformerModel", node_features: torch.Tensor, global_features: torch.Tensor
) -> DecodeCache:
    decoder = model.decoder
    # Some mavrp encoders mutate input tensors in-place during preprocessing.
    # Pass non-leaf clones so gradients still flow back to the tracked leaves
    # used by the explainer, while avoiding "view of a leaf Variable" errors.
    node_features_model = node_features.clone()
    global_features_model = global_features.clone()
    encoded = list(model.encoder((node_features_model, global_features_model)))
    node_embeddings = encoded[0]
    global_embeddings = encoded[1]
    decoder.glimpse.precalculate(node_embeddings)
    h_hat = global_embeddings.unsqueeze(1)
    q_global: torch.Tensor | int
    if decoder.q_global is not None:
        q_global = decoder.q_global(h_hat)
    else:
        q_global = 0

    attn_matrix: Optional[torch.Tensor] = None
    if decoder.edge_attn_scale is not None and len(encoded) == 4:
        try:
            import torch_geometric.utils
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for edge-attention explainability"
            ) from exc
        edge_index = encoded[2]
        edge_attn_scores = encoded[3]
        edge_scores = (
            edge_attn_scores.mean(-1) if edge_attn_scores.dim() > 1 else edge_attn_scores
        )
        batch_size, seq_len = node_embeddings.shape[:2]
        batch_vec = torch.arange(batch_size, device=node_embeddings.device).repeat_interleave(seq_len)
        attn_matrix = torch_geometric.utils.to_dense_adj(
            edge_index=edge_index, edge_attr=edge_scores, batch=batch_vec
        )

    return DecodeCache(
        node_embeddings=node_embeddings,
        global_embeddings=global_embeddings,
        q_global=q_global,
        attn_matrix=attn_matrix,
    )


def _build_common(
    node_features: torch.Tensor, global_features: torch.Tensor
) -> Dict[str, torch.Tensor]:
    locations = node_features[:, :, :2]
    demands = node_features[:, :, 2]
    demands_b = node_features[:, :, 3]
    time_windows = node_features[:, :, 4:6]
    services = node_features[:, :, 6]
    capacities = global_features[:, 0].unsqueeze(1)
    open_routes = global_features[:, 1] > 0.5
    mixed_backhauls = global_features[:, 2] > 0.5
    distance_limits = global_features[:, 3].unsqueeze(1)
    time_limits = global_features[:, 4].unsqueeze(1)

    deltas = torch.cdist(locations, locations)
    if bool(open_routes.any().item()):
        deltas = deltas.clone()
        zeros = torch.zeros_like(deltas[:, :, 0])
        deltas[:, :, 0] = torch.where(open_routes.unsqueeze(-1), zeros, deltas[:, :, 0])

    backhauls_instances = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
    backhauls_instances = backhauls_instances.unsqueeze(-1).expand_as(demands_b)
    backhauls_mask = (demands_b > 0) & backhauls_instances
    linehauls_mask = (demands > 0) & backhauls_instances
    invalid_arcs = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)

    earliest_start_time, latest_start_time = time_windows.unbind(-1)
    inf_time = torch.full_like(latest_start_time, float("inf"))
    latest_start_time = torch.where(open_routes.unsqueeze(-1), inf_time, latest_start_time)

    return {
        "locations": locations,
        "demands": demands,
        "demands_b": demands_b,
        "time_windows": time_windows,
        "services": services,
        "capacities": capacities,
        "open_routes": open_routes,
        "mixed_backhauls": mixed_backhauls,
        "distance_limits": distance_limits,
        "time_limits": time_limits,
        "deltas": deltas,
        "invalid_arcs": invalid_arcs,
        "earliest_start_time": earliest_start_time,
        "latest_start_time": latest_start_time,
    }


def _init_state(common: Dict[str, torch.Tensor]) -> DecodeState:
    batch_size, seq_len = common["demands"].shape
    device = common["demands"].device
    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    not_served = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    not_served[:, 0] = False
    leave_time = common["earliest_start_time"][:, 0].unsqueeze(1).clone()
    zeros = torch.zeros((batch_size, 1), dtype=common["demands"].dtype, device=device)
    return DecodeState(
        current_node=current_node,
        not_served=not_served,
        leave_time=leave_time,
        deliveries=zeros.clone(),
        pickups=zeros.clone(),
        distance=zeros.clone(),
        total_distance=zeros.clone(),
        is_depot=torch.ones(batch_size, dtype=torch.bool, device=device),
    )


def _compute_action_masks(
    common: Dict[str, torch.Tensor],
    state: DecodeState,
    recourse_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = state.not_served.shape
    device = state.current_node.device
    batch_indices = torch.arange(batch_size, device=device)

    deltas = common["deltas"]
    demands = common["demands"]
    demands_b = common["demands_b"]
    services = common["services"]
    capacities = common["capacities"]
    distance_limits = common["distance_limits"]
    time_limits = common["time_limits"]
    earliest_start_time = common["earliest_start_time"]
    latest_start_time = common["latest_start_time"]
    mixed_backhauls = common["mixed_backhauls"].unsqueeze(-1)
    current = state.current_node

    potential_arrival_times = state.leave_time + deltas[batch_indices, current, :]
    potential_start_times = torch.maximum(potential_arrival_times, earliest_start_time)
    exceed_latest_start_time = potential_start_times > latest_start_time
    exceed_time_limit = (
        potential_start_times
        + services[batch_indices, :]
        + deltas[batch_indices, :, 0]
        > time_limits
    )
    exceed_infinite_time_limit = potential_start_times.isinf()

    potential_distance = state.distance + deltas[batch_indices, current, :]
    potential_distance_to_depot = potential_distance + deltas[batch_indices, :, 0]
    exceed_distance_limit = potential_distance_to_depot > distance_limits
    exceed_infinite_distance_limit = potential_distance_to_depot.isinf()

    exceed_capacity = (
        (state.deliveries + demands > capacities)
        | (state.pickups + demands_b > capacities)
    )
    cannot_serve_linehaul = mixed_backhauls & (demands + state.pickups > capacities)

    base_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    base_mask = base_mask.masked_fill(state.not_served, False)
    base_mask = base_mask.masked_fill(
        common["invalid_arcs"][batch_indices, current, :], True
    )
    base_mask = base_mask.scatter(1, current.unsqueeze(1), True)
    base_mask[:, 0] = state.is_depot & state.not_served.any(dim=1)

    full_mask = base_mask.masked_fill(
        exceed_capacity
        | cannot_serve_linehaul
        | exceed_latest_start_time
        | exceed_infinite_time_limit
        | exceed_distance_limit
        | exceed_infinite_distance_limit
        | exceed_time_limit,
        True,
    )
    policy_mask = base_mask if recourse_enabled else full_mask
    return policy_mask, full_mask, potential_distance


def _step_logits_and_mask(
    model: "TransformerModel",
    cache: DecodeCache,
    common: Dict[str, torch.Tensor],
    state: DecodeState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder = model.decoder
    batch_size, seq_len = cache.node_embeddings.shape[:2]
    device = cache.node_embeddings.device
    batch_indices = torch.arange(batch_size, device=device)

    deltas = common["deltas"]
    capacities = common["capacities"]
    distance_limits = common["distance_limits"]
    current = state.current_node

    recourse_enabled = _is_recourse_decoder(model)
    policy_mask, full_mask, potential_distance = _compute_action_masks(
        common, state, recourse_enabled
    )

    last = cache.node_embeddings[batch_indices, current, :].unsqueeze(1)

    remaining_distance = torch.nan_to_num(distance_limits - state.distance, posinf=10.0)
    remaining_demand = capacities - state.deliveries
    remaining_demand_b = capacities - state.pickups
    distance_to_depot = deltas[batch_indices, current, 0]
    distance_to_depot = torch.nan_to_num(distance_to_depot, posinf=1e9).unsqueeze(-1)

    open_routes = common["open_routes"].unsqueeze(-1).to(cache.node_embeddings.dtype)
    mixed_backhaul_float = common["mixed_backhauls"].unsqueeze(-1).to(cache.node_embeddings.dtype)

    dynamic_data = torch.cat(
        (
            last.squeeze(1),
            state.leave_time,
            remaining_distance,
            remaining_demand,
            remaining_demand_b,
            distance_to_depot,
            open_routes,
            mixed_backhaul_float,
        ),
        dim=-1,
    )
    dynamic_context = decoder.dynamic_context_embedding(dynamic_data)
    h_c = cache.q_global + dynamic_context.unsqueeze(1)

    dist_feat = deltas[batch_indices, current, :].unsqueeze(-1)
    logits = decoder.glimpse(h_c, policy_mask, dist=dist_feat)
    if decoder.edge_distance_scale is not None:
        logits = logits - decoder.edge_distance_scale * deltas[batch_indices, current, :]
    if cache.attn_matrix is not None and decoder.edge_attn_scale is not None:
        attn_feat = cache.attn_matrix[batch_indices, current, :]
        logits = logits + decoder.edge_attn_scale * attn_feat

    logits = logits.masked_fill(policy_mask, float("-inf"))
    logprobs = torch.log_softmax(logits, dim=-1)
    return logits, policy_mask, full_mask, logprobs, potential_distance


def _step_update_state(
    common: Dict[str, torch.Tensor],
    state: DecodeState,
    selected_node: torch.Tensor,
    potential_distance: torch.Tensor,
    full_mask: torch.Tensor,
    recourse_enabled: bool,
) -> DecodeState:
    batch_indices = torch.arange(selected_node.size(0), device=selected_node.device)
    demands = common["demands"]
    demands_b = common["demands_b"]
    services = common["services"]
    earliest_start_time = common["earliest_start_time"]
    time_windows = common["time_windows"]
    deltas = common["deltas"]

    potential_arrival_times = state.leave_time + deltas[batch_indices, state.current_node, :]
    potential_start_times = torch.maximum(potential_arrival_times, earliest_start_time)

    not_served = state.not_served.clone()
    not_served[batch_indices, selected_node] = False

    selected_demands = demands[batch_indices, selected_node]
    selected_demands_b = demands_b[batch_indices, selected_node]
    deliveries = state.deliveries + selected_demands.unsqueeze(1)
    pickups = state.pickups + selected_demands_b.unsqueeze(1)

    distance = potential_distance.gather(1, selected_node.unsqueeze(-1))
    selected_service_time = services[batch_indices, selected_node].unsqueeze(1)
    selected_start_times = potential_start_times.gather(1, selected_node.unsqueeze(-1))
    leave_time = selected_start_times + selected_service_time

    selected_is_depot = selected_node == 0
    selected_is_feasible = ~full_mask.gather(1, selected_node.unsqueeze(-1)).squeeze(-1)
    recourse_triggered = recourse_enabled & (~selected_is_depot) & (~selected_is_feasible)

    total_distance = state.total_distance.clone()
    if bool(selected_is_depot.any().item()):
        depot_cost = deltas[
            batch_indices[selected_is_depot], selected_node[selected_is_depot], 0
        ].unsqueeze(-1)
        total_distance[selected_is_depot] = (
            total_distance[selected_is_depot]
            + distance[selected_is_depot]
            + depot_cost
        )
        deliveries[selected_is_depot] = 0
        pickups[selected_is_depot] = 0
        distance[selected_is_depot] = 0
        leave_time[selected_is_depot] = time_windows[selected_is_depot, 0, 0].unsqueeze(1)

    current_node = selected_node.clone()
    is_depot = selected_is_depot.clone()

    if bool(recourse_triggered.any().item()):
        recourse_cost = _estimate_recourse_trip_cost(common, selected_node).unsqueeze(-1)
        total_distance[recourse_triggered] = (
            total_distance[recourse_triggered] + recourse_cost[recourse_triggered]
        )
        deliveries[recourse_triggered] = state.deliveries[recourse_triggered]
        pickups[recourse_triggered] = state.pickups[recourse_triggered]
        distance[recourse_triggered] = state.distance[recourse_triggered]
        leave_time[recourse_triggered] = state.leave_time[recourse_triggered]
        current_node[recourse_triggered] = state.current_node[recourse_triggered]
        is_depot[recourse_triggered] = state.is_depot[recourse_triggered]

    not_served[:, 0] = ~is_depot

    return DecodeState(
        current_node=current_node.detach(),
        not_served=not_served.detach(),
        leave_time=leave_time.detach(),
        deliveries=deliveries.detach(),
        pickups=pickups.detach(),
        distance=distance.detach(),
        total_distance=total_distance.detach(),
        is_depot=is_depot.detach(),
    )


def _feature_slice(
    tensor: torch.Tensor, spec: Dict[str, Any]
) -> torch.Tensor:
    index = spec["index"]
    if tensor.dim() == 3:
        return tensor[:, :, index]
    return tensor[:, index]


def _extract_feature_grads(
    node_grad: Optional[torch.Tensor],
    global_grad: Optional[torch.Tensor],
    selected_features: Sequence[str],
) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for name in selected_features:
        spec = FEATURE_SPEC_BY_NAME[name]
        grad_source = node_grad if spec["source"] == "node" else global_grad
        if grad_source is None:
            out[name] = None
        else:
            out[name] = _feature_slice(grad_source, spec)
    return out


def _grad_to_instance_scores(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grad is None:
        return None
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.abs().reshape(grad.shape[0], -1).mean(dim=-1)


def _grad_to_node_scores(grad: Optional[torch.Tensor], num_nodes: int) -> Optional[torch.Tensor]:
    if grad is None or grad.dim() < 2:
        return None
    if grad.shape[1] != num_nodes:
        return None
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    scores = grad.abs()
    if scores.dim() > 2:
        scores = scores.sum(dim=tuple(range(2, scores.dim())))
    return scores


def _perturb_topk_nodes(
    node_features: torch.Tensor, node_scores: torch.Tensor, candidate_mask: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    perturbed = node_features.clone()
    customer_mask = candidate_mask.clone()
    customer_mask[:, 0] = False
    batch_size = node_scores.size(0)
    max_available = int(customer_mask.sum(dim=-1).max().item())
    if max_available <= 0:
        empty_idx = torch.zeros((batch_size, 0), dtype=torch.long, device=node_scores.device)
        return perturbed, empty_idx, 0

    k_eff = min(k, max_available)
    topk_idx = torch.zeros((batch_size, k_eff), dtype=torch.long, device=node_scores.device)

    for batch_idx in range(batch_size):
        valid_nodes = torch.nonzero(customer_mask[batch_idx], as_tuple=False).squeeze(-1)
        if valid_nodes.numel() == 0:
            continue
        local_scores = node_scores[batch_idx, valid_nodes]
        local_k = min(k_eff, int(valid_nodes.numel()))
        local_top = torch.topk(local_scores, local_k, dim=-1).indices
        selected_nodes = valid_nodes[local_top]
        topk_idx[batch_idx, :local_k] = selected_nodes
        perturbed[batch_idx, selected_nodes, :2] = perturbed[batch_idx, 0, :2]

    return perturbed, topk_idx, k_eff


def _estimate_recourse_trip_cost(
    common: Dict[str, torch.Tensor], action: torch.Tensor
) -> torch.Tensor:
    deltas = common["deltas"]
    batch_indices = torch.arange(action.shape[0], device=action.device)
    dist_depot_to_next = deltas[batch_indices, 0, action]
    dist_next_to_depot = deltas[batch_indices, action, 0]
    open_route = common["open_routes"]
    return dist_depot_to_next + dist_next_to_depot * (~open_route)


def _infer_recourse_for_action(
    common: Dict[str, torch.Tensor],
    action: torch.Tensor,
    full_mask: torch.Tensor,
    recourse_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = action.shape[0]
    device = action.device
    is_depot = action == 0
    action_feasible = ~full_mask.gather(1, action.unsqueeze(-1)).squeeze(-1)

    recourse_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    recourse_cost_est = torch.zeros(
        batch_size, dtype=common["deltas"].dtype, device=common["deltas"].device
    )

    if not recourse_enabled:
        return recourse_flags, recourse_cost_est, action_feasible

    recourse_flags = (~is_depot) & (~action_feasible)
    if bool(recourse_flags.any().item()):
        recourse_trip_cost = _estimate_recourse_trip_cost(common, action)
        recourse_cost_est = torch.where(
            recourse_flags, recourse_trip_cost, torch.zeros_like(recourse_trip_cost)
        )
    return recourse_flags, recourse_cost_est, action_feasible


def _compute_feasibility_node_scores(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    state: DecodeState,
    action: torch.Tensor,
    base_feasible: torch.Tensor,
    base_recourse_cost: torch.Tensor,
    decision_node_scores: torch.Tensor,
    top_m: int,
    cost_weight: float,
    recourse_enabled: bool,
) -> torch.Tensor:
    scores = torch.zeros_like(decision_node_scores)
    if not recourse_enabled:
        return scores

    batch_size, num_nodes = decision_node_scores.shape
    num_customers = num_nodes - 1
    if num_customers <= 0:
        return scores

    m_eff = num_customers if top_m <= 0 else min(top_m, num_customers)
    customer_scores = decision_node_scores[:, 1:]
    candidate_idx = torch.topk(customer_scores, k=m_eff, dim=-1).indices + 1
    is_depot_action = action == 0
    candidate_idx[:, -1] = torch.where(is_depot_action, candidate_idx[:, -1], action)

    common_base = _build_common(node_features, global_features)
    depot_locs = common_base["locations"][:, :1, :]
    distance_scale = (
        torch.norm(common_base["locations"][:, 1:, :] - depot_locs, dim=-1)
        .mean(dim=-1)
        .clamp_min(1e-6)
    )
    base_feasible_float = base_feasible.float()

    for rank in range(m_eff):
        node_idx = candidate_idx[:, rank]
        node_perturbed = node_features.clone()
        scatter_idx = node_idx.view(batch_size, 1, 1).expand(-1, 1, 2)
        node_perturbed[:, :, :2].scatter_(1, scatter_idx, depot_locs)
        common_pert = _build_common(node_perturbed, global_features)
        _, full_mask_pert, _ = _compute_action_masks(
            common_pert, state, recourse_enabled
        )
        _, recourse_cost_pert, feasible_pert = _infer_recourse_for_action(
            common_pert, action, full_mask_pert, recourse_enabled
        )
        delta_feasible = (feasible_pert.float() - base_feasible_float).abs()
        delta_cost = (recourse_cost_pert - base_recourse_cost).abs()
        delta_cost_norm = torch.clamp(delta_cost / distance_scale, min=0.0, max=1.0)
        sensitivity = delta_feasible + (cost_weight * delta_cost_norm)
        sensitivity = torch.where(
            is_depot_action, torch.zeros_like(sensitivity), sensitivity
        )

        prev = scores.gather(-1, node_idx.unsqueeze(-1)).squeeze(-1)
        updated = torch.maximum(prev, sensitivity)
        scores.scatter_(-1, node_idx.unsqueeze(-1), updated.unsqueeze(-1))

    scores[:, 0] = 0.0
    return scores


def _slice_state(state: DecodeState, index: int) -> DecodeState:
    return DecodeState(
        current_node=state.current_node[index : index + 1].clone(),
        not_served=state.not_served[index : index + 1].clone(),
        leave_time=state.leave_time[index : index + 1].clone(),
        deliveries=state.deliveries[index : index + 1].clone(),
        pickups=state.pickups[index : index + 1].clone(),
        distance=state.distance[index : index + 1].clone(),
        total_distance=state.total_distance[index : index + 1].clone(),
        is_depot=state.is_depot[index : index + 1].clone(),
    )


def _apply_counterfactual_change(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    spec: Dict[str, Any],
    target_node: int,
    delta: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    node_cf = node_features.clone()
    global_cf = global_features.clone()
    raw_index = int(spec["raw_index"])
    direction = str(spec["direction"])

    if spec["source"] == "node":
        current_value = node_cf[0, target_node, raw_index]
        if direction == "increase":
            updated = current_value + float(delta)
        else:
            updated = torch.clamp(current_value - float(delta), min=0.0)
        node_cf[0, target_node, raw_index] = updated
        new_value = float(node_cf[0, target_node, raw_index].item())
    else:
        current_value = global_cf[0, raw_index]
        if direction == "increase":
            updated = current_value + float(delta)
        else:
            updated = torch.clamp(current_value - float(delta), min=0.0)
        global_cf[0, raw_index] = updated
        new_value = float(global_cf[0, raw_index].item())

    return node_cf, global_cf, new_value


def _replay_single_step(
    model: "TransformerModel",
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    state: DecodeState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    common = _build_common(node_features, global_features)
    cache = _encode_inputs(model, node_features, global_features)
    logits, _, full_mask, logprobs, _ = _step_logits_and_mask(model, cache, common, state)
    action = logprobs.argmax(dim=-1)
    return action, full_mask, logits


def _propose_counterfactual(
    model: "TransformerModel",
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    state: DecodeState,
    batch_index: int,
    alt_action: int,
    has_alt: bool,
    contrastive_logit_gap: float,
    contrastive_grad_by_feature: Dict[str, Optional[torch.Tensor]],
    full_mask: torch.Tensor,
    recourse_enabled: bool,
) -> Optional[Dict[str, Any]]:
    if (not has_alt) or alt_action <= 0 or (not math.isfinite(float(contrastive_logit_gap))):
        return None
    if float(contrastive_logit_gap) <= 0.0:
        return None

    base_target_feasible = bool((~full_mask[batch_index, alt_action]).item())
    candidate_rows: List[Dict[str, Any]] = []

    for spec in COUNTERFACTUAL_SPECS:
        grad_tensor = contrastive_grad_by_feature.get(str(spec["feature"]))
        if grad_tensor is None:
            continue

        if spec["source"] == "node":
            if alt_action >= node_features.shape[1]:
                continue
            if grad_tensor.dim() == 3:
                comp = int(spec.get("component", 0))
                grad_value = float(grad_tensor[batch_index, alt_action, comp].item())
            elif grad_tensor.dim() == 2:
                grad_value = float(grad_tensor[batch_index, alt_action].item())
            else:
                continue
            current_value = float(
                node_features[batch_index, alt_action, int(spec["raw_index"])].item()
            )
        else:
            if grad_tensor.dim() == 2:
                grad_value = float(grad_tensor[batch_index, 0].item())
            else:
                grad_value = float(grad_tensor[batch_index].item())
            current_value = float(
                global_features[batch_index, int(spec["raw_index"])].item()
            )

        if not math.isfinite(grad_value) or not math.isfinite(current_value):
            continue

        direction = str(spec["direction"])
        if direction == "increase":
            if grad_value >= -1e-8:
                continue
            delta = float(contrastive_logit_gap) / max(-grad_value, 1e-8)
        else:
            if grad_value <= 1e-8 or current_value <= 0.0:
                continue
            delta = float(contrastive_logit_gap) / max(grad_value, 1e-8)
            if delta > current_value:
                continue

        if not math.isfinite(delta) or delta <= 0.0:
            continue

        reference = max(abs(current_value), 1.0)
        relative_delta = delta / reference
        if relative_delta > 5.0:
            continue

        candidate_rows.append(
            {
                "spec": spec,
                "delta": float(delta),
                "relative_delta": float(relative_delta),
                "current_value": float(current_value),
            }
        )

    if not candidate_rows:
        return None

    candidate_rows.sort(key=lambda row: (row["relative_delta"], row["delta"]))
    state_one = _slice_state(state, batch_index)
    node_one = node_features[batch_index : batch_index + 1].detach()
    global_one = global_features[batch_index : batch_index + 1].detach()

    best_attempt: Optional[Dict[str, Any]] = None
    with torch.no_grad():
        for row in candidate_rows[:3]:
            spec = row["spec"]
            delta_apply = float(row["delta"]) * 1.05
            node_cf, global_cf, new_value = _apply_counterfactual_change(
                node_one, global_one, spec, alt_action, delta_apply
            )
            action_cf, full_mask_cf, _ = _replay_single_step(
                model, node_cf, global_cf, state_one
            )
            target_selected_after = bool(int(action_cf.item()) == int(alt_action))
            target_feasible_after = bool((~full_mask_cf[0, alt_action]).item())
            target_recourse_after = bool(
                recourse_enabled and (alt_action != 0) and (not target_feasible_after)
            )

            payload = {
                "status": "approximate",
                "target_action": int(alt_action),
                "feature": str(spec["id"]),
                "direction": str(spec["direction"]),
                "scope": str(spec["source"]),
                "node": int(alt_action) if spec["source"] == "node" else None,
                "delta": float(delta_apply),
                "estimated_delta": float(row["delta"]),
                "relative_delta": float(row["relative_delta"]),
                "new_value": float(new_value),
                "target_feasible_before": bool(base_target_feasible),
                "target_feasible_after": bool(target_feasible_after),
                "target_selected_after": bool(target_selected_after),
                "target_recourse_after": bool(target_recourse_after),
            }
            if target_selected_after:
                payload["status"] = "switch"
                return payload
            if target_feasible_after and (not base_target_feasible):
                payload["status"] = "make_feasible"
                return payload
            if best_attempt is None:
                best_attempt = payload

    return best_attempt


def _collect_step_record(
    step: int,
    action: torch.Tensor,
    selected_logit: torch.Tensor,
    selected_logprob: torch.Tensor,
    node_scores: torch.Tensor,
    decision_node_scores: torch.Tensor,
    feasibility_node_scores: torch.Tensor,
    use_feasibility_importance: bool,
    topk_metrics: Dict[int, Dict[str, float]],
    top1_nodes: torch.Tensor,
    done_ratio: float,
    top_features: List[Dict[str, float]],
    top_constraints: List[Dict[str, float]],
    feature_attr_mean: Dict[str, float],
    constraint_attr_mean: Dict[str, float],
    contrastive_alt_available_rate: float,
    contrastive_logit_gap_mean: float,
    contrastive_logprob_gap_mean: float,
    contrastive_top_constraints: List[Dict[str, float]],
    recourse_flags: torch.Tensor,
    recourse_cost_est: torch.Tensor,
    action_feasible: torch.Tensor,
    done_before: torch.Tensor,
    done_after: torch.Tensor,
) -> Dict[str, Any]:
    recourse_rate = float(recourse_flags.float().mean().item())
    return {
        "step": int(step),
        "done_ratio": float(done_ratio),
        "actions": action.detach().cpu().tolist(),
        "selected_logit_mean": float(selected_logit.mean().item()),
        "selected_logprob_mean": float(selected_logprob.mean().item()),
        "mean_node_attr": float(node_scores.mean().item()),
        "mean_node_attr_decision": float(decision_node_scores.mean().item()),
        "mean_node_attr_feasibility": float(feasibility_node_scores.mean().item()),
        "use_feasibility_importance": bool(use_feasibility_importance),
        "top1_nodes": top1_nodes.detach().cpu().tolist(),
        "top_features": top_features,
        "top_constraints": top_constraints,
        "feature_attr_mean": feature_attr_mean,
        "constraint_attr_mean": constraint_attr_mean,
        "contrastive": {
            "alt_available_rate": float(contrastive_alt_available_rate),
            "mean_logit_gap": float(contrastive_logit_gap_mean),
            "mean_logprob_gap": float(contrastive_logprob_gap_mean),
            "top_constraints": contrastive_top_constraints,
        },
        "recourse_rate": recourse_rate,
        "recourse_flags": recourse_flags.detach().cpu().tolist(),
        "recourse_cost_est_mean": float(recourse_cost_est.mean().item()),
        "chosen_action_feasible_rate": float(action_feasible.float().mean().item()),
        "done_before": done_before.detach().cpu().tolist(),
        "done_after": done_after.detach().cpu().tolist(),
        "deletion": {str(k): m for k, m in topk_metrics.items()},
    }


def run(args: argparse.Namespace) -> Path:
    if args.seed is not None:
        L.seed_everything(int(args.seed), workers=True)

    config, config_id, checkpoint_path = _resolve_config(args)
    model = _load_model(config, checkpoint_path)
    if args.randomize_weights:
        _randomize_model_weights(model)
    recourse_enabled = _is_recourse_decoder(model)

    topk_list = _parse_topk_nodes(args.topk_nodes)
    selected_features = _parse_attr_features(args.attr_features)
    feasibility_weight = _parse_feasibility_weight(
        args.feasibility_weight, recourse_enabled
    )
    feasibility_top_m = _parse_feasibility_top_m(args.feasibility_top_m)
    feasibility_cost_weight = _parse_feasibility_cost_weight(
        args.feasibility_cost_weight
    )
    use_feasibility_importance = recourse_enabled and (feasibility_weight > 0.0)
    importance_mode = (
        "decision+feasibility" if use_feasibility_importance else "decision-only"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-seed immediately before dataset generation so instance sampling is
    # aligned across different model architectures. Model construction above
    # may consume RNG state and otherwise shift the sampled batch.
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    if data_seed is not None:
        L.seed_everything(int(data_seed), workers=True)
    node_features, global_features, variant_meta = _prepare_inputs(config, args.num_instances)
    batch_size, num_nodes = node_features.shape[:2]
    max_attr_k = max(topk_list)
    num_store = min(args.num_instances, args.max_instances_to_store) if args.save_instance_traces else 0
    instance_traces = (
        _init_instance_traces(node_features, num_store, variant_meta)
        if num_store > 0
        else []
    )

    per_k_logit_drop: Dict[int, List[float]] = defaultdict(list)
    per_k_logprob_drop: Dict[int, List[float]] = defaultdict(list)
    per_k_flip_rate: Dict[int, List[float]] = defaultdict(list)
    per_feature_attr: Dict[str, List[float]] = defaultdict(list)
    per_feature_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
    per_constraint_attr: Dict[str, List[float]] = defaultdict(list)
    per_constraint_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
    recourse_rate_history: List[float] = []
    recourse_cost_est_history: List[float] = []
    chosen_feasible_rate_history: List[float] = []
    counterfactual_available_history: List[float] = []
    counterfactual_switch_history: List[float] = []
    counterfactual_make_feasible_history: List[float] = []
    counterfactual_approximate_history: List[float] = []
    counterfactual_relative_delta_history: List[float] = []
    counterfactual_by_feature: Dict[str, List[float]] = defaultdict(list)
    contrastive_alt_available_history: List[float] = []
    contrastive_logit_gap_history: List[float] = []
    contrastive_logprob_gap_history: List[float] = []
    decision_node_attr_history: List[float] = []
    feasibility_node_attr_history: List[float] = []

    step_records: List[Dict[str, Any]] = []
    state = _init_state(_build_common(node_features, global_features))
    top_k_effective = 0
    executed_steps = 0

    for step in range(args.max_steps):
        if not bool(state.not_served.any().item()):
            break
        executed_steps += 1

        done_before = ~state.not_served.any(dim=1)

        node_inputs = node_features.detach().clone().requires_grad_(True)
        global_inputs = global_features.detach().clone().requires_grad_(True)

        common = _build_common(node_inputs, global_inputs)
        cache = _encode_inputs(model, node_inputs, global_inputs)
        logits, policy_mask, full_mask, logprobs, potential_distance = _step_logits_and_mask(
            model, cache, common, state
        )

        action = logprobs.argmax(dim=-1)
        selected_logit = logits.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        selected_logprob = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        (
            alt_action_policy,
            has_alt_policy,
            alt_logit_policy,
            alt_logprob_policy,
        ) = _select_best_alternative(
            logits=logits,
            logprobs=logprobs,
            chosen_action=action,
            action_mask=~policy_mask,
        )
        (
            alt_action_full,
            has_alt_full,
            alt_logit_full,
            alt_logprob_full,
        ) = _select_best_alternative(
            logits=logits,
            logprobs=logprobs,
            chosen_action=action,
            action_mask=~full_mask,
        )
        has_alt = has_alt_full | has_alt_policy
        alt_action = torch.where(has_alt_full, alt_action_full, alt_action_policy)
        alt_logit = torch.where(has_alt_full, alt_logit_full, alt_logit_policy)
        alt_logprob = torch.where(has_alt_full, alt_logprob_full, alt_logprob_policy)
        contrastive_source_code = torch.zeros_like(action)
        contrastive_source_code = torch.where(
            has_alt_policy, torch.ones_like(contrastive_source_code), contrastive_source_code
        )
        contrastive_source_code = torch.where(
            has_alt_full,
            torch.full_like(contrastive_source_code, 2),
            contrastive_source_code,
        )
        contrastive_logit_gap = selected_logit - alt_logit
        contrastive_logprob_gap = selected_logprob - alt_logprob
        contrastive_alt_available_history.append(float(has_alt.float().mean().item()))
        contrastive_logit_gap_history.append(_masked_mean(contrastive_logit_gap, has_alt))
        contrastive_logprob_gap_history.append(
            _masked_mean(contrastive_logprob_gap, has_alt)
        )

        (
            recourse_flags,
            recourse_cost_est,
            action_feasible,
        ) = _infer_recourse_for_action(
            common,
            action.detach(),
            full_mask.detach(),
            recourse_enabled,
        )
        recourse_rate_history.append(float(recourse_flags.float().mean().item()))
        recourse_cost_est_history.append(float(recourse_cost_est.mean().item()))
        chosen_feasible_rate_history.append(float(action_feasible.float().mean().item()))
        alt_action_feasible = (
            ~full_mask.gather(1, alt_action.unsqueeze(-1)).squeeze(-1)
        ) & has_alt
        alt_recourse_flags = (
            has_alt
            & recourse_enabled
            & (alt_action != 0)
            & (~alt_action_feasible)
        )

        grads = torch.autograd.grad(
            outputs=selected_logit.sum(),
            inputs=[node_inputs, global_inputs],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        contrastive_target = (selected_logit - alt_logit) * has_alt.float()
        contrastive_grads = torch.autograd.grad(
            outputs=contrastive_target.sum(),
            inputs=[node_inputs, global_inputs],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_by_feature = _extract_feature_grads(grads[0], grads[1], selected_features)
        contrastive_grad_by_feature = _extract_feature_grads(
            contrastive_grads[0], contrastive_grads[1], selected_features
        )

        step_feature_attr_mean: Dict[str, float] = {}
        step_contrastive_feature_attr_mean: Dict[str, float] = {}
        instance_feature_attr: Dict[str, torch.Tensor] = {}
        instance_contrastive_feature_attr: Dict[str, torch.Tensor] = {}
        decision_node_scores = torch.zeros(
            (batch_size, num_nodes), dtype=node_inputs.dtype, device=node_inputs.device
        )

        for name in selected_features:
            grad = grad_by_feature.get(name)
            inst_score = _grad_to_instance_scores(grad)
            if inst_score is None:
                inst_score = torch.zeros(batch_size, dtype=node_inputs.dtype, device=node_inputs.device)
                mean_score = 0.0
            else:
                mean_score = float(inst_score.mean().item())
            step_feature_attr_mean[name] = mean_score
            instance_feature_attr[name] = inst_score
            per_feature_attr[name].append(mean_score)

            node_score = _grad_to_node_scores(grad, num_nodes)
            if node_score is not None:
                decision_node_scores = decision_node_scores + node_score

            contrastive_grad = contrastive_grad_by_feature.get(name)
            contrastive_inst_score = _grad_to_instance_scores(contrastive_grad)
            if contrastive_inst_score is None:
                contrastive_inst_score = torch.zeros(
                    batch_size, dtype=node_inputs.dtype, device=node_inputs.device
                )
                contrastive_mean_score = 0.0
            else:
                contrastive_mean_score = float(contrastive_inst_score.mean().item())
            step_contrastive_feature_attr_mean[name] = contrastive_mean_score
            instance_contrastive_feature_attr[name] = contrastive_inst_score
            per_feature_contrastive_attr[name].append(contrastive_mean_score)

        decision_node_scores[:, 0] = 0.0
        decision_node_attr_history.append(float(decision_node_scores[:, 1:].mean().item()))
        feasibility_node_scores = torch.zeros_like(decision_node_scores)
        if use_feasibility_importance:
            feasibility_node_scores = _compute_feasibility_node_scores(
                node_features=node_inputs.detach(),
                global_features=global_inputs.detach(),
                state=state,
                action=action.detach(),
                base_feasible=action_feasible.detach(),
                base_recourse_cost=recourse_cost_est.detach(),
                decision_node_scores=decision_node_scores.detach(),
                top_m=feasibility_top_m,
                cost_weight=feasibility_cost_weight,
                recourse_enabled=recourse_enabled,
            )
            node_scores = (
                _normalize_node_scores(decision_node_scores.detach())
                + feasibility_weight * feasibility_node_scores
            )
        else:
            node_scores = decision_node_scores
        feasibility_node_scores[:, 0] = 0.0
        feasibility_node_attr_history.append(
            float(feasibility_node_scores[:, 1:].mean().item())
        )

        rank_candidate_mask = state.not_served.clone()
        rank_candidate_mask[:, 0] = True
        customer_candidate_mask = state.not_served.clone()
        customer_candidate_mask[:, 0] = False

        masked_node_scores = torch.where(
            rank_candidate_mask,
            node_scores,
            torch.full_like(node_scores, float("-inf")),
        )
        top1_nodes = masked_node_scores.argmax(dim=-1)
        top_nodes_all, top_scores_all, top_nodes_valid_all, top_k_effective = _top_nodes_and_scores(
            node_scores, rank_candidate_mask, max_attr_k
        )
        (
            top_nodes_decision_all,
            top_scores_decision_all,
            top_nodes_decision_valid_all,
            _,
        ) = _top_nodes_and_scores(
            decision_node_scores, rank_candidate_mask, max_attr_k
        )
        if use_feasibility_importance:
            (
                top_nodes_feasibility_all,
                top_scores_feasibility_all,
                top_nodes_feasibility_valid_all,
                _,
            ) = _top_nodes_and_scores(
                feasibility_node_scores, rank_candidate_mask, max_attr_k
            )
        else:
            top_nodes_feasibility_all = None
            top_scores_feasibility_all = None
            top_nodes_feasibility_valid_all = None

        top_features_payload = _top_feature_payload(step_feature_attr_mean, top_n=3)
        step_constraint_attr_mean = _aggregate_constraint_scores(step_feature_attr_mean)
        top_constraints_payload = _top_constraint_payload(step_constraint_attr_mean, top_n=None)
        step_contrastive_constraint_attr_mean = _aggregate_constraint_scores(
            step_contrastive_feature_attr_mean
        )
        top_contrastive_constraints_payload = _top_constraint_payload(
            step_contrastive_constraint_attr_mean, top_n=None
        )
        for group_name, score in step_constraint_attr_mean.items():
            per_constraint_attr[group_name].append(float(score))
        for group_name, score in step_contrastive_constraint_attr_mean.items():
            per_constraint_contrastive_attr[group_name].append(float(score))

        topk_metrics: Dict[int, Dict[str, float]] = {}
        with torch.no_grad():
            for k in topk_list:
                node_perturbed, _, k_eff = _perturb_topk_nodes(
                    node_features,
                    _normalize_node_scores(node_scores.detach()),
                    customer_candidate_mask,
                    k,
                )
                common_del = _build_common(node_perturbed, global_features)
                cache_del = _encode_inputs(model, node_perturbed, global_features)
                logits_del, _, _, logprobs_del, _ = _step_logits_and_mask(
                    model, cache_del, common_del, state
                )
                sel_logit_del = logits_del.gather(-1, action.detach().unsqueeze(-1)).squeeze(-1)
                sel_logprob_del = logprobs_del.gather(-1, action.detach().unsqueeze(-1)).squeeze(-1)
                action_del = logprobs_del.argmax(dim=-1)

                logit_drop = selected_logit.detach() - sel_logit_del
                logprob_drop = selected_logprob.detach() - sel_logprob_del
                flip_rate = (action_del != action.detach()).float().mean()

                logit_drop_mean = float(logit_drop.mean().item())
                logprob_drop_mean = float(logprob_drop.mean().item())
                flip_rate_mean = float(flip_rate.item())

                per_k_logit_drop[k].append(logit_drop_mean)
                per_k_logprob_drop[k].append(logprob_drop_mean)
                per_k_flip_rate[k].append(flip_rate_mean)

                topk_metrics[k] = {
                    "k_effective": int(k_eff),
                    "logit_drop_mean": logit_drop_mean,
                    "mean_logit_drop": logit_drop_mean,
                    "logprob_drop_mean": logprob_drop_mean,
                    "mean_logprob_drop": logprob_drop_mean,
                    "action_flip_rate": flip_rate_mean,
                    "mean_action_flip_rate": flip_rate_mean,
                }

        next_state = _step_update_state(
            common,
            state,
            action.detach(),
            potential_distance.detach(),
            full_mask=full_mask.detach(),
            recourse_enabled=recourse_enabled,
        )
        done_after = ~next_state.not_served.any(dim=1)
        done_ratio = float(done_after.float().mean().item())

        record = _collect_step_record(
            step=step,
            action=action.detach(),
            selected_logit=selected_logit.detach(),
            selected_logprob=selected_logprob.detach(),
            node_scores=node_scores.detach(),
            decision_node_scores=decision_node_scores.detach(),
            feasibility_node_scores=feasibility_node_scores.detach(),
            use_feasibility_importance=use_feasibility_importance,
            topk_metrics=topk_metrics,
            top1_nodes=top1_nodes.detach(),
            done_ratio=done_ratio,
            top_features=top_features_payload,
            top_constraints=top_constraints_payload,
            feature_attr_mean=step_feature_attr_mean,
            constraint_attr_mean=step_constraint_attr_mean,
            contrastive_alt_available_rate=float(has_alt.float().mean().item()),
            contrastive_logit_gap_mean=_masked_mean(contrastive_logit_gap.detach(), has_alt),
            contrastive_logprob_gap_mean=_masked_mean(contrastive_logprob_gap.detach(), has_alt),
            contrastive_top_constraints=top_contrastive_constraints_payload,
            recourse_flags=recourse_flags.detach(),
            recourse_cost_est=recourse_cost_est.detach(),
            action_feasible=action_feasible.detach(),
            done_before=done_before.detach(),
            done_after=done_after.detach(),
        )
        if args.save_step_records:
            step_records.append(record)

        if num_store > 0:
            step_top_nodes = top_nodes_all[:num_store].detach().cpu().tolist()
            step_top_scores = top_scores_all[:num_store].detach().cpu().tolist()
            step_top_valid = top_nodes_valid_all[:num_store].detach().cpu().tolist()
            step_top_nodes_decision = top_nodes_decision_all[:num_store].detach().cpu().tolist()
            step_top_scores_decision = top_scores_decision_all[:num_store].detach().cpu().tolist()
            step_top_valid_decision = (
                top_nodes_decision_valid_all[:num_store].detach().cpu().tolist()
            )
            if use_feasibility_importance:
                step_top_nodes_feasibility = (
                    top_nodes_feasibility_all[:num_store].detach().cpu().tolist()
                    if top_nodes_feasibility_all is not None
                    else []
                )
                step_top_scores_feasibility = (
                    top_scores_feasibility_all[:num_store].detach().cpu().tolist()
                    if top_scores_feasibility_all is not None
                    else []
                )
                step_top_valid_feasibility = (
                    top_nodes_feasibility_valid_all[:num_store].detach().cpu().tolist()
                    if top_nodes_feasibility_valid_all is not None
                    else []
                )
            else:
                step_top_nodes_feasibility = []
                step_top_scores_feasibility = []
                step_top_valid_feasibility = []
            alt_action_store = alt_action[:num_store].detach().cpu().tolist()
            has_alt_store = has_alt[:num_store].detach().cpu().tolist()
            contrastive_logit_gap_store = contrastive_logit_gap[:num_store].detach().cpu().tolist()
            contrastive_logprob_gap_store = contrastive_logprob_gap[:num_store].detach().cpu().tolist()
            alt_source_store = contrastive_source_code[:num_store].detach().cpu().tolist()
            alt_feasible_store = alt_action_feasible[:num_store].detach().cpu().tolist()
            alt_recourse_store = alt_recourse_flags[:num_store].detach().cpu().tolist()
            action_feasible_store = action_feasible[:num_store].detach().cpu().tolist()
            recourse_store = recourse_flags[:num_store].detach().cpu().tolist()
            recourse_cost_store = recourse_cost_est[:num_store].detach().cpu().tolist()
            for i in range(num_store):
                inst_feat_scores = {
                    key: float(instance_feature_attr[key][i].item())
                    for key in instance_feature_attr
                }
                inst_top_features = _top_feature_payload(inst_feat_scores, top_n=3)
                inst_constraint_scores = _aggregate_constraint_scores(inst_feat_scores)
                inst_top_constraints = _top_constraint_payload(inst_constraint_scores, top_n=None)

                inst_contrastive_feat_scores = {
                    key: float(instance_contrastive_feature_attr[key][i].item())
                    for key in instance_contrastive_feature_attr
                }
                inst_contrastive_constraint_scores = _aggregate_constraint_scores(
                    inst_contrastive_feat_scores
                )
                inst_top_contrastive_constraints = _top_constraint_payload(
                    inst_contrastive_constraint_scores, top_n=None
                )
                counterfactual_payload = _propose_counterfactual(
                    model=model,
                    node_features=node_features,
                    global_features=global_features,
                    state=state,
                    batch_index=i,
                    alt_action=int(alt_action_store[i]),
                    has_alt=bool(has_alt_store[i]),
                    contrastive_logit_gap=float(contrastive_logit_gap_store[i]),
                    contrastive_grad_by_feature=contrastive_grad_by_feature,
                    full_mask=full_mask.detach(),
                    recourse_enabled=recourse_enabled,
                )
                if counterfactual_payload is None:
                    counterfactual_available_history.append(0.0)
                    counterfactual_switch_history.append(0.0)
                    counterfactual_make_feasible_history.append(0.0)
                    counterfactual_approximate_history.append(0.0)
                else:
                    status = str(counterfactual_payload.get("status", "approximate"))
                    feature_name = str(counterfactual_payload.get("feature", "unknown"))
                    counterfactual_available_history.append(1.0)
                    counterfactual_switch_history.append(1.0 if status == "switch" else 0.0)
                    counterfactual_make_feasible_history.append(
                        1.0 if status == "make_feasible" else 0.0
                    )
                    counterfactual_approximate_history.append(
                        1.0 if status == "approximate" else 0.0
                    )
                    rel_delta = counterfactual_payload.get("relative_delta", None)
                    if rel_delta is not None:
                        try:
                            rel_delta_val = float(rel_delta)
                        except (TypeError, ValueError):
                            rel_delta_val = float("nan")
                        if math.isfinite(rel_delta_val):
                            counterfactual_relative_delta_history.append(rel_delta_val)
                    counterfactual_by_feature[feature_name].append(1.0)

                instance_traces[i]["done_before"].append(bool(done_before[i].item()))
                instance_traces[i]["done_after"].append(bool(done_after[i].item()))
                instance_traces[i]["actions"].append(int(action[i].item()))
                instance_traces[i]["top_nodes"].append(
                    [
                        int(node)
                        for node, valid in zip(step_top_nodes[i], step_top_valid[i])
                        if bool(valid)
                    ]
                )
                instance_traces[i]["top_scores"].append(
                    [
                        float(score)
                        for score, valid in zip(step_top_scores[i], step_top_valid[i])
                        if bool(valid)
                    ]
                )
                instance_traces[i]["top_nodes_decision"].append(
                    [
                        int(node)
                        for node, valid in zip(
                            step_top_nodes_decision[i], step_top_valid_decision[i]
                        )
                        if bool(valid)
                    ]
                )
                instance_traces[i]["top_scores_decision"].append(
                    [
                        float(score)
                        for score, valid in zip(
                            step_top_scores_decision[i], step_top_valid_decision[i]
                        )
                        if bool(valid)
                    ]
                )
                if use_feasibility_importance:
                    instance_traces[i]["top_nodes_feasibility"].append(
                        [
                            int(node)
                            for node, valid in zip(
                                step_top_nodes_feasibility[i],
                                step_top_valid_feasibility[i],
                            )
                            if bool(valid)
                        ]
                    )
                    instance_traces[i]["top_scores_feasibility"].append(
                        [
                            float(score)
                            for score, valid in zip(
                                step_top_scores_feasibility[i],
                                step_top_valid_feasibility[i],
                            )
                            if bool(valid)
                        ]
                    )
                else:
                    instance_traces[i]["top_nodes_feasibility"].append([])
                    instance_traces[i]["top_scores_feasibility"].append([])
                instance_traces[i]["top_features"].append(inst_top_features)
                instance_traces[i]["top_constraints"].append(inst_top_constraints)
                if has_alt_store[i]:
                    instance_traces[i]["contrastive_alt_action"].append(int(alt_action_store[i]))
                    if alt_source_store[i] == 2:
                        contrastive_source = "full_feasible"
                    elif alt_source_store[i] == 1:
                        contrastive_source = "policy_masked"
                    else:
                        contrastive_source = "none"
                    instance_traces[i]["contrastive_alt_source"].append(contrastive_source)
                    instance_traces[i]["contrastive_alt_feasible"].append(
                        bool(alt_feasible_store[i])
                    )
                    instance_traces[i]["contrastive_alt_recourse"].append(
                        bool(alt_recourse_store[i])
                    )
                    instance_traces[i]["contrastive_logit_gap"].append(
                        float(contrastive_logit_gap_store[i])
                    )
                    instance_traces[i]["contrastive_logprob_gap"].append(
                        float(contrastive_logprob_gap_store[i])
                    )
                    instance_traces[i]["contrastive_top_constraints"].append(
                        inst_top_contrastive_constraints
                    )
                else:
                    instance_traces[i]["contrastive_alt_action"].append(-1)
                    instance_traces[i]["contrastive_alt_source"].append("none")
                    instance_traces[i]["contrastive_alt_feasible"].append(False)
                    instance_traces[i]["contrastive_alt_recourse"].append(False)
                    instance_traces[i]["contrastive_logit_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_logprob_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_top_constraints"].append([])
                instance_traces[i]["chosen_feasible"].append(bool(action_feasible_store[i]))
                instance_traces[i]["recourse_triggered"].append(bool(recourse_store[i]))
                instance_traces[i]["recourse_cost_est"].append(
                    float(recourse_cost_store[i])
                )
                instance_traces[i]["counterfactuals"].append(counterfactual_payload)

        state = next_state

    final_costs = state.total_distance.squeeze(1)
    done_all = not bool(state.not_served.any().item())
    trajectory_summary = _summarize_trajectory(instance_traces)

    summary = {
        "num_instances": int(args.num_instances),
        "num_steps": int(executed_steps),
        "done_all": bool(done_all),
        "mean_final_reward": float((-final_costs).mean().item()) if final_costs.numel() > 0 else None,
        "mean_final_cost": float(final_costs.mean().item()) if final_costs.numel() > 0 else None,
        "chosen_action_feasible_rate": _safe_mean(chosen_feasible_rate_history),
        "recourse_event_rate": _safe_mean(recourse_rate_history),
        "recourse_cost_est_mean": _safe_mean(recourse_cost_est_history),
        "deletion_faithfulness": {
            str(k): {
                "mean_logit_drop": _safe_mean(per_k_logit_drop[k]),
                "mean_logprob_drop": _safe_mean(per_k_logprob_drop[k]),
                "mean_action_flip_rate": _safe_mean(per_k_flip_rate[k]),
            }
            for k in topk_list
        },
        "feature_importance_mean": {
            key: _safe_mean(per_feature_attr[key]) for key in sorted(per_feature_attr.keys())
        },
        "constraint_importance_mean": {
            key: _safe_mean(per_constraint_attr[key])
            for key in sorted(per_constraint_attr.keys())
        },
        "contrastive": {
            "alt_available_rate": _safe_mean(contrastive_alt_available_history),
            "mean_logit_gap": _safe_mean(contrastive_logit_gap_history),
            "mean_logprob_gap": _safe_mean(contrastive_logprob_gap_history),
            "feature_importance_mean": {
                key: _safe_mean(per_feature_contrastive_attr[key])
                for key in sorted(per_feature_contrastive_attr.keys())
            },
            "constraint_importance_mean": {
                key: _safe_mean(per_constraint_contrastive_attr[key])
                for key in sorted(per_constraint_contrastive_attr.keys())
            },
        },
        "node_importance": {
            "mode": importance_mode,
            "feasibility_weight": float(feasibility_weight),
            "feasibility_top_m": int(feasibility_top_m),
            "feasibility_cost_weight": float(feasibility_cost_weight),
            "decision_mean": _safe_mean(decision_node_attr_history),
            "feasibility_mean": _safe_mean(feasibility_node_attr_history),
        },
        "counterfactuals": {
            "available_rate": _safe_mean(counterfactual_available_history),
            "switch_rate": _safe_mean(counterfactual_switch_history),
            "make_feasible_rate": _safe_mean(counterfactual_make_feasible_history),
            "approximate_rate": _safe_mean(counterfactual_approximate_history),
            "mean_relative_delta": _safe_mean(counterfactual_relative_delta_history),
            "feature_frequency": {
                key: _safe_mean(counterfactual_by_feature[key])
                for key in sorted(counterfactual_by_feature.keys())
            },
        },
        "trajectory": trajectory_summary,
        "instance_variant_counts": {
            key: sum(1 for meta in variant_meta if meta["code"] == key)
            for key in sorted({meta["code"] for meta in variant_meta})
        },
    }

    feat_total = sum(summary["feature_importance_mean"].values())
    summary["feature_importance_share"] = {
        key: (value / feat_total if feat_total > 0 else 0.0)
        for key, value in summary["feature_importance_mean"].items()
    }

    constraint_total = sum(summary["constraint_importance_mean"].values())
    summary["constraint_importance_share"] = {
        key: (value / constraint_total if constraint_total > 0 else 0.0)
        for key, value in summary["constraint_importance_mean"].items()
    }

    contrastive_feature_total = sum(summary["contrastive"]["feature_importance_mean"].values())
    summary["contrastive"]["feature_importance_share"] = {
        key: (value / contrastive_feature_total if contrastive_feature_total > 0 else 0.0)
        for key, value in summary["contrastive"]["feature_importance_mean"].items()
    }

    contrastive_constraint_total = sum(summary["contrastive"]["constraint_importance_mean"].values())
    summary["contrastive"]["constraint_importance_share"] = {
        key: (value / contrastive_constraint_total if contrastive_constraint_total > 0 else 0.0)
        for key, value in summary["contrastive"]["constraint_importance_mean"].items()
    }

    cf_total = sum(summary["counterfactuals"]["feature_frequency"].values())
    summary["counterfactuals"]["feature_share"] = {
        key: (value / cf_total if cf_total > 0 else 0.0)
        for key, value in summary["counterfactuals"]["feature_frequency"].items()
    }

    run_name = repr(config)
    run_group = f"{config.problem}/{config.graph_size}"
    model_label = f"{run_group}/{run_name}"
    model_slug = _slugify(model_label)

    report = {
        "timestamp": int(time.time()),
        "config": {
            "seed": int(args.seed) if args.seed is not None else None,
            "data_seed": int(data_seed) if data_seed is not None else None,
            "device": str(config.device),
            "num_instances": int(args.num_instances),
            "max_steps": int(args.max_steps),
            "topk_nodes": [int(k) for k in topk_list],
            "attr_features": list(selected_features),
            "constraint_groups": sorted({_feature_to_constraint_group(name) for name in selected_features}),
            "instance_variant_encoding": "['o' if open_route] + vrp + ['m' if mixed_backhaul] + ['b' if backhaul] + ['l' if distance_limit] + ['tw' if time_windows]",
            "model_target": "mavrp.env.models.TransformerModel",
            "encoder_target": f"{config.encoder.__module__}.{config.encoder.__name__}",
            "decoder_target": f"{config.decoder.__module__}.{config.decoder.__name__}",
            "run_name": run_name,
            "run_group": run_group,
            "model_label": model_label,
            "model_slug": model_slug,
            "config_id": config_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_path_resolved": str(checkpoint_path.resolve()),
            "checkpoint_kind": checkpoint_path.name,
            "randomize_weights": bool(args.randomize_weights),
            "model_state": "randomized" if args.randomize_weights else "trained",
            "node_importance_mode": importance_mode,
            "feasibility_weight": float(feasibility_weight),
            "feasibility_top_m": int(feasibility_top_m),
            "feasibility_cost_weight": float(feasibility_cost_weight),
            "problem": config.problem,
            "graph_size": int(config.graph_size),
            "config_repr": run_name,
        },
        "summary": summary,
    }

    if args.save_step_records:
        report["steps"] = step_records
    if num_store > 0:
        for trace in instance_traces:
            done_step = None
            for i, flag in enumerate(trace["done_after"]):
                if flag:
                    done_step = i
                    break
            trace["done_step"] = done_step
            trace["top_k_effective"] = int(top_k_effective)
        report["instances"] = instance_traces

    random_suffix = "_randomized" if args.randomize_weights else ""
    output_path = output_dir / f"action_explainer_{model_slug}{random_suffix}_{int(time.time())}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved XAI report to {output_path}")
    return output_path


def _preflight_check() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch_geometric"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "The current g-unirouting environment cannot import torch_geometric "
            "(or one of its native dependencies such as torch_cluster). "
            "Fix the target .venv before running action_explainer."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Action-level explainability for g-unirouting TransformerModel checkpoints."
    )
    parser.add_argument(
        "--config-id",
        type=int,
        default=None,
        help="Index in Config.all(). Optional if --checkpoint can be matched to a config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to baseline.pt or checkpoint.ckpt. If omitted, uses the resolved config default baseline.pt path.",
    )
    parser.add_argument("--problem", default=None, help="Optional config.problem override.")
    parser.add_argument("--graph-size", type=int, default=None, help="Optional config.graph_size override.")
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument(
        "--attribution-methods",
        default="gradient,integrated_gradients",
        help=(
            "Comma-separated attribution methods to generate. "
            "Supported: gradient, integrated_gradients. "
            "Default generates both and writes a bundle JSON."
        ),
    )
    parser.add_argument(
        "--feasibility-weight",
        type=float,
        default=None,
        help=(
            "Weight of the feasibility-sensitive node importance term. "
            "Defaults to 1.0 for RecourseDecoder and 0.0 otherwise."
        ),
    )
    parser.add_argument(
        "--feasibility-top-m",
        type=int,
        default=8,
        help="Top-M customer nodes probed for feasibility sensitivity (0 = all customers).",
    )
    parser.add_argument(
        "--feasibility-cost-weight",
        type=float,
        default=0.25,
        help="Relative weight of recourse-cost sensitivity inside the feasibility term.",
    )
    parser.add_argument(
        "--attr-features",
        default="auto",
        help="Comma-separated feature names or 'auto'.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Number of interpolation points used by Integrated Gradients when requested.",
    )
    parser.add_argument(
        "--ig-baseline",
        choices=[
            "mean-fill",
            "zero-with-customers-at-depot",
            "zero-all",
            "zero-with-current-locs",
        ],
        default="mean-fill",
        help="Baseline used by Integrated Gradients when requested.",
    )
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, mps).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--randomize-weights",
        action="store_true",
        help="Reset model weights after loading the architecture/checkpoint (used for XAI sanity checks).",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Optional seed used specifically for dataset generation. "
            "Defaults to --seed. Use this to align exactly the same sampled instances "
            "across different model architectures."
        ),
    )
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-step summary records.",
    )
    parser.add_argument(
        "--save-instance-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-instance traces used by plotting/text scripts.",
    )
    return parser


def _parse_attribution_methods(raw: str) -> List[str]:
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        return ["gradient", "integrated_gradients"]

    normalized: List[str] = []
    for token in tokens:
        if token in {"gradient", "grad", "saliency", "gradient_local"}:
            key = "gradient"
        elif token in {"integrated_gradients", "ig"}:
            key = "integrated_gradients"
        else:
            raise ValueError(
                f"Unsupported attribution method {token!r}. "
                "Use 'gradient' and/or 'integrated_gradients'."
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def _load_ig_module() -> Any:
    xai_dir = Path(__file__).resolve().parent
    if str(xai_dir) not in sys.path:
        sys.path.insert(0, str(xai_dir))
    import integrated_gradients_explainer as module  # type: ignore

    return module


def _build_bundle_report(
    output_dir: Path,
    gradient_report_path: Optional[Path],
    ig_report_path: Optional[Path],
) -> Optional[Path]:
    report_paths = {
        "gradient": gradient_report_path,
        "integrated_gradients": ig_report_path,
    }
    present = {key: value for key, value in report_paths.items() if value is not None}
    if len(present) <= 1:
        return None

    payload: Dict[str, Any] = {
        "kind": "xai_dual_bundle",
        "timestamp": int(time.time()),
        "reports": {
            key: {
                "path": str(path),
                "path_resolved": str(path.resolve()),
            }
            for key, path in present.items()
        },
    }

    model_slug = "bundle"
    model_label_base = ""
    config_summary: Dict[str, Any] = {}
    for key, path in present.items():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        cfg = data.get("config", {}) or {}
        if not model_label_base:
            model_label_base = str(cfg.get("model_label_base", "")).strip()
        if model_slug == "bundle":
            model_slug = str(cfg.get("model_slug", "")).strip() or model_slug
        config_summary[key] = {
            "model_label": cfg.get("model_label"),
            "attribution_method": cfg.get("attribution_method"),
            "checkpoint_path": cfg.get("checkpoint_path"),
            "checkpoint_path_resolved": cfg.get("checkpoint_path_resolved"),
        }

    if model_label_base:
        payload["model_label_base"] = model_label_base
    if config_summary:
        payload["config"] = config_summary

    bundle_path = output_dir / f"xai_bundle_{_slugify(model_slug)}_{int(time.time())}.json"
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved XAI bundle to {bundle_path}")
    return bundle_path


def _run_requested_methods(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    methods = _parse_attribution_methods(args.attribution_methods)
    out: Dict[str, Optional[Path]] = {"gradient": None, "integrated_gradients": None}

    if "gradient" in methods:
        out["gradient"] = run(args)

    if "integrated_gradients" in methods:
        ig_module = _load_ig_module()
        out["integrated_gradients"] = ig_module.run(
            argparse.Namespace(**vars(args)), grad_base_module=sys.modules[__name__]
        )

    _build_bundle_report(
        output_dir=Path(args.output_dir),
        gradient_report_path=out["gradient"],
        ig_report_path=out["integrated_gradients"],
    )
    return out


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _preflight_check()
    _run_requested_methods(args)


if __name__ == "__main__":
    main()
