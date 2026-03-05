"""Pure decode functions (relocated from action_explainer.py, no logic change)."""
from __future__ import annotations

import math

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mavrp.env.models import TransformerModel

from domain.constants import CONSTRAINT_GROUP_RULES, COUNTERFACTUAL_SPECS, FEATURE_SPEC_BY_NAME
from engine.decode_types import DecodeCache, DecodeState
from mavrp.env.decoders.ops import (
    build_common,
    compute_action_masks,
    estimate_recourse_trip_cost,
)
from utils.math_utils import safe_mean


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def safe_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def is_recourse_decoder(model: "TransformerModel") -> bool:
    return model.decoder.__class__.__name__ == "RecourseDecoder"


# ---------------------------------------------------------------------------
# Variant metadata
# ---------------------------------------------------------------------------


def variant_metadata_from_inputs(
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


def init_instance_traces(
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
            "contrastive_policy_alt_action": [],
            "contrastive_policy_alt_feasible": [],
            "contrastive_policy_alt_recourse": [],
            "contrastive_policy_logit_gap": [],
            "contrastive_policy_logprob_gap": [],
            "contrastive_feasible_alt_action": [],
            "contrastive_feasible_alt_feasible": [],
            "contrastive_feasible_alt_recourse": [],
            "contrastive_feasible_logit_gap": [],
            "contrastive_feasible_logprob_gap": [],
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


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------


def loc_distance(locs: Sequence[Sequence[float]], src: int, dst: int) -> float:
    if not (0 <= src < len(locs) and 0 <= dst < len(locs)):
        return float("nan")
    src_xy = locs[src]
    dst_xy = locs[dst]
    if len(src_xy) < 2 or len(dst_xy) < 2:
        return float("nan")
    return float(math.hypot(float(src_xy[0]) - float(dst_xy[0]), float(src_xy[1]) - float(dst_xy[1])))


def first_true_index(flags: Sequence[bool]) -> Optional[int]:
    for idx, flag in enumerate(flags):
        if bool(flag):
            return int(idx)
    return None


def count_true_bursts(flags: Sequence[bool]) -> int:
    bursts = 0
    prev = False
    for flag in flags:
        curr = bool(flag)
        if curr and not prev:
            bursts += 1
        prev = curr
    return int(bursts)


def mean_constraint_share_per_step(payloads: Sequence[Any]) -> Dict[str, float]:
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
    return {name: safe_mean(values) for name, values in per_group.items() if values}


def dominant_constraint_name(shares: Dict[str, float]) -> str:
    if not shares:
        return "none"
    best_name, best_value = max(shares.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    return str(best_name) if float(best_value) > 0 else "none"


def summarize_trajectory(instance_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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

        first_depot = next(
            (idx for idx, action in enumerate(actions) if action == 0), None
        )
        if first_depot is not None:
            first_depot_step.append(float(first_depot))
            first_depot_step_norm.append(float(first_depot / max(step_count, 1)))

        per_action_distances: List[float] = []
        per_customer_hops: List[float] = []
        current = 0
        for action in actions:
            hop = loc_distance(locs, current, action)
            if math.isfinite(hop):
                per_action_distances.append(hop)
                if action > 0 and current > 0:
                    per_customer_hops.append(hop)
            current = action
        action_distance_mean.append(safe_mean(per_action_distances))
        customer_hop_distance_mean.append(safe_mean(per_customer_hops))

        recourse_events_per_trace.append(float(sum(1 for flag in recourse_flags if flag)))
        recourse_burst_count.append(float(count_true_bursts(recourse_flags)))
        first_recourse = first_true_index(recourse_flags)
        if first_recourse is not None:
            first_recourse_step.append(float(first_recourse))
            first_recourse_step_norm.append(float(first_recourse / max(step_count, 1)))

        split_idx = max(1, step_count // 2)
        early_share = mean_constraint_share_per_step(top_constraints[:split_idx])
        late_share = mean_constraint_share_per_step(top_constraints[split_idx:])
        if not late_share and early_share:
            late_share = dict(early_share)
        for name, value in early_share.items():
            early_constraint_terms[name].append(float(value))
        for name, value in late_share.items():
            late_constraint_terms[name].append(float(value))

    early_constraint_share = {
        name: safe_mean(values)
        for name, values in sorted(early_constraint_terms.items())
    }
    late_constraint_share = {
        name: safe_mean(values)
        for name, values in sorted(late_constraint_terms.items())
    }
    early_top = dominant_constraint_name(early_constraint_share)
    late_top = dominant_constraint_name(late_constraint_share)

    return {
        "basis": "stored_instance_traces",
        "num_traces": int(len([trace for trace in instance_traces if trace.get("actions")])),
        "stored_step_mean": safe_mean(stored_steps),
        "mean_customer_actions_per_instance": safe_mean(customer_actions_per_trace),
        "mean_depot_returns_per_instance": safe_mean(depot_returns_per_trace),
        "mean_depot_return_share": safe_mean(depot_return_share_per_trace),
        "mean_first_depot_return_step": safe_mean(first_depot_step),
        "mean_first_depot_return_step_norm": safe_mean(first_depot_step_norm),
        "mean_action_distance": safe_mean(action_distance_mean),
        "mean_customer_hop_distance": safe_mean(customer_hop_distance_mean),
        "mean_recourse_events_per_instance": safe_mean(recourse_events_per_trace),
        "mean_recourse_burst_count": safe_mean(recourse_burst_count),
        "mean_first_recourse_step": safe_mean(first_recourse_step),
        "mean_first_recourse_step_norm": safe_mean(first_recourse_step_norm),
        "early_constraint_share": early_constraint_share,
        "late_constraint_share": late_constraint_share,
        "early_top_constraint": early_top,
        "late_top_constraint": late_top,
        "dominant_constraint_shift": f"{early_top}->{late_top}",
    }


# ---------------------------------------------------------------------------
# Encode inputs
# ---------------------------------------------------------------------------


def encode_inputs(
    model: "TransformerModel",
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> DecodeCache:
    decoder = model.decoder
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


# ---------------------------------------------------------------------------
# Common pre-computed tensors — imported from mavrp.env.decoders.ops
# ---------------------------------------------------------------------------
# build_common, compute_action_masks, estimate_recourse_trip_cost are imported
# at the top of this file. Local definitions have been removed to avoid drift.


def init_state(common: Dict[str, torch.Tensor]) -> DecodeState:
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


# ---------------------------------------------------------------------------
# Action masks — imported from mavrp.env.decoders.ops (see top of file)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logits and step execution
# ---------------------------------------------------------------------------


def step_logits_and_mask(
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

    recourse_enabled = is_recourse_decoder(model)
    policy_mask, full_mask, potential_distance = compute_action_masks(
        common, state, recourse_enabled
    )

    last = cache.node_embeddings[batch_indices, current, :].unsqueeze(1)

    # Run GRU for RecourseDecoder and thread hidden state through DecodeState.
    # Without this step the GRU hidden is always zero (latent bug fixed here).
    if recourse_enabled:
        last, new_hidden = decoder.history(last, state.hidden)
        state.hidden = new_hidden  # side-effect: propagate to step_update_state

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


def step_update_state(
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
        recourse_cost = estimate_recourse_trip_cost(common, selected_node).unsqueeze(-1)
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

    # Thread GRU hidden state through; reset on route-end (going to depot normally).
    hidden = state.hidden
    if hidden is not None and bool(selected_is_depot.any().item()):
        end_route = selected_is_depot & ~recourse_triggered  # = selected_is_depot (depot never recourses)
        if bool(end_route.any().item()):
            hidden = hidden.clone()
            hidden[:, end_route, :] = 0.0

    return DecodeState(
        current_node=current_node.detach(),
        not_served=not_served.detach(),
        leave_time=leave_time.detach(),
        deliveries=deliveries.detach(),
        pickups=pickups.detach(),
        distance=distance.detach(),
        total_distance=total_distance.detach(),
        is_depot=is_depot.detach(),
        hidden=hidden,
    )


# ---------------------------------------------------------------------------
# Feature gradient extraction
# ---------------------------------------------------------------------------


def feature_slice(tensor: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
    index = spec["index"]
    if tensor.dim() == 3:
        return tensor[:, :, index]
    return tensor[:, index]


def extract_feature_grads(
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
            out[name] = feature_slice(grad_source, spec)
    return out


def grad_to_instance_scores(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grad is None:
        return None
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.abs().reshape(grad.shape[0], -1).mean(dim=-1)


def grad_to_node_scores(
    grad: Optional[torch.Tensor], num_nodes: int
) -> Optional[torch.Tensor]:
    if grad is None or grad.dim() < 2:
        return None
    if grad.shape[1] != num_nodes:
        return None
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    scores = grad.abs()
    if scores.dim() > 2:
        scores = scores.sum(dim=tuple(range(2, scores.dim())))
    return scores


# ---------------------------------------------------------------------------
# Node scoring helpers
# ---------------------------------------------------------------------------


def normalize_node_scores(node_scores: torch.Tensor) -> torch.Tensor:
    scores = node_scores.clone()
    if scores.size(-1) <= 1:
        return scores.zero_()
    customer_scores = scores[:, 1:]
    max_scores = customer_scores.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    scores[:, 1:] = customer_scores / max_scores
    scores[:, 0] = 0.0
    return scores


def top_nodes_and_scores(
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


def perturb_topk_nodes(
    node_features: torch.Tensor,
    node_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
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


# ---------------------------------------------------------------------------
# Recourse helpers
# ---------------------------------------------------------------------------
# estimate_recourse_trip_cost is imported from mavrp.env.decoders.ops (see top).
# Note: open-route cost is zero because build_common sets deltas[:,:,0]=0 for
# open-route batches, so deltas[b, action, 0] is already 0 there.


def infer_recourse_for_action(
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
        recourse_trip_cost = estimate_recourse_trip_cost(common, action)
        recourse_cost_est = torch.where(
            recourse_flags, recourse_trip_cost, torch.zeros_like(recourse_trip_cost)
        )
    return recourse_flags, recourse_cost_est, action_feasible


# ---------------------------------------------------------------------------
# Contrastive alternative selection
# ---------------------------------------------------------------------------


def select_best_alternative(
    logits: torch.Tensor,
    logprobs: torch.Tensor,
    chosen_action: torch.Tensor,
    action_mask: torch.Tensor,
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


# ---------------------------------------------------------------------------
# Feasibility node scoring
# ---------------------------------------------------------------------------


def compute_feasibility_node_scores(
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

    common_base = build_common(node_features, global_features)
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
        common_pert = build_common(node_perturbed, global_features)
        _, full_mask_pert, _ = compute_action_masks(common_pert, state, recourse_enabled)
        _, recourse_cost_pert, feasible_pert = infer_recourse_for_action(
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


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def slice_state(state: DecodeState, index: int) -> DecodeState:
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


# ---------------------------------------------------------------------------
# Counterfactual proposal
# ---------------------------------------------------------------------------


def apply_counterfactual_change(
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


def replay_single_step(
    model: "TransformerModel",
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    state: DecodeState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    common = build_common(node_features, global_features)
    cache = encode_inputs(model, node_features, global_features)
    logits, _, full_mask, logprobs, _ = step_logits_and_mask(model, cache, common, state)
    action = logprobs.argmax(dim=-1)
    return action, full_mask, logits


def propose_counterfactual(
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
    state_one = slice_state(state, batch_index)
    node_one = node_features[batch_index : batch_index + 1].detach()
    global_one = global_features[batch_index : batch_index + 1].detach()

    best_attempt: Optional[Dict[str, Any]] = None
    with torch.no_grad():
        for row in candidate_rows[:3]:
            spec = row["spec"]
            delta_apply = float(row["delta"]) * 1.05
            node_cf, global_cf, new_value = apply_counterfactual_change(
                node_one, global_one, spec, alt_action, delta_apply
            )
            action_cf, full_mask_cf, _ = replay_single_step(model, node_cf, global_cf, state_one)
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


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def parse_topk_nodes(raw: str) -> List[int]:
    if isinstance(raw, str):
        tokens = str(raw).strip().strip("[]")
        values = [int(tok.strip()) for tok in tokens.split(",") if tok.strip()]
        values = sorted({v for v in values if v > 0})
        return values or [1, 3, 5]
    return [1, 3, 5]


def parse_attr_features(raw: Optional[str]) -> List[str]:
    from domain.constants import FEATURE_SPECS, FEATURE_SPEC_BY_NAME

    if raw is None or raw.lower() == "auto":
        selected = [spec["name"] for spec in FEATURE_SPECS]
    else:
        selected = [token.strip() for token in raw.split(",") if token.strip()]
        selected = [name for name in selected if name in FEATURE_SPEC_BY_NAME]
    if "locs" not in selected:
        selected = ["locs"] + selected
    return selected


def parse_feasibility_weight(raw_weight: Optional[float], recourse_enabled: bool) -> float:
    if raw_weight is None:
        return 1.0 if recourse_enabled else 0.0
    return max(float(raw_weight), 0.0)


def parse_feasibility_top_m(raw_top_m: int) -> int:
    return max(int(raw_top_m), 0)


def parse_feasibility_cost_weight(raw_cost_weight: float) -> float:
    return max(float(raw_cost_weight), 0.0)
