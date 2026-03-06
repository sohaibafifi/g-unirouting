"""Pure decode helpers: problem tensor pre-computation and action masks.

These functions are shared between the training decoders (recourse.py,
endtoend.py) and the XAI engine (xai-poo/engine/decode_ops.py).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from mavrp.env.decoders.types import DecodeState


def build_common(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pre-compute all problem tensors needed every decode step."""
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


def estimate_recourse_trip_cost(
    common: Dict[str, torch.Tensor], action: torch.Tensor
) -> torch.Tensor:
    """Depot-out-and-back distance for a violated node selection."""
    deltas = common["deltas"]
    batch_indices = torch.arange(action.size(0), device=action.device)
    return deltas[batch_indices, 0, action] + deltas[batch_indices, action, 0]


def compute_action_masks(
    common: Dict[str, torch.Tensor],
    state: DecodeState,
    recourse_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (policy_mask, full_mask, potential_distance).

    Both masks use True = excluded (masked_fill convention).
    policy_mask: visited-only mask for RecourseDecoder; full constraint mask for E2E.
    full_mask:   always the full constraint-feasibility mask.
    """
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
