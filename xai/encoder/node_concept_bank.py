from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from encoder.encoder_probe import _quantile_states


NODE_CORE_CONCEPT_NAMES: List[str] = [
    "service_order_state",
    "route_role_state",
    "local_cost_contribution_state",
]

NODE_CONCEPT_DISPLAY_NAMES: Dict[str, str] = {
    "service_order_state": "service order",
    "route_role_state": "route role",
    "local_cost_contribution_state": "local cost impact",
}

NODE_CONCEPT_CLASS_ORDERS: Dict[str, List[str]] = {
    "service_order_state": ["served_early", "served_middle", "served_late"],
    "route_role_state": [
        "singleton_route",
        "route_start",
        "route_middle",
        "route_end",
    ],
    "local_cost_contribution_state": [
        "low_local_cost_impact",
        "medium_local_cost_impact",
        "high_local_cost_impact",
    ],
}


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _summarize_raw(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _step_distance(
    locations: np.ndarray,
    src: int,
    dst: int,
    open_route: bool,
) -> float:
    if src == dst:
        return 0.0
    if open_route and dst == 0:
        return 0.0
    src_xy = locations[src]
    dst_xy = locations[dst]
    return float(np.linalg.norm(src_xy - dst_xy))


def _extract_routes(route_tokens: Sequence[int], num_customers: int) -> List[List[int]]:
    routes: List[List[int]] = []
    current_route: List[int] = []
    seen: set[int] = set()

    for token in route_tokens:
        node = int(token)
        if node == 0:
            if current_route:
                routes.append(current_route)
                current_route = []
            if len(seen) >= num_customers:
                break
            continue
        if node < 1 or node > num_customers or node in seen:
            continue
        current_route.append(node)
        seen.add(node)

    if current_route:
        routes.append(current_route)

    missing = [node for node in range(1, num_customers + 1) if node not in seen]
    routes.extend([[node] for node in missing])
    return routes


def _route_total_cost(
    routes: Sequence[Sequence[int]],
    locations: np.ndarray,
    open_route: bool,
) -> float:
    total_cost = 0.0
    for route in routes:
        prev = 0
        for node in route:
            total_cost += _step_distance(locations, prev, int(node), open_route)
            prev = int(node)
        total_cost += _step_distance(locations, prev, 0, open_route)
    return total_cost


def _service_order_state_from_rank(rank_normalized: float) -> str:
    if rank_normalized < (1.0 / 3.0):
        return "served_early"
    if rank_normalized < (2.0 / 3.0):
        return "served_middle"
    return "served_late"


def compute_node_concept_bank(
    node_features: torch.Tensor,
    routes: torch.Tensor,
    metadata: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    node_np = _to_numpy(node_features)
    route_np = _to_numpy(routes).astype(np.int64, copy=False)

    batch_size = node_np.shape[0]
    num_customers = node_np.shape[1] - 1

    service_order_normalized = np.zeros((batch_size, num_customers), dtype=np.float64)
    local_cost_share = np.zeros((batch_size, num_customers), dtype=np.float64)
    route_role_state = np.full(
        (batch_size, num_customers),
        "singleton_route",
        dtype=object,
    )

    for instance_idx in range(batch_size):
        locations = node_np[instance_idx, :, :2]
        open_route = bool(metadata[instance_idx]["flags"]["open_route"])
        parsed_routes = _extract_routes(route_np[instance_idx].tolist(), num_customers)
        flat_visit_order = [node for route in parsed_routes for node in route]
        order_denominator = max(num_customers - 1, 1)

        for visit_rank, node in enumerate(flat_visit_order):
            service_order_normalized[instance_idx, node - 1] = float(
                visit_rank / order_denominator
            )

        total_cost = max(_route_total_cost(parsed_routes, locations, open_route), 1e-6)
        for route in parsed_routes:
            route_length = len(route)
            for position, node in enumerate(route):
                if route_length == 1:
                    role = "singleton_route"
                elif position == 0:
                    role = "route_start"
                elif position == route_length - 1:
                    role = "route_end"
                else:
                    role = "route_middle"
                route_role_state[instance_idx, node - 1] = role

                prev_node = 0 if position == 0 else int(route[position - 1])
                next_node = 0 if position == route_length - 1 else int(route[position + 1])
                local_delta = (
                    _step_distance(locations, prev_node, int(node), open_route)
                    + _step_distance(locations, int(node), next_node, open_route)
                    - _step_distance(locations, prev_node, next_node, open_route)
                )
                local_cost_share[instance_idx, node - 1] = float(local_delta / total_cost)

    flattened_service_order = service_order_normalized.reshape(-1)
    flattened_route_role = route_role_state.reshape(-1).astype(str)
    flattened_local_cost_share = local_cost_share.reshape(-1)

    concept_states: Dict[str, List[str]] = {
        "service_order_state": [
            _service_order_state_from_rank(float(value))
            for value in flattened_service_order.tolist()
        ],
        "route_role_state": flattened_route_role.tolist(),
        "local_cost_contribution_state": _quantile_states(
            values=flattened_local_cost_share,
            active_mask=np.ones_like(flattened_local_cost_share, dtype=bool),
            labels=[
                "low_local_cost_impact",
                "medium_local_cost_impact",
                "high_local_cost_impact",
            ],
            off_label="medium_local_cost_impact",
        ),
    }

    core_concept_signatures = [
        "|".join(
            [
                concept_states["service_order_state"][idx],
                concept_states["route_role_state"][idx],
                concept_states["local_cost_contribution_state"][idx],
            ]
        )
        for idx in range(flattened_service_order.shape[0])
    ]

    instance_indices = np.repeat(np.arange(batch_size, dtype=np.int64), num_customers)
    customer_indices = np.tile(np.arange(1, num_customers + 1, dtype=np.int64), batch_size)

    return {
        "core_concept_names": list(NODE_CORE_CONCEPT_NAMES),
        "concept_display_names": dict(NODE_CONCEPT_DISPLAY_NAMES),
        "concept_class_orders": dict(NODE_CONCEPT_CLASS_ORDERS),
        "concept_states": concept_states,
        "concept_raw_values": {
            "service_order_state": flattened_service_order,
            "local_cost_contribution_state": flattened_local_cost_share,
        },
        "raw_value_summary": {
            "service_order_state": _summarize_raw(flattened_service_order),
            "local_cost_contribution_state": _summarize_raw(flattened_local_cost_share),
        },
        "core_concept_signatures": core_concept_signatures,
        "instance_indices": instance_indices,
        "customer_indices": customer_indices,
    }
