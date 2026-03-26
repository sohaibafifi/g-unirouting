from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from encoder.encoder_probe import _quantile_states


EDGE_CORE_CONCEPT_NAMES: List[str] = [
    "same_route_state",
    "edge_in_solution_state",
    "local_edge_cost_contribution_state",
]

EDGE_CONCEPT_DISPLAY_NAMES: Dict[str, str] = {
    "same_route_state": "same route",
    "edge_in_solution_state": "solution edge",
    "local_edge_cost_contribution_state": "local edge cost",
}

EDGE_CONCEPT_CLASS_ORDERS: Dict[str, List[str]] = {
    "same_route_state": ["different_route", "same_route"],
    "edge_in_solution_state": ["off", "on"],
    "local_edge_cost_contribution_state": [
        "not_in_solution",
        "low_cost_edge",
        "medium_cost_edge",
        "high_cost_edge",
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


def _sample_without_replacement(
    items: List[Tuple[int, int, int, float]],
    limit: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int, float]]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    selected_idx = rng.choice(len(items), size=limit, replace=False)
    selected_idx = np.sort(selected_idx.astype(np.int64))
    return [items[int(idx)] for idx in selected_idx.tolist()]


def compute_edge_concept_bank(
    node_features: torch.Tensor,
    routes: torch.Tensor,
    metadata: Sequence[Dict[str, Any]],
    max_probe_edges: int,
    seed: int,
) -> Dict[str, Any]:
    node_np = _to_numpy(node_features)
    route_np = _to_numpy(routes).astype(np.int64, copy=False)

    batch_size = node_np.shape[0]
    num_customers = node_np.shape[1] - 1
    rng = np.random.default_rng(seed)

    per_instance_budget = max(1, int(max_probe_edges) // max(batch_size, 1))
    per_relation_quota = max(1, per_instance_budget // 3)

    sampled_pairs: List[Tuple[int, int, int, float, str]] = []

    for instance_idx in range(batch_size):
        locations = node_np[instance_idx, :, :2]
        open_route = bool(metadata[instance_idx]["flags"]["open_route"])
        parsed_routes = _extract_routes(route_np[instance_idx].tolist(), num_customers)
        total_solution_cost = max(_route_total_cost(parsed_routes, locations, open_route), 1e-6)

        route_of: Dict[int, int] = {}
        next_of: Dict[int, int] = {}
        for route_id, route in enumerate(parsed_routes):
            for pos, node in enumerate(route):
                route_of[int(node)] = int(route_id)
                if pos < len(route) - 1:
                    next_of[int(node)] = int(route[pos + 1])

        relation_groups: Dict[str, List[Tuple[int, int, int, float]]] = {
            "solution_edge": [],
            "same_route_non_successor": [],
            "different_route": [],
        }
        for src in range(1, num_customers + 1):
            for dst in range(1, num_customers + 1):
                if src == dst:
                    continue
                same_route = route_of.get(src) == route_of.get(dst)
                is_successor = same_route and next_of.get(src) == dst
                if is_successor:
                    relation_groups["solution_edge"].append(
                        (
                            instance_idx,
                            src,
                            dst,
                            _step_distance(locations, src, dst, open_route) / total_solution_cost,
                        )
                    )
                elif same_route:
                    relation_groups["same_route_non_successor"].append(
                        (instance_idx, src, dst, 0.0)
                    )
                else:
                    relation_groups["different_route"].append((instance_idx, src, dst, 0.0))

        chosen: List[Tuple[int, int, int, float, str]] = []
        leftovers: List[Tuple[int, int, int, float, str]] = []
        for relation_name in ("solution_edge", "same_route_non_successor", "different_route"):
            items = relation_groups[relation_name]
            if len(items) <= per_relation_quota:
                chosen.extend(
                    [(i, s, d, score, relation_name) for i, s, d, score in items]
                )
                continue
            selected = _sample_without_replacement(items, per_relation_quota, rng)
            selected_set = set((i, s, d) for i, s, d, _ in selected)
            chosen.extend(
                [(i, s, d, score, relation_name) for i, s, d, score in selected]
            )
            leftovers.extend(
                [
                    (i, s, d, score, relation_name)
                    for i, s, d, score in items
                    if (i, s, d) not in selected_set
                ]
            )

        remaining_budget = max(0, per_instance_budget - len(chosen))
        if remaining_budget > 0 and leftovers:
            extra = _sample_without_replacement(
                [(i, s, d, score) for i, s, d, score, _ in leftovers],
                remaining_budget,
                rng,
            )
            extra_set = set((i, s, d) for i, s, d, _ in extra)
            chosen.extend(
                [
                    (i, s, d, score, relation_name)
                    for i, s, d, score, relation_name in leftovers
                    if (i, s, d) in extra_set
                ]
            )

        sampled_pairs.extend(chosen)

    if not sampled_pairs:
        raise RuntimeError("No edge pairs were sampled for the edge-level concept bank.")

    instance_indices = np.asarray([item[0] for item in sampled_pairs], dtype=np.int64)
    src_indices = np.asarray([item[1] for item in sampled_pairs], dtype=np.int64)
    dst_indices = np.asarray([item[2] for item in sampled_pairs], dtype=np.int64)
    local_edge_cost_share = np.asarray([item[3] for item in sampled_pairs], dtype=np.float64)
    relation_labels = [item[4] for item in sampled_pairs]

    same_route_labels = [
        "same_route" if label != "different_route" else "different_route"
        for label in relation_labels
    ]
    edge_in_solution_labels = [
        "on" if label == "solution_edge" else "off"
        for label in relation_labels
    ]
    solution_mask = np.asarray(
        [label == "solution_edge" for label in relation_labels],
        dtype=bool,
    )
    edge_cost_state = _quantile_states(
        values=local_edge_cost_share,
        active_mask=solution_mask,
        labels=["low_cost_edge", "medium_cost_edge", "high_cost_edge"],
        off_label="not_in_solution",
    )

    concept_states: Dict[str, List[str]] = {
        "same_route_state": same_route_labels,
        "edge_in_solution_state": edge_in_solution_labels,
        "local_edge_cost_contribution_state": edge_cost_state,
    }

    core_concept_signatures = [
        "|".join(
            [
                concept_states["same_route_state"][idx],
                concept_states["edge_in_solution_state"][idx],
                concept_states["local_edge_cost_contribution_state"][idx],
            ]
        )
        for idx in range(len(sampled_pairs))
    ]

    return {
        "core_concept_names": list(EDGE_CORE_CONCEPT_NAMES),
        "concept_display_names": dict(EDGE_CONCEPT_DISPLAY_NAMES),
        "concept_class_orders": dict(EDGE_CONCEPT_CLASS_ORDERS),
        "concept_states": concept_states,
        "concept_raw_values": {
            "local_edge_cost_contribution_state": local_edge_cost_share,
        },
        "raw_value_summary": {
            "local_edge_cost_contribution_state": _summarize_raw(local_edge_cost_share),
        },
        "core_concept_signatures": core_concept_signatures,
        "instance_indices": instance_indices,
        "src_indices": src_indices,
        "dst_indices": dst_indices,
        "relation_labels": relation_labels,
    }
