"""TypedDicts for XAI report structure."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class ReportConfig(TypedDict, total=False):
    seed: Optional[int]
    data_seed: Optional[int]
    device: str
    num_instances: int
    max_steps: int
    topk_nodes: List[int]
    attr_features: List[str]
    constraint_groups: List[str]
    model_target: str
    encoder_target: str
    decoder_target: str
    run_name: str
    run_group: str
    model_label: str
    model_slug: str
    config_id: Optional[int]
    checkpoint_path: str
    checkpoint_path_resolved: str
    checkpoint_kind: str
    randomize_weights: bool
    model_state: str
    node_importance_mode: str
    feasibility_weight: float
    feasibility_top_m: int
    feasibility_cost_weight: float
    problem: str
    graph_size: int
    config_repr: str
    attribution_method: str
    ig_steps: int
    ig_baseline: str


class InstanceTrace(TypedDict, total=False):
    instance_index: int
    instance_variant_code: str
    instance_variant_flags: Dict[str, bool]
    instance_active_constraints: List[str]
    instance_structural_constraints: List[Dict[str, Any]]
    locs: List[List[float]]
    demand_linehaul: List[float]
    demand_backhaul: List[float]
    vehicle_capacity: float
    distance_limit: float
    depot_time_limit: float
    actions: List[int]
    done_before: List[bool]
    done_after: List[bool]
    top_nodes: List[List[int]]
    top_scores: List[List[float]]
    top_features: List[List[Dict[str, Any]]]
    top_constraints: List[List[Dict[str, Any]]]
    top_constraint_states: List[List[Dict[str, Any]]]
    decoder_state_constraints: List[List[Dict[str, Any]]]
    decoder_state_constraint_states: List[List[Dict[str, Any]]]
    decoder_dynamic_states: List[List[Dict[str, Any]]]
    counterfactuals: List[Optional[Dict[str, Any]]]
    solution_features: Dict[str, List[float]]


class ReportData(TypedDict, total=False):
    timestamp: int
    config: ReportConfig
    summary: Dict[str, Any]
    steps: List[Dict[str, Any]]
    instances: List[InstanceTrace]
