"""Domain constants: feature specs, constraint groups, labels, colors."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Feature specifications (node / global input layout)
# ---------------------------------------------------------------------------

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

FEATURE_SPEC_BY_NAME: Dict[str, Dict[str, Any]] = {
    spec["name"]: spec for spec in FEATURE_SPECS
}

# ---------------------------------------------------------------------------
# Constraint grouping rules
# ---------------------------------------------------------------------------

CONSTRAINT_GROUP_RULES: List[Tuple[str, set]] = [
    ("geometry", {"locs"}),
    ("distance_limit", {"distance_limit"}),
    ("time_windows_service", {"time_windows", "service_time", "depot_tw_end"}),
    ("capacity_demands", {"demand_linehaul", "demand_backhaul", "vehicle_capacity"}),
    ("route_openness", {"open_route"}),
    ("flow_structure", {"mixed_backhaul", "has_backhaul"}),
]

CONSTRAINT_FAMILIES: Tuple[str, ...] = tuple(name for name, _ in CONSTRAINT_GROUP_RULES)

CONSTRAINT_STATE_TO_FAMILY: Dict[str, str] = {
    "short_hop": "geometry",
    "medium_hop": "geometry",
    "long_hop": "geometry",
    "has_distance_limit": "distance_limit",
    "no_distance_limit": "distance_limit",
    "distance_slack_high": "distance_limit",
    "distance_slack_medium": "distance_limit",
    "distance_slack_low": "distance_limit",
    "has_time_windows": "time_windows_service",
    "no_time_windows": "time_windows_service",
    "no_tw": "time_windows_service",
    "tw_slack_high": "time_windows_service",
    "tw_slack_medium": "time_windows_service",
    "tw_slack_low": "time_windows_service",
    "load_low": "capacity_demands",
    "load_medium": "capacity_demands",
    "load_high": "capacity_demands",
    "closed_route": "route_openness",
    "open_route": "route_openness",
    "linehaul_only": "flow_structure",
    "backhaul": "flow_structure",
    "mixed_backhaul": "flow_structure",
}

# ---------------------------------------------------------------------------
# Counterfactual perturbation specs
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Visualization: constraint colors and English labels
# ---------------------------------------------------------------------------

CONSTRAINT_COLORS: Dict[str, str] = {
    "geometry": "#4c78a8",
    "distance_limit": "#9c755f",
    "space_distance": "#4c78a8",
    "time_windows_service": "#f58518",
    "capacity_demands": "#54a24b",
    "route_openness": "#e45756",
    "flow_structure": "#72b7b2",
    "route_structure": "#e45756",
    "route_recourse": "#e45756",
    "other": "#b279a2",
}

for _state_name, _family_name in CONSTRAINT_STATE_TO_FAMILY.items():
    CONSTRAINT_COLORS.setdefault(_state_name, CONSTRAINT_COLORS[_family_name])

for _legacy_state, _legacy_family in {
    "geometry_only": "geometry",
    "loose_limit": "distance_limit",
    "service_only": "time_windows_service",
    "loose_tw": "time_windows_service",
    "closed_linehaul": "flow_structure",
    "open_linehaul": "flow_structure",
    "closed_backhaul": "flow_structure",
    "open_backhaul": "flow_structure",
    "closed_mixed_backhaul": "flow_structure",
    "open_mixed_backhaul": "flow_structure",
}.items():
    CONSTRAINT_COLORS.setdefault(_legacy_state, CONSTRAINT_COLORS[_legacy_family])

CONSTRAINT_LABELS_EN: Dict[str, str] = {
    "geometry": "geometry",
    "distance_limit": "distance limit",
    "time_windows_service": "time windows / service",
    "capacity_demands": "capacity / demands",
    "route_openness": "route openness",
    "flow_structure": "flow structure",
    "route_structure": "route structure",
    "route_recourse": "route structure",
    "other": "other",
    "short_hop": "short hop",
    "medium_hop": "medium hop",
    "long_hop": "long hop",
    "has_distance_limit": "distance limit active",
    "no_distance_limit": "no distance limit",
    "distance_slack_high": "high distance slack",
    "distance_slack_medium": "medium distance slack",
    "distance_slack_low": "low distance slack",
    "has_time_windows": "time windows active",
    "no_time_windows": "no time windows",
    "no_tw": "no time windows",
    "tw_slack_high": "high TW slack",
    "tw_slack_medium": "medium TW slack",
    "tw_slack_low": "low TW slack",
    "load_low": "low load",
    "load_medium": "medium load",
    "load_high": "high load",
    "closed_route": "closed route",
    "open_route": "open route",
    "linehaul_only": "linehaul only",
    "backhaul": "backhaul",
    "mixed_backhaul": "mixed backhaul",
    "space_distance": "space / distance",
    "geometry_only": "geometry only",
    "loose_limit": "loose distance limit",
    "service_only": "service only",
    "loose_tw": "loose time windows",
    "closed_linehaul": "closed linehaul",
    "open_linehaul": "open linehaul",
    "closed_backhaul": "closed backhaul",
    "open_backhaul": "open backhaul",
    "closed_mixed_backhaul": "closed mixed backhaul",
    "open_mixed_backhaul": "open mixed backhaul",
}

# ---------------------------------------------------------------------------
# French labels (used by text_explanations.py)
# ---------------------------------------------------------------------------

FEATURE_LABELS_FR: Dict[str, str] = {
    "locs": "localisation",
    "demand_linehaul": "demande linehaul",
    "demand_backhaul": "demande backhaul",
    "time_windows": "fenêtres de temps",
    "service_time": "temps de service",
    "vehicle_capacity": "capacité véhicule",
    "open_route": "route ouverte",
    "mixed_backhaul": "backhaul mixte",
    "distance_limit": "limite distance",
    "depot_tw_end": "horizon dépôt",
    "has_backhaul": "présence backhaul",
    "current_time": "temps courant",
    "current_route_length": "longueur route courante",
    "used_capacity_linehaul": "charge linehaul utilisée",
    "used_capacity_backhaul": "charge backhaul utilisée",
    "speed": "vitesse",
}

CONSTRAINT_LABELS_FR: Dict[str, str] = {
    "geometry": "géométrie",
    "distance_limit": "limite de distance",
    "time_windows_service": "fenêtres de temps et service",
    "capacity_demands": "capacités et demandes",
    "route_openness": "ouverture de route",
    "flow_structure": "structure de flux",
    "route_structure": "structure de route",
    "route_recourse": "structure de route",
    "other": "autres signaux",
    "short_hop": "trajet court",
    "medium_hop": "trajet intermédiaire",
    "long_hop": "trajet long",
    "has_distance_limit": "limite de distance active",
    "no_distance_limit": "pas de limite distance",
    "distance_slack_high": "grande marge de distance",
    "distance_slack_medium": "marge de distance intermédiaire",
    "distance_slack_low": "faible marge de distance",
    "has_time_windows": "fenêtres de temps actives",
    "no_time_windows": "sans fenêtres de temps",
    "no_tw": "sans fenêtres de temps",
    "tw_slack_high": "grande marge TW",
    "tw_slack_medium": "marge TW intermédiaire",
    "tw_slack_low": "faible marge TW",
    "load_low": "charge faible",
    "load_medium": "charge moyenne",
    "load_high": "charge élevée",
    "closed_route": "route fermée",
    "open_route": "route ouverte",
    "linehaul_only": "linehaul seul",
    "backhaul": "backhaul",
    "mixed_backhaul": "backhaul mixte",
    "space_distance": "géométrie et distance",
    "geometry_only": "géométrie sans limite",
    "loose_limit": "limite distance large",
    "service_only": "service sans fenêtres de temps",
    "loose_tw": "fenêtres de temps larges",
    "closed_linehaul": "route fermée linehaul",
    "open_linehaul": "route ouverte linehaul",
    "closed_backhaul": "route fermée backhaul",
    "open_backhaul": "route ouverte backhaul",
    "closed_mixed_backhaul": "route fermée mixed backhaul",
    "open_mixed_backhaul": "route ouverte mixed backhaul",
}

COUNTERFACTUAL_FEATURE_LABELS_FR: Dict[str, str] = {
    "time_window_end": "borne haute TW",
    "service_time": "temps de service",
    "demand_linehaul": "demande linehaul",
    "demand_backhaul": "demande backhaul",
    "vehicle_capacity": "capacité véhicule",
    "distance_limit": "limite distance",
    "depot_tw_end": "horizon dépôt",
}
