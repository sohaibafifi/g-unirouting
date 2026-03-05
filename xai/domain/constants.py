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
    ("space_distance", {"locs", "distance_limit"}),
    ("time_windows_service", {"time_windows", "service_time", "depot_tw_end"}),
    ("capacity_demands", {"demand_linehaul", "demand_backhaul", "vehicle_capacity"}),
    ("route_structure", {"open_route", "mixed_backhaul", "has_backhaul"}),
]

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
    "space_distance": "#4c78a8",
    "time_windows_service": "#f58518",
    "capacity_demands": "#54a24b",
    "route_structure": "#e45756",
    "route_recourse": "#e45756",
    "other": "#b279a2",
}

CONSTRAINT_LABELS_EN: Dict[str, str] = {
    "space_distance": "space / distance",
    "time_windows_service": "time windows / service",
    "capacity_demands": "capacity / demands",
    "route_structure": "route structure",
    "route_recourse": "route structure",
    "other": "other",
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
    "space_distance": "géométrie et distance",
    "time_windows_service": "fenêtres de temps et service",
    "capacity_demands": "capacités et demandes",
    "route_structure": "structure de route",
    "route_recourse": "structure de route",
    "other": "autres signaux",
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
