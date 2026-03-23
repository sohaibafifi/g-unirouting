"""Pure score/attribution operations over domain features and constraints."""
from __future__ import annotations

import math

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

from domain.constants import CONSTRAINT_GROUP_RULES, CONSTRAINT_STATE_TO_FAMILY


def feature_to_constraint_group(feature_name: str) -> str:
    """Map a feature name to its constraint group name."""
    for group_name, members in CONSTRAINT_GROUP_RULES:
        if feature_name in members:
            return group_name
    return "other"


def top_feature_payload(
    feature_scores: Dict[str, float], top_n: int = 3
) -> List[Dict[str, float]]:
    """Return the top-N features by score as a payload list."""
    if not feature_scores:
        return []
    sanitized: Dict[str, float] = {}
    for key, value in feature_scores.items():
        sanitized[key] = max(float(value), 0.0) if math.isfinite(float(value)) else 0.0
    total = sum(sanitized.values())
    items = sorted(sanitized.items(), key=lambda x: x[1], reverse=True)[:top_n]
    payload = []
    for name, score in items:
        share = (score / total) if total > 0 else 0.0
        payload.append({"feature": name, "score": float(score), "share": float(share)})
    return payload


def aggregate_constraint_scores(
    feature_scores: Dict[str, float],
) -> Dict[str, float]:
    """Aggregate per-feature scores into per-constraint-group scores."""
    grouped: Dict[str, float] = defaultdict(float)
    for feature_name, score in feature_scores.items():
        if not math.isfinite(float(score)):
            continue
        grouped[feature_to_constraint_group(feature_name)] += max(float(score), 0.0)
    return dict(grouped)


def top_constraint_payload(
    constraint_scores: Dict[str, float], top_n: Optional[int] = 3
) -> List[Dict[str, float]]:
    """Return the top-N constraint groups by score as a payload list."""
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


def aggregate_decoder_state_constraint_scores(
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Aggregate dynamic decoder-state features into constraint-family scores."""

    def _finite(value: Any) -> float:
        try:
            cast = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return cast if math.isfinite(cast) else float("nan")

    def _clamp01(value: Any) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return max(0.0, min(1.0, cast))

    def _inverse_ratio(value: Any) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return max(0.0, min(1.0, 1.0 - cast))

    def _scaled_tanh(value: Any, scale: float) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return float(math.tanh(max(cast, 0.0) / max(scale, 1e-8)))

    variant_flags = variant_flags or {}
    is_customer = _clamp01(solution_features.get("selected_is_customer", 1.0)) >= 0.5

    linehaul_util = _clamp01(solution_features.get("used_capacity_linehaul_share", 0.0))
    backhaul_util = _clamp01(solution_features.get("used_capacity_backhaul_share", 0.0))
    total_util = _clamp01(linehaul_util + backhaul_util)

    distance_slack_pressure = _inverse_ratio(
        solution_features.get("distance_budget_slack_norm", float("nan"))
    )
    travel_pressure = (
        _scaled_tanh(solution_features.get("selected_travel_distance", float("nan")), 0.35)
        if is_customer
        else 0.0
    )

    tw_tightness = (
        _inverse_ratio(solution_features.get("selected_tw_slack_norm", float("nan")))
        if is_customer
        else 0.0
    )
    depot_time_pressure = _inverse_ratio(
        solution_features.get("depot_time_budget_slack_norm", float("nan"))
    )
    wait_pressure = (
        _scaled_tanh(solution_features.get("selected_wait_time", float("nan")), 0.15)
        if is_customer
        else 0.0
    )
    service_pressure = (
        _scaled_tanh(solution_features.get("selected_service_time", float("nan")), 0.18)
        if is_customer
        else 0.0
    )

    has_distance_limit = bool((variant_flags or {}).get("distance_limit", False))
    has_time_windows = bool((variant_flags or {}).get("time_windows", False))

    scores = {
        "geometry": travel_pressure,
        "capacity_demands": max(
            linehaul_util,
            backhaul_util,
            total_util,
        ),
    }
    if has_distance_limit:
        scores["distance_limit"] = distance_slack_pressure
    if has_time_windows:
        scores["time_windows_service"] = max(
            tw_tightness,
            depot_time_pressure,
            0.55 * wait_pressure + 0.45 * service_pressure,
        )

    return {
        key: float(value)
        for key, value in scores.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }


def instance_structural_constraint_labels(
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Return fixed structural constraints for the instance."""
    variant_flags = variant_flags or {}
    open_route = bool(variant_flags.get("open_route", False))
    backhaul = bool(variant_flags.get("backhaul", False))
    mixed_backhaul = bool(variant_flags.get("mixed_backhaul", False))
    has_distance_limit = bool(variant_flags.get("distance_limit", False))
    has_time_windows = bool(variant_flags.get("time_windows", False))

    flow_structure_state = "linehaul_only"
    if mixed_backhaul and backhaul:
        flow_structure_state = "mixed_backhaul"
    elif backhaul:
        flow_structure_state = "backhaul"

    return {
        "route_openness": "open_route" if open_route else "closed_route",
        "flow_structure": flow_structure_state,
        "distance_limit": (
            "has_distance_limit" if has_distance_limit else "no_distance_limit"
        ),
        "time_windows_service": (
            "has_time_windows" if has_time_windows else "no_time_windows"
        ),
    }


def instance_structural_constraint_payload(
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Serialize fixed structural constraints for the instance."""
    labels = instance_structural_constraint_labels(variant_flags=variant_flags)
    ordered_families = [
        "route_openness",
        "flow_structure",
        "distance_limit",
        "time_windows_service",
    ]
    payload: List[Dict[str, str]] = []
    for family_name in ordered_families:
        state_name = labels.get(family_name)
        if not state_name:
            continue
        payload.append(
            {
                "constraint": family_name,
                "constraint_family": family_name,
                "constraint_state": state_name,
            }
        )
    return payload


def decoder_dynamic_state_labels(
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Return dynamic decoder states derived from the current decision context."""

    def _finite(value: Any) -> float:
        try:
            cast = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return cast if math.isfinite(cast) else float("nan")

    def _clamp01(value: Any) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return max(0.0, min(1.0, cast))

    def _inverse_ratio(value: Any) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return max(0.0, min(1.0, 1.0 - cast))

    def _scaled_tanh(value: Any, scale: float) -> float:
        cast = _finite(value)
        if not math.isfinite(cast):
            return 0.0
        return float(math.tanh(max(cast, 0.0) / max(scale, 1e-8)))

    def _bucket_pressure(value: float, low: float = 0.33, high: float = 0.66) -> str:
        if value >= high:
            return "high"
        if value >= low:
            return "medium"
        return "low"

    variant_flags = variant_flags or {}
    has_distance_limit = bool(variant_flags.get("distance_limit", False))
    has_time_windows = bool(variant_flags.get("time_windows", False))

    linehaul_util = _clamp01(solution_features.get("used_capacity_linehaul_share", 0.0))
    backhaul_util = _clamp01(solution_features.get("used_capacity_backhaul_share", 0.0))
    total_util = _clamp01(max(linehaul_util, backhaul_util, linehaul_util + backhaul_util))
    load_bucket = _bucket_pressure(total_util)
    capacity_state = {
        "low": "load_low",
        "medium": "load_medium",
        "high": "load_high",
    }[load_bucket]

    travel_pressure = _scaled_tanh(
        solution_features.get("selected_travel_distance", float("nan")),
        0.35,
    )
    distance_slack_pressure = _inverse_ratio(
        solution_features.get("distance_budget_slack_norm", float("nan"))
    )
    geometry_bucket = _bucket_pressure(travel_pressure)
    geometry_state = {
        "low": "short_hop",
        "medium": "medium_hop",
        "high": "long_hop",
    }[geometry_bucket]

    distance_pressure = distance_slack_pressure
    distance_bucket = _bucket_pressure(distance_pressure)
    if has_distance_limit:
        distance_state = {
            "low": "distance_slack_high",
            "medium": "distance_slack_medium",
            "high": "distance_slack_low",
        }[distance_bucket]

    tw_tightness = _inverse_ratio(
        solution_features.get("selected_tw_slack_norm", float("nan"))
    )
    depot_time_pressure = _inverse_ratio(
        solution_features.get("depot_time_budget_slack_norm", float("nan"))
    )
    time_pressure = max(tw_tightness, depot_time_pressure)
    time_bucket = _bucket_pressure(time_pressure)
    if has_time_windows:
        time_state = {
            "low": "tw_slack_high",
            "medium": "tw_slack_medium",
            "high": "tw_slack_low",
        }[time_bucket]

    labels = {
        "capacity_demands": capacity_state,
        "geometry": geometry_state,
    }
    if has_distance_limit:
        labels["distance_limit"] = distance_state
    if has_time_windows:
        labels["time_windows_service"] = time_state
    return labels


def aggregate_decoder_dynamic_state_scores(
    constraint_scores: Mapping[str, Any],
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Project family-level scores onto the current dynamic decoder states."""

    state_labels = decoder_dynamic_state_labels(
        solution_features=solution_features,
        variant_flags=variant_flags,
    )
    grouped: Dict[str, float] = defaultdict(float)
    for family_name, raw_score in constraint_scores.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score <= 0.0:
            continue
        state_name = state_labels.get(str(family_name))
        if not state_name:
            continue
        grouped[state_name] += score
    return dict(grouped)


def top_decoder_dynamic_state_payload(
    constraint_scores: Mapping[str, Any],
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
    top_n: Optional[int] = 3,
) -> List[Dict[str, float]]:
    """Return dynamic decoder states with family metadata."""

    state_scores = aggregate_decoder_dynamic_state_scores(
        constraint_scores=constraint_scores,
        solution_features=solution_features,
        variant_flags=variant_flags,
    )
    if not state_scores:
        return []

    state_labels = decoder_dynamic_state_labels(
        solution_features=solution_features,
        variant_flags=variant_flags,
    )
    total = sum(max(float(v), 0.0) for v in state_scores.values())
    items = sorted(state_scores.items(), key=lambda item: item[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]

    family_by_state = {state: family for family, state in state_labels.items()}
    payload: List[Dict[str, float]] = []
    for state_name, raw_score in items:
        score = max(float(raw_score), 0.0)
        payload.append(
            {
                "constraint": state_name,
                "constraint_state": state_name,
                "constraint_family": family_by_state.get(state_name, "other"),
                "score": score,
                "share": (score / total) if total > 0 else 0.0,
            }
        )
    return payload


# Backward-compatible aliases used by older code paths.
def decoder_constraint_state_labels(
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    return decoder_dynamic_state_labels(
        solution_features=solution_features,
        variant_flags=variant_flags,
    )


def aggregate_constraint_state_scores(
    constraint_scores: Mapping[str, Any],
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    return aggregate_decoder_dynamic_state_scores(
        constraint_scores=constraint_scores,
        solution_features=solution_features,
        variant_flags=variant_flags,
    )


def top_constraint_state_payload(
    constraint_scores: Mapping[str, Any],
    solution_features: Mapping[str, Any],
    variant_flags: Optional[Mapping[str, Any]] = None,
    top_n: Optional[int] = 3,
) -> List[Dict[str, float]]:
    return top_decoder_dynamic_state_payload(
        constraint_scores=constraint_scores,
        solution_features=solution_features,
        variant_flags=variant_flags,
        top_n=top_n,
    )


def top_state_payload(
    state_scores: Mapping[str, Any],
    top_n: Optional[int] = 3,
) -> List[Dict[str, float]]:
    """Serialize already-computed state scores with family metadata."""

    if not state_scores:
        return []
    total = sum(
        max(float(value), 0.0)
        for value in state_scores.values()
        if math.isfinite(float(value))
    )
    items = sorted(
        (
            (str(name), max(float(value), 0.0))
            for name, value in state_scores.items()
            if math.isfinite(float(value)) and float(value) > 0.0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_n is not None:
        items = items[:top_n]

    payload: List[Dict[str, float]] = []
    for state_name, score in items:
        payload.append(
            {
                "constraint": state_name,
                "constraint_state": state_name,
                "constraint_family": CONSTRAINT_STATE_TO_FAMILY.get(state_name, "other"),
                "score": score,
                "share": (score / total) if total > 0 else 0.0,
            }
        )
    return payload


def entropy_concentration(scores: List[float]) -> float:
    """Normalized entropy concentration: 1.0 = fully concentrated, 0.0 = uniform."""
    positive = [max(float(s), 0.0) for s in scores if float(s) > 0]
    if not positive:
        return float("nan")
    total = sum(positive)
    if total <= 0:
        return float("nan")
    probs = [s / total for s in positive]
    if len(probs) == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 1.0
    return 1.0 - (entropy / max_entropy)
