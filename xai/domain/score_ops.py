"""Pure score/attribution operations over domain features and constraints."""
from __future__ import annotations

import math

from collections import defaultdict
from typing import Dict, List, Optional

from domain.constants import CONSTRAINT_GROUP_RULES


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
