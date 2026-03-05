"""Pure numeric utility functions shared across XAI modules."""
from __future__ import annotations

import math

from typing import Iterable

import torch


def safe_mean(values: Iterable[float]) -> float:
    """Return the mean of finite values, or nan if there are none.

    Note: returns nan on empty (statistically correct), unlike the old
    action_explainer._safe_mean which returned 0.0. Call sites in
    action_explainer that build the summary dict handle nan via fmt_float.
    """
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def safe_std(values: Iterable[float]) -> float:
    """Return the population std of finite values, or nan if fewer than 2."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of values where mask is True; returns 0.0 if no True entries."""
    if values.numel() == 0 or not bool(mask.any().item()):
        return 0.0
    return float(values[mask].mean().item())
