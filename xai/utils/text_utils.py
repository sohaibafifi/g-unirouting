"""Pure text/formatting utilities shared across XAI modules."""
from __future__ import annotations

import math
import re

from typing import Any


def fmt_float(value: Any, ndigits: int = 4) -> str:
    """Format a float to ndigits decimal places, or 'nan' for non-finite/None."""
    if value is None:
        return "nan"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{ndigits}f}"


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-_.")
    return text.lower() or "unknown"


def cell_text(value: Any) -> str:
    """Return str(value) with a '-' fallback for empty strings."""
    text = str(value)
    return text if text else "-"
