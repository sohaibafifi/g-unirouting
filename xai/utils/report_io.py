"""Report loading utilities: ReportLoader and ReportFilter."""
from __future__ import annotations

import glob
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional


class ReportLoader:
    """Load XAI JSON reports from a glob pattern with optional filtering.

    Args:
        pattern: Glob string to find report files.
        latest: If set, only consider the N most recently modified files.
        skip_errors: If True, silently skip files that fail to parse.
        exclude_randomized: If True, drop reports with randomize_weights=True.
        sort_by_mtime: If True, sort files by mtime descending (newest first).
    """

    def __init__(
        self,
        pattern: str,
        latest: Optional[int] = None,
        skip_errors: bool = True,
        exclude_randomized: bool = False,
        sort_by_mtime: bool = True,
    ) -> None:
        self.pattern = pattern
        self.latest = latest
        self.skip_errors = skip_errors
        self.exclude_randomized = exclude_randomized
        self.sort_by_mtime = sort_by_mtime

    def load(self) -> List[Dict[str, Any]]:
        """Return list of ``{"file": path_str, "data": dict}`` wrappers."""
        files: List[str] = glob.glob(self.pattern)
        if self.sort_by_mtime:
            files = sorted(files, key=os.path.getmtime, reverse=True)
        if self.latest is not None:
            files = files[: self.latest]

        reports: List[Dict[str, Any]] = []
        for path in files:
            wrapper = self.load_one(path)
            if wrapper is None:
                if not self.skip_errors:
                    raise RuntimeError(f"Failed to load report: {path}")
                continue
            if self.exclude_randomized:
                cfg = wrapper["data"].get("config", {}) or {}
                if bool(cfg.get("randomize_weights", False)):
                    continue
            reports.append(wrapper)
        return reports

    @staticmethod
    def load_one(path: str | Path) -> Optional[Dict[str, Any]]:
        """Load a single JSON report; returns None on any error."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        return {"file": str(path), "data": data}


class ReportFilter:
    """Static filter helpers for report lists."""

    @staticmethod
    def by_method(
        reports: List[Dict[str, Any]], method: str
    ) -> List[Dict[str, Any]]:
        """Keep only reports whose attribution_method matches *method*."""

        def _method(report: Dict[str, Any]) -> str:
            cfg = report.get("data", {}).get("config", {}) or {}
            m = str(cfg.get("attribution_method", "")).strip().lower()
            if m in {"integrated_gradients", "gradient", "deeplift"}:
                return m
            label = str(cfg.get("model_label", "")).strip().lower()
            if "[ig:" in label:
                return "integrated_gradients"
            if "[deeplift:" in label or "[deep-lift:" in label:
                return "deeplift"
            return "gradient"

        if method == "all":
            return reports
        return [r for r in reports if _method(r) == method]

    @staticmethod
    def by_checkpoint(
        reports: List[Dict[str, Any]], checkpoint: str
    ) -> List[Dict[str, Any]]:
        """Keep only reports whose resolved checkpoint path matches *checkpoint*."""
        target = str(Path(checkpoint).resolve())
        out = []
        for r in reports:
            cfg = r.get("data", {}).get("config", {}) or {}
            ckpt = str(
                cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or ""
            ).strip()
            if ckpt == target:
                out.append(r)
        return out
