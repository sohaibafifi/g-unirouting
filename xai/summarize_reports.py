"""Summarize action_explainer JSON reports in a Rich table."""
from __future__ import annotations

import argparse
import csv

from pathlib import Path
from typing import Any, Dict, List

from evaluation.table_printers import SummaryTablePrinter
from utils.report_io import ReportLoader


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_model_name(row: Dict[str, Any]) -> str:
    model_name = row.get("model_label", "")
    if model_name:
        return str(model_name)
    ckpt = row.get("checkpoint_path", "")
    if ckpt:
        return str(ckpt).split("/runs/")[-1].replace("/checkpoints/last.ckpt", "")
    return Path(str(row.get("file", ""))).name


def to_rows(reports: List[Dict[str, Any]], dedupe_ckpt: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ckpts: set = set()

    for report in reports:
        data = report["data"]
        cfg = data.get("config", {})
        summary = data.get("summary", {})
        ckpt_raw = cfg.get("checkpoint_path", "")
        ckpt = str(Path(ckpt_raw).resolve()) if ckpt_raw else ""
        if not ckpt:
            continue
        if dedupe_ckpt and ckpt in seen_ckpts:
            continue
        if dedupe_ckpt:
            seen_ckpts.add(ckpt)

        row = {
            "file": report["file"],
            "checkpoint_path": ckpt,
            "model_label": cfg.get("model_label", _resolve_model_name(cfg)),
            "encoder_target": cfg.get("encoder_target", ""),
            "importance_mode": cfg.get("node_importance_mode", "decision-only"),
            "feasibility_weight": cfg.get("feasibility_weight", 0.0),
            "num_instances": summary.get("num_instances"),
            "num_steps": summary.get("num_steps"),
            "done_all": summary.get("done_all"),
            "mean_final_reward": summary.get("mean_final_reward"),
            "chosen_feasible_rate": summary.get("chosen_action_feasible_rate"),
            "recourse_event_rate": summary.get("recourse_event_rate"),
            "flip_at_1": _safe_get(
                summary, ["deletion_faithfulness", "1", "mean_action_flip_rate"], 0.0
            ),
            "flip_at_3": _safe_get(
                summary, ["deletion_faithfulness", "3", "mean_action_flip_rate"], 0.0
            ),
            "flip_at_5": _safe_get(
                summary, ["deletion_faithfulness", "5", "mean_action_flip_rate"], 0.0
            ),
            "dlogp_at_1": _safe_get(
                summary, ["deletion_faithfulness", "1", "mean_logprob_drop"], 0.0
            ),
            "dlogp_at_3": _safe_get(
                summary, ["deletion_faithfulness", "3", "mean_logprob_drop"], 0.0
            ),
            "dlogp_at_5": _safe_get(
                summary, ["deletion_faithfulness", "5", "mean_logprob_drop"], 0.0
            ),
        }
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize action_explainer JSON reports"
    )
    parser.add_argument(
        "--pattern", default="logs/xai/action_explainer_*.json"
    )
    parser.add_argument("--latest", type=int, default=None)
    parser.add_argument(
        "--sort-by",
        default="flip_at_5",
        choices=["flip_at_1", "flip_at_3", "flip_at_5", "dlogp_at_5", "mean_final_reward"],
    )
    parser.add_argument("--dedupe-ckpt", action="store_true")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    loader = ReportLoader(args.pattern, latest=args.latest)
    reports = loader.load()
    rows = to_rows(reports, dedupe_ckpt=args.dedupe_ckpt)
    rows.sort(key=lambda r: float(r.get(args.sort_by) or 0.0), reverse=True)

    SummaryTablePrinter().print(rows)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
