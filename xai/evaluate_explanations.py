"""Evaluate explanation quality and stability across action_explainer reports."""
from __future__ import annotations

import argparse
import csv
import math

from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import (
    aggregate_report_rows,
    deletion_rows,
    evaluate_reproducibility,
    evaluate_robustness,
    evaluate_single_report,
    report_method,
)
from evaluation.table_printers import (
    DeletionTablePrinter,
    EvalTablePrinter,
    RobustnessTablePrinter,
    StabilityTablePrinter,
)
from utils.report_io import ReportFilter, ReportLoader


def _filter_by_method(reports: List[Dict[str, Any]], method_filter: str) -> List[Dict[str, Any]]:
    if method_filter == "all":
        return reports
    return [r for r in reports if report_method(r) == method_filter]


def _write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate explanation quality and stability across action_explainer reports."
    )
    parser.add_argument("--pattern", default="logs/xai/action_explainer_*.json")
    parser.add_argument(
        "--method-filter",
        choices=["all", "gradient", "integrated_gradients"],
        default="all",
    )
    parser.add_argument("--latest", type=int, default=None)
    parser.add_argument(
        "--sort-by",
        default="clarity",
        choices=[
            "focus_top1", "focus_top3", "clarity", "contrast_gap",
            "counterfactual_available_rate", "counterfactual_switch_rate",
            "counterfactual_make_feasible_rate", "counterfactual_mean_relative_delta",
            "trajectory_depot_returns", "trajectory_depot_share",
            "trajectory_customer_hop_distance", "trajectory_recourse_burst_count",
            "trajectory_late_capacity_share",
            "trajectory_mean_selected_tw_slack_norm",
            "trajectory_late_tw_tight_share",
            "trajectory_recourse_under_tw_tight_share",
            "optional_consistency", "recourse_rate",
        ],
    )
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--stability-csv", default=None)
    parser.add_argument(
        "--stability-mode",
        choices=["reproducibility", "robustness"],
        default="reproducibility",
    )
    parser.add_argument("--aggregate-by-model", action="store_true")
    parser.add_argument(
        "--table-layout", choices=["auto", "wide", "compact"], default="auto"
    )
    args = parser.parse_args()

    loader = ReportLoader(args.pattern, latest=None)
    reports = _filter_by_method(loader.load(), args.method_filter)
    if args.latest is not None:
        reports = reports[: args.latest]

    do_aggregate = args.aggregate_by_model or args.stability_mode == "robustness"
    if do_aggregate:
        report_rows = aggregate_report_rows(reports)
    else:
        report_rows = [evaluate_single_report(r) for r in reports]

    report_rows = sorted(
        report_rows,
        key=lambda row: (
            float("-inf")
            if not math.isfinite(float(row.get(args.sort_by, float("nan"))))
            else float(row[args.sort_by])
        ),
        reverse=not args.ascending,
    )

    layout = args.table_layout if args.table_layout != "wide" else "wide"
    EvalTablePrinter(layout=layout).print(report_rows)

    ig_reports = [r for r in reports if report_method(r) == "integrated_gradients"]
    if ig_reports:
        DeletionTablePrinter(layout=layout).print(
            deletion_rows(ig_reports, aggregate=do_aggregate)
        )

    if args.stability_mode == "robustness":
        stability_rows = evaluate_robustness(reports)
        RobustnessTablePrinter(layout=layout).print(stability_rows)
    else:
        stability_rows = evaluate_reproducibility(reports)
        StabilityTablePrinter(layout=layout).print(stability_rows)

    if args.output_csv:
        _write_csv(report_rows, args.output_csv)
        print(f"\nWrote evaluation CSV: {args.output_csv}")
    if args.stability_csv:
        _write_csv(stability_rows, args.stability_csv)
        print(f"Wrote stability CSV: {args.stability_csv}")


if __name__ == "__main__":
    main()
