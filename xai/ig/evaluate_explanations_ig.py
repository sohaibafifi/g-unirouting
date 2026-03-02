from __future__ import annotations

import argparse
import math
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rich import box
from rich.table import Table


XAI_DIR = Path(__file__).resolve().parents[1]
if str(XAI_DIR) not in sys.path:
    sys.path.insert(0, str(XAI_DIR))

import evaluate_explanations as shared_eval  # noqa: E402


def _safe_mean(values: Iterable[float]) -> float:
    return shared_eval._safe_mean(values)


def _extract_deletion_metrics(report: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    data = report["data"]
    summary = data.get("summary", {}) or {}
    deletion = summary.get("deletion_faithfulness", {}) or {}
    out: Dict[int, Dict[str, float]] = {}

    if isinstance(deletion, dict) and deletion:
        for raw_k, payload in deletion.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            out[k] = {
                "logit_drop": float(payload.get("mean_logit_drop", float("nan"))),
                "logprob_drop": float(payload.get("mean_logprob_drop", float("nan"))),
                "flip_rate": float(payload.get("mean_action_flip_rate", float("nan"))),
            }
        if out:
            return out

    steps = data.get("steps", []) or []
    accum: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"logit_drop": [], "logprob_drop": [], "flip_rate": []}
    )
    for step in steps:
        payload = step.get("deletion", {}) or {}
        if not isinstance(payload, dict):
            continue
        for raw_k, metric in payload.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            if not isinstance(metric, dict):
                continue
            for src_key, dst_key in [
                ("mean_logit_drop", "logit_drop"),
                ("mean_logprob_drop", "logprob_drop"),
                ("mean_action_flip_rate", "flip_rate"),
            ]:
                try:
                    value = float(metric.get(src_key, float("nan")))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    accum[k][dst_key].append(value)

    for k, vals in accum.items():
        out[k] = {
            "logit_drop": _safe_mean(vals["logit_drop"]),
            "logprob_drop": _safe_mean(vals["logprob_drop"]),
            "flip_rate": _safe_mean(vals["flip_rate"]),
        }
    return out


def _deletion_report_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        row = shared_eval._evaluate_single_report(report)
        row["deletion_metrics"] = _extract_deletion_metrics(report)
        rows.append(row)
    return rows


def _aggregate_deletion_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[shared_eval._shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        metrics_by_report = [_extract_deletion_metrics(report) for report in group_reports]
        all_k = sorted({k for metrics in metrics_by_report for k in metrics.keys()})
        agg_metrics: Dict[int, Dict[str, float]] = {}
        for k in all_k:
            agg_metrics[k] = {
                "flip_rate": _safe_mean(
                    metrics.get(k, {}).get("flip_rate", float("nan"))
                    for metrics in metrics_by_report
                ),
                "logprob_drop": _safe_mean(
                    metrics.get(k, {}).get("logprob_drop", float("nan"))
                    for metrics in metrics_by_report
                ),
                "logit_drop": _safe_mean(
                    metrics.get(k, {}).get("logit_drop", float("nan"))
                    for metrics in metrics_by_report
                ),
            }

        seeds = sorted(
            {
                seed
                for seed in (shared_eval._report_seed(report) for report in group_reports)
                if seed is not None
            }
        )
        rows.append(
            {
                "model": shared_eval._model_name(group_reports[0]),
                "variant_mix": shared_eval._variant_mix_across_reports(group_reports),
                "runs": len(group_reports),
                "seeds": len(seeds),
                "seed_range": f"{min(seeds)}..{max(seeds)}" if seeds else "-",
                "deletion_metrics": agg_metrics,
            }
        )
    return rows


def _format_deletion_metric_block(
    deletion_metrics: Dict[int, Dict[str, float]], key: str, label: str
) -> str:
    if not deletion_metrics:
        return "-"
    lines = []
    for k in sorted(deletion_metrics.keys()):
        value = deletion_metrics[k].get(key, float("nan"))
        lines.append(f"{label}@{k}={shared_eval._fmt_float(value)}")
    return "\n".join(lines) if lines else "-"


def _print_deletion_table(rows: List[Dict[str, Any]], layout: str = "auto") -> None:
    console, compact = shared_eval._table_console(layout, compact_threshold=150)
    if not rows:
        console.print("[yellow]No IG deletion-faithfulness metrics found[/yellow]")
        return

    aggregated = "runs" in rows[0]

    if compact:
        table = Table(
            title="IG Deletion Faithfulness",
            box=box.SIMPLE_HEAVY,
            header_style="bold yellow",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", ratio=3)
        table.add_column("setup", overflow="fold", ratio=2)
        table.add_column("metrics", overflow="fold", ratio=3)

        for row in rows:
            setup_lines = [f"variants: {shared_eval._cell_text(row['variant_mix'])}"]
            if aggregated:
                setup_lines.append(
                    f"runs={int(row['runs'])} seeds={int(row['seeds'])} range={row['seed_range']}"
                )
            metrics_block = [
                _format_deletion_metric_block(row["deletion_metrics"], "flip_rate", "flip"),
                _format_deletion_metric_block(
                    row["deletion_metrics"], "logprob_drop", "dlogp"
                ),
                _format_deletion_metric_block(
                    row["deletion_metrics"], "logit_drop", "dlogit"
                ),
            ]
            table.add_row(
                shared_eval._cell_text(row["model"]),
                "\n".join(setup_lines),
                "\n".join(block for block in metrics_block if block and block != "-"),
            )
        console.print(table)
        return

    table = Table(
        title="IG Deletion Faithfulness",
        box=box.SIMPLE_HEAVY,
        header_style="bold yellow",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("model", style="bold", overflow="fold", max_width=42)
    table.add_column("variants", overflow="fold", max_width=32)
    if aggregated:
        table.add_column("runs", justify="right")
        table.add_column("seeds", justify="right")
        table.add_column("seed_range", no_wrap=True, max_width=12)
    table.add_column("flip@k", overflow="fold", max_width=20)
    table.add_column("dlogp@k", overflow="fold", max_width=20)
    table.add_column("dlogit@k", overflow="fold", max_width=20)

    for row in rows:
        cells = [str(row["model"]), str(row["variant_mix"])]
        if aggregated:
            cells.extend(
                [
                    str(int(row["runs"])),
                    str(int(row["seeds"])),
                    str(row["seed_range"]),
                ]
            )
        cells.extend(
            [
                _format_deletion_metric_block(row["deletion_metrics"], "flip_rate", "flip"),
                _format_deletion_metric_block(
                    row["deletion_metrics"], "logprob_drop", "dlogp"
                ),
                _format_deletion_metric_block(
                    row["deletion_metrics"], "logit_drop", "dlogit"
                ),
            ]
        )
        table.add_row(*cells)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IG explanation quality and show an IG-specific deletion-faithfulness table."
    )
    parser.add_argument(
        "--pattern",
        default="logs/xai/action_explainer_ig_*.json",
        help="Glob pattern for IG reports.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only inspect latest N reports by mtime.",
    )
    parser.add_argument(
        "--sort-by",
        default="clarity",
        choices=[
            "focus_top1",
            "focus_top3",
            "clarity",
            "contrast_gap",
            "counterfactual_available_rate",
            "counterfactual_switch_rate",
            "counterfactual_make_feasible_rate",
            "counterfactual_mean_relative_delta",
            "trajectory_depot_returns",
            "trajectory_depot_share",
            "trajectory_customer_hop_distance",
            "trajectory_recourse_burst_count",
            "trajectory_late_capacity_share",
            "optional_consistency",
            "recourse_rate",
        ],
        help="Sort key for the main evaluation table.",
    )
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV path for the main evaluation table.")
    parser.add_argument("--stability-csv", default=None, help="Optional CSV path for the stability table.")
    parser.add_argument(
        "--stability-mode",
        choices=["reproducibility", "robustness"],
        default="reproducibility",
        help=(
            "reproducibility compares same-seed repeated runs step-by-step; "
            "robustness compares different seeds via cross-seed dispersion of aggregate metrics."
        ),
    )
    parser.add_argument(
        "--aggregate-by-model",
        action="store_true",
        help=(
            "Collapse the top evaluation table to one row per comparable model setting. "
            "This is enabled automatically in robustness mode."
        ),
    )
    parser.add_argument(
        "--table-layout",
        choices=["auto", "wide", "compact"],
        default="auto",
        help="Table rendering layout. 'auto' switches to a compact multiline table on narrow terminals.",
    )
    args = parser.parse_args()

    reports = shared_eval._load_reports(args.pattern, args.latest)
    aggregate_report_rows = args.aggregate_by_model or args.stability_mode == "robustness"
    if aggregate_report_rows:
        report_rows = shared_eval._aggregate_report_rows(reports)
        deletion_rows = _aggregate_deletion_rows(reports)
    else:
        report_rows = [shared_eval._evaluate_single_report(report) for report in reports]
        deletion_rows = _deletion_report_rows(reports)

    report_rows = sorted(
        report_rows,
        key=lambda row: (
            float("-inf")
            if not math.isfinite(float(row.get(args.sort_by, float("nan"))))
            else float(row[args.sort_by])
        ),
        reverse=not args.ascending,
    )
    deletion_rows = sorted(deletion_rows, key=lambda row: str(row.get("model", "")))

    render_layout = "compact" if args.table_layout == "compact" else "auto"
    if args.table_layout == "wide":
        render_layout = "wide"

    shared_eval._print_report_table(report_rows, layout=render_layout)
    _print_deletion_table(deletion_rows, layout=render_layout)

    if args.stability_mode == "robustness":
        stability_rows = shared_eval._evaluate_robustness(reports)
        shared_eval._print_robustness_table(stability_rows, layout=render_layout)
    else:
        stability_rows = shared_eval._evaluate_reproducibility(reports)
        shared_eval._print_stability_table(stability_rows, layout=render_layout)

    if args.output_csv:
        shared_eval._write_csv(report_rows, args.output_csv)
        print(f"\nWrote evaluation CSV: {args.output_csv}")
    if args.stability_csv:
        shared_eval._write_csv(stability_rows, args.stability_csv)
        print(f"Wrote stability CSV: {args.stability_csv}")


if __name__ == "__main__":
    main()
