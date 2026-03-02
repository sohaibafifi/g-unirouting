from __future__ import annotations

import argparse
import json
import math
import time

from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

import action_explainer_ig as ig_main


def _parse_baselines(raw: str) -> List[str]:
    values = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not values:
        raise ValueError("--ig-baselines must contain at least one baseline")
    bad = [value for value in values if value not in ig_main.IG_BASELINE_MODES]
    if bad:
        raise ValueError(
            f"Unsupported baselines: {', '.join(bad)}. "
            f"Expected values from: {', '.join(ig_main.IG_BASELINE_MODES)}"
        )
    return values


def _parse_topk(raw: str) -> List[int]:
    values = sorted({int(tok.strip()) for tok in str(raw).strip("[]").split(",") if tok.strip()})
    if not values:
        raise ValueError("--topk-nodes must contain at least one integer")
    return values


def _run_args(args: argparse.Namespace, baseline: str) -> argparse.Namespace:
    store_count = max(1, int(args.max_instances_to_store))
    return argparse.Namespace(
        config_id=args.config_id,
        checkpoint=args.checkpoint,
        problem=args.problem,
        graph_size=args.graph_size,
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        topk_nodes=args.topk_nodes,
        attr_features=args.attr_features,
        ig_steps=args.ig_steps,
        ig_baseline=baseline,
        device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        output_dir=args.output_dir,
        max_instances_to_store=store_count,
        save_step_records=args.save_step_records,
        save_instance_traces=args.save_instance_traces,
    )


def _load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_from_report(report_path: Path, topk_values: List[int]) -> Dict[str, Any]:
    data = _load_report(report_path)
    cfg = data.get("config", {}) or {}
    summary = data.get("summary", {}) or {}
    deletion = summary.get("deletion_faithfulness", {}) or {}

    row: Dict[str, Any] = {
        "baseline": str(cfg.get("ig_baseline", "")),
        "model": str(cfg.get("model_label", "")),
        "path": str(report_path),
        "steps": int(summary.get("num_steps", 0)),
        "focus@1": 0.0,
        "focus@3": 0.0,
        "clarity": 0.0,
        "contrast": 0.0,
        "recourse": 0.0,
        "feasible": 0.0,
    }
    for k in topk_values:
        row[f"flip@{k}"] = float((deletion.get(str(k)) or {}).get("mean_action_flip_rate", 0.0))
    return row


def _hydrate_metrics_from_shared(report_path: Path, topk_values: List[int]) -> Dict[str, Any]:
    import evaluate_explanations as shared_eval

    raw = _load_report(report_path)
    wrapped = {"path": report_path, "file": str(report_path), "data": raw}
    metrics = shared_eval._evaluate_single_report(wrapped)
    row = _row_from_report(report_path, topk_values)
    row["focus@1"] = float(metrics["focus_top1"])
    row["focus@3"] = float(metrics["focus_top3"])
    row["clarity"] = float(metrics["clarity"])
    row["contrast"] = float(metrics["contrast_gap"])
    row["recourse"] = float(metrics["recourse_rate"])
    row["feasible"] = float(metrics["chosen_feasible_rate"])
    return row


def _print_table(rows: List[Dict[str, Any]], topk_values: List[int]) -> None:
    def _fmt(value: float) -> str:
        return "n/a" if not math.isfinite(float(value)) else f"{float(value):.4f}"

    console = Console()
    table = Table(title="IG Baseline Comparison")
    table.add_column("baseline")
    table.add_column("metrics")
    for row in rows:
        flips = " ".join(f"flip@{k}={_fmt(row[f'flip@{k}'])}" for k in topk_values)
        metrics = (
            f"f1={_fmt(row['focus@1'])} f3={_fmt(row['focus@3'])}\n"
            f"clr={_fmt(row['clarity'])} ctr={_fmt(row['contrast'])}\n"
            f"{flips}\n"
            f"rec={_fmt(row['recourse'])} feas={_fmt(row['feasible'])}"
        )
        table.add_row(row["baseline"], metrics)
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the IG explainer across several baselines and compare the summaries."
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument("--attr-features", default="auto")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument(
        "--ig-baselines",
        default="mean-fill,zero-with-current-locs,zero-with-customers-at-depot",
        help="Comma-separated list of IG baselines to run.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-instance-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--summary-output", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be >= 2")

    baselines = _parse_baselines(args.ig_baselines)
    topk_values = _parse_topk(args.topk_nodes)
    ig_main.grad_base._preflight_check()

    rows: List[Dict[str, Any]] = []
    report_paths: List[str] = []
    for baseline in baselines:
        report_path = ig_main.run(_run_args(args, baseline))
        report_paths.append(str(report_path))
        rows.append(_hydrate_metrics_from_shared(report_path, topk_values))

    _print_table(rows, topk_values)

    summary_payload = {
        "timestamp": int(time.time()),
        "baselines": baselines,
        "reports": report_paths,
        "rows": rows,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
    else:
        summary_dir = Path("logs/xai/ig")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"baseline_compare_{int(time.time())}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    print(f"Saved baseline comparison to {summary_path}")


if __name__ == "__main__":
    main()
