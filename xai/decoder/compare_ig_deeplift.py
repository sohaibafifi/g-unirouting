"""Run IG and DeepLIFT with a shared baseline and compare the summaries."""
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder.deeplift_explainer import run as deeplift_run
from evaluation.metrics import evaluate_single_report
from evaluation.table_printers import MethodComparisonTablePrinter
from engine.deeplift_attribution import _load_captum_deeplift
from engine.ig_attribution import IG_BASELINE_MODES
from decoder.integrated_gradients_explainer import run as ig_run
from engine.model_io import ModelLoader


def _verbose_print(args: argparse.Namespace, message: str) -> None:
    if bool(getattr(args, "verbose", False)):
        print(message)


def _parse_topk(raw: str) -> List[int]:
    values = sorted({int(tok.strip()) for tok in str(raw).strip("[]").split(",") if tok.strip()})
    if not values:
        raise ValueError("--topk-nodes must contain at least one integer")
    return values


def _hydrate_row(report_path: Path, topk_values: List[int]) -> Dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    wrapped = {"file": str(report_path), "data": raw}
    metrics = evaluate_single_report(wrapped)
    cfg = raw.get("config", {}) or {}
    summary = raw.get("summary", {}) or {}
    deletion = summary.get("deletion_faithfulness", {}) or {}

    row: Dict[str, Any] = {
        "method": str(cfg.get("attribution_method", "")),
        "baseline": str(
            cfg.get("reference_baseline") or cfg.get("ig_baseline") or cfg.get("deeplift_baseline") or ""
        ),
        "model": str(cfg.get("model_label", "")),
        "path": str(report_path),
        "steps": int(summary.get("num_steps", 0)),
        "focus@1": float(metrics["focus_top1"]),
        "focus@3": float(metrics["focus_top3"]),
        "clarity": float(metrics["clarity"]),
        "contrast": float(metrics["contrast_gap"]),
        "recourse": float(metrics["recourse_rate"]),
        "feasible": float(metrics["chosen_feasible_rate"]),
    }
    for k in topk_values:
        row[f"flip@{k}"] = float(
            (deletion.get(str(k)) or {}).get("mean_action_flip_rate", 0.0)
        )
    return row


def _ig_args(args: argparse.Namespace) -> argparse.Namespace:
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
        ig_baseline=args.baseline,
        device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        output_dir=args.output_dir,
        max_instances_to_store=max(1, int(args.max_instances_to_store)),
        save_step_records=args.save_step_records,
        save_instance_traces=True,
        feasibility_weight=None,
        feasibility_top_m=8,
        feasibility_cost_weight=0.25,
        randomize_weights=False,
    )


def _deeplift_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config_id=args.config_id,
        checkpoint=args.checkpoint,
        problem=args.problem,
        graph_size=args.graph_size,
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        topk_nodes=args.topk_nodes,
        attr_features=args.attr_features,
        deeplift_baseline=args.baseline,
        device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        output_dir=args.output_dir,
        max_instances_to_store=max(1, int(args.max_instances_to_store)),
        save_step_records=args.save_step_records,
        save_instance_traces=True,
        feasibility_weight=None,
        feasibility_top_m=8,
        feasibility_cost_weight=0.25,
        randomize_weights=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IG and DeepLIFT with the same baseline and compare their summaries."
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
    parser.add_argument("--baseline", choices=IG_BASELINE_MODES, default="mean-fill")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be >= 2")

    ModelLoader.preflight_check()
    _load_captum_deeplift()
    topk_values = _parse_topk(args.topk_nodes)

    runs = [
        ("integrated_gradients", ig_run, _ig_args(args)),
        ("deeplift", deeplift_run, _deeplift_args(args)),
    ]

    rows: List[Dict[str, Any]] = []
    report_paths: List[str] = []
    for idx, (method, runner, run_args) in enumerate(runs, start=1):
        _verbose_print(args, f"[{idx}/{len(runs)}] Running {method}")
        report_path = runner(run_args)
        _verbose_print(args, f"[{idx}/{len(runs)}] Report saved: {report_path}")
        report_paths.append(str(report_path))
        row = _hydrate_row(report_path, topk_values)
        rows.append(row)

    MethodComparisonTablePrinter().print(
        rows,
        topk_values,
        title=f"IG vs DeepLIFT Comparison ({args.baseline})",
    )

    summary_payload = {
        "timestamp": int(time.time()),
        "baseline": args.baseline,
        "reports": report_paths,
        "rows": rows,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
    else:
        summary_dir = Path(args.output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"ig_vs_deeplift_{int(time.time())}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    print(f"Saved IG vs DeepLIFT comparison to {summary_path}")


if __name__ == "__main__":
    main()
