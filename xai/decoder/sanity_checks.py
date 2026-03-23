"""XAI sanity checks: compare trained reports against randomized-weight reruns."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.table import Table

from evaluation.metrics import (
    evaluate_single_report,
    model_name as metrics_model_name,
    pairwise_stability,
)
from utils.math_utils import safe_mean
from utils.report_io import ReportLoader
from utils.text_utils import fmt_float


def _verbose_print(args: argparse.Namespace, message: str) -> None:
    if bool(getattr(args, "verbose", False)):
        print(message)


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = __import__("json").load(handle)
    except Exception:
        return None
    return {"file": str(path), "data": data}


def _discover_reports(pattern: str, latest: Optional[int]) -> List[Dict[str, Any]]:
    loader = ReportLoader(pattern, latest=latest, exclude_randomized=True)
    return loader.load()


def _normalize_attr_features(raw: Any) -> str:
    if isinstance(raw, list):
        return ",".join(str(v) for v in raw)
    return str(raw or "auto")


def _normalize_topk(raw: Any) -> str:
    if isinstance(raw, list):
        return "[" + ",".join(str(int(v)) for v in raw) + "]"
    return str(raw or "[1,3,5]")


def _build_randomized_cmd(report: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    cfg = report["data"].get("config", {}) or {}
    checkpoint = str(cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or "")
    if not checkpoint:
        raise ValueError("Report does not contain a usable checkpoint path")

    instances = report["data"].get("instances", []) or []
    max_store = len(instances)

    cmd = [
        args.python_bin,
        "xai/action_explainer.py",
        f"--checkpoint={checkpoint}",
        f"--num-instances={int(cfg.get('num_instances', args.num_instances_fallback))}",
        f"--max-steps={int(cfg.get('max_steps', args.max_steps_fallback))}",
        f"--topk-nodes={_normalize_topk(cfg.get('topk_nodes', [1, 3, 5]))}",
        f"--attr-features={_normalize_attr_features(cfg.get('attr_features', 'auto'))}",
        f"--feasibility-top-m={int(cfg.get('feasibility_top_m', 8))}",
        f"--feasibility-cost-weight={float(cfg.get('feasibility_cost_weight', 0.25))}",
        f"--max-instances-to-store={max_store}",
        f"--output-dir={args.output_dir}",
        "--randomize-weights",
    ]
    if cfg.get("feasibility_weight", None) is not None:
        cmd.append(f"--feasibility-weight={float(cfg.get('feasibility_weight', 0.0))}")
    if cfg.get("device", None):
        cmd.append(f"--device={cfg['device']}")
    if cfg.get("seed", None) is not None:
        cmd.append(f"--seed={int(cfg['seed'])}")
    if cfg.get("data_seed", None) is not None:
        cmd.append(f"--data-seed={int(cfg['data_seed'])}")
    return cmd


def _extract_saved_report_path(stdout: str) -> Optional[Path]:
    matches = re.findall(r"Saved XAI report to (.+)", stdout or "")
    if not matches:
        return None
    return Path(matches[-1].strip()).resolve()


def _run_randomized_report(
    report: Dict[str, Any], args: argparse.Namespace
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = _build_randomized_cmd(report, args)
    _verbose_print(args, "[run] " + " ".join(cmd))
    if args.dry_run:
        print("[dry-run] " + " ".join(cmd))
        return None, None

    proc = subprocess.run(cmd, cwd=args.project_root, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        return None, stderr or stdout or f"exit code {proc.returncode}"

    report_path = _extract_saved_report_path(proc.stdout)
    if report_path is None or not report_path.exists():
        return None, "randomized report path could not be parsed from stdout"
    _verbose_print(args, f"[ok] randomized report: {report_path}")
    wrapper = _load_report(report_path)
    if wrapper is None:
        return None, "randomized report could not be loaded"
    return wrapper, None


def _compare_reports(
    trained: Dict[str, Any], randomized: Dict[str, Any]
) -> Dict[str, Any]:
    trained_metrics = evaluate_single_report(trained)
    randomized_metrics = evaluate_single_report(randomized)
    pair = pairwise_stability(trained, randomized)
    trained_contrast = float(safe_mean([trained_metrics["contrast_gap"]]))
    random_contrast = float(safe_mean([randomized_metrics["contrast_gap"]]))

    return {
        "model": metrics_model_name(trained),
        "trained_clarity": trained_metrics["clarity"],
        "random_clarity": randomized_metrics["clarity"],
        "clarity_drop": float(trained_metrics["clarity"] - randomized_metrics["clarity"]),
        "trained_focus_top1": trained_metrics["focus_top1"],
        "random_focus_top1": randomized_metrics["focus_top1"],
        "focus_top1_drop": float(
            trained_metrics["focus_top1"] - randomized_metrics["focus_top1"]
        ),
        "trained_contrast": trained_contrast,
        "random_contrast": random_contrast,
        "contrast_drop": float(trained_contrast - random_contrast),
        "overlap_at_1": pair["overlap_at_1"],
        "overlap_at_3": pair["overlap_at_3"],
        "constraint_top1_match": pair["constraint_top1_match"],
        "alt_action_match": pair["alt_action_match"],
        "compared_steps": pair["compared_steps"],
    }


def _print_table(rows: List[Dict[str, Any]], layout: str = "auto") -> None:
    console = Console()
    width = getattr(console.size, "width", console.width)
    compact = layout == "compact" or (layout == "auto" and width < 150)
    if not rows:
        console.print("[yellow]No sanity-check rows to display[/yellow]")
        return

    if compact:
        table = Table(
            title="XAI Sanity Checks",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", ratio=3)
        table.add_column("trained vs randomized", overflow="fold", ratio=3)
        table.add_column("change", overflow="fold", ratio=2)
        table.add_column("similarity", overflow="fold", ratio=2)
        for row in rows:
            table.add_row(
                str(row["model"]),
                "\n".join(
                    [
                        f"clr={fmt_float(row['trained_clarity'])} -> {fmt_float(row['random_clarity'])}",
                        f"f1={fmt_float(row['trained_focus_top1'])} -> {fmt_float(row['random_focus_top1'])}",
                        f"ctr={fmt_float(row['trained_contrast'])} -> {fmt_float(row['random_contrast'])}",
                    ]
                ),
                "\n".join(
                    [
                        f"d_clr={fmt_float(row['clarity_drop'])}",
                        f"d_f1={fmt_float(row['focus_top1_drop'])}",
                        f"d_ctr={fmt_float(row['contrast_drop'])}",
                    ]
                ),
                "\n".join(
                    [
                        f"ov1={fmt_float(row['overlap_at_1'])}",
                        f"ov3={fmt_float(row['overlap_at_3'])}",
                        f"c1={fmt_float(row['constraint_top1_match'])}",
                        f"alt={fmt_float(row['alt_action_match'])}",
                    ]
                ),
            )
        console.print(table)
        return

    table = Table(
        title="XAI Sanity Checks",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("model", style="bold", overflow="fold", max_width=42)
    table.add_column("steps", justify="right")
    table.add_column("clr", justify="right")
    table.add_column("rand_clr", justify="right")
    table.add_column("d_clr", justify="right")
    table.add_column("f1", justify="right")
    table.add_column("rand_f1", justify="right")
    table.add_column("d_f1", justify="right")
    table.add_column("ctr", justify="right")
    table.add_column("rand_ctr", justify="right")
    table.add_column("d_ctr", justify="right")
    table.add_column("ov@1", justify="right")
    table.add_column("ov@3", justify="right")
    table.add_column("constraint@1", justify="right")
    table.add_column("alt_match", justify="right")

    for row in rows:
        table.add_row(
            str(row["model"]),
            fmt_float(row["compared_steps"], ndigits=1),
            fmt_float(row["trained_clarity"]),
            fmt_float(row["random_clarity"]),
            fmt_float(row["clarity_drop"]),
            fmt_float(row["trained_focus_top1"]),
            fmt_float(row["random_focus_top1"]),
            fmt_float(row["focus_top1_drop"]),
            fmt_float(row["trained_contrast"]),
            fmt_float(row["random_contrast"]),
            fmt_float(row["contrast_drop"]),
            fmt_float(row["overlap_at_1"]),
            fmt_float(row["overlap_at_3"]),
            fmt_float(row["constraint_top1_match"]),
            fmt_float(row["alt_action_match"]),
        )
    console.print(table)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run XAI sanity checks by comparing trained reports to randomized-weight reruns."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument(
        "--report", action="append", default=None,
        help="Explicit action_explainer JSON path. Can be passed multiple times.",
    )
    parser.add_argument("--pattern", default="logs/xai/action_explainer_*.json")
    parser.add_argument("--latest", type=int, default=10)
    parser.add_argument("--output-dir", default="logs/xai/sanity")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--table-layout", choices=["auto", "compact", "wide"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-instances-fallback", type=int, default=128)
    parser.add_argument("--max-steps-fallback", type=int, default=300)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print discovered reports, rerun commands, and randomized report paths.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = str(Path(args.project_root).resolve())

    if args.report:
        reports: List[Dict[str, Any]] = []
        for raw in args.report:
            wrapper = _load_report(Path(raw))
            if wrapper is None:
                continue
            cfg = wrapper["data"].get("config", {}) or {}
            if bool(cfg.get("randomize_weights", False)):
                continue
            reports.append(wrapper)
    else:
        pattern = str(Path(args.project_root) / args.pattern)
        reports = _discover_reports(pattern, args.latest)
    _verbose_print(args, f"Loaded {len(reports)} trained report(s) for sanity checks.")

    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []
    for idx, report in enumerate(reports, start=1):
        _verbose_print(
            args,
            f"[{idx}/{len(reports)}] Comparing report: {report.get('file', '<unknown>')}",
        )
        randomized, error = _run_randomized_report(report, args)
        if error is not None:
            failures.append((metrics_model_name(report), error))
            continue
        if randomized is None:
            continue
        row = _compare_reports(report, randomized)
        rows.append(row)
        _verbose_print(
            args,
            (
                f"[{idx}/{len(reports)}] Result: "
                f"d_clr={fmt_float(row['clarity_drop'])} "
                f"ov@1={fmt_float(row['overlap_at_1'])}"
            ),
        )

    rows.sort(key=lambda row: (row["clarity_drop"], row["focus_top1_drop"]), reverse=True)
    _print_table(rows, layout=args.table_layout)

    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)

    if failures:
        print("\nFailures:")
        for model, message in failures:
            print(f"- {model}: {message}")


if __name__ == "__main__":
    main()
