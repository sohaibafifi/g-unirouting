import argparse
import csv
import glob
import json
import math
import os

from pathlib import Path
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.table import Table


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_reports(pattern: str, latest: int | None = None) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if latest is not None:
        files = files[:latest]

    reports = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        reports.append({"file": file_path, "data": data})
    return reports


def to_rows(reports: List[Dict[str, Any]], dedupe_ckpt: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ckpts = set()

    for report in reports:
        data = report["data"]
        cfg = data.get("config", {})
        summary = data.get("summary", {})
        ckpt_raw = cfg.get("checkpoint_path", "")
        ckpt = str(Path(ckpt_raw).resolve()) if ckpt_raw else ""
        if not ckpt:
            continue

        if dedupe_ckpt and ckpt and ckpt in seen_ckpts:
            continue
        if dedupe_ckpt and ckpt:
            seen_ckpts.add(ckpt)

        row = {
            "file": report["file"],
            "checkpoint_path": ckpt,
            "model_label": cfg.get("model_label", ""),
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


def _fmt_float(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "nan"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{ndigits}f}"


def _resolve_model_name(row: Dict[str, Any]) -> str:
    model_name = row.get("model_label", "")
    if model_name:
        return str(model_name)
    ckpt = row.get("checkpoint_path", "")
    if ckpt:
        return str(ckpt).split("/runs/")[-1].replace("/checkpoints/last.ckpt", "")
    return Path(str(row.get("file", ""))).name


def print_table(rows: List[Dict[str, Any]]) -> None:
    console = Console(width=180)
    if not rows:
        console.print("[yellow]No reports matched[/yellow]")
        return

    table = Table(
        title="XAI Report Summary",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("model", style="bold", no_wrap=True, overflow="ellipsis")
    table.add_column("mode", style="magenta", justify="center", no_wrap=True)
    table.add_column("w_feas", justify="right")
    table.add_column("flip@1", justify="right")
    table.add_column("flip@3", justify="right")
    table.add_column("flip@5", justify="right")
    table.add_column("dlogp@1", justify="right")
    table.add_column("dlogp@3", justify="right")
    table.add_column("dlogp@5", justify="right")
    table.add_column("reward", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("done", justify="center")
    table.add_column("feasible", justify="right")
    table.add_column("recourse", justify="right")

    for row in rows:
        done = bool(row.get("done_all", False))
        done_str = "[green]True[/green]" if done else "[red]False[/red]"
        table.add_row(
            _resolve_model_name(row),
            str(row.get("importance_mode", "decision-only")),
            _fmt_float(row.get("feasibility_weight"), ndigits=2),
            _fmt_float(row.get("flip_at_1")),
            _fmt_float(row.get("flip_at_3")),
            _fmt_float(row.get("flip_at_5")),
            _fmt_float(row.get("dlogp_at_1")),
            _fmt_float(row.get("dlogp_at_3")),
            _fmt_float(row.get("dlogp_at_5")),
            _fmt_float(row.get("mean_final_reward")),
            str(row.get("num_steps", "nan")),
            done_str,
            _fmt_float(row.get("chosen_feasible_rate")),
            _fmt_float(row.get("recourse_event_rate")),
        )
    console.print(table)


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
        "--pattern",
        default="logs/xai/action_explainer_*.json",
        help="Glob pattern to load reports",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only consider latest N files by mtime",
    )
    parser.add_argument(
        "--sort-by",
        default="flip_at_5",
        choices=[
            "flip_at_1",
            "flip_at_3",
            "flip_at_5",
            "dlogp_at_5",
            "mean_final_reward",
        ],
        help="Sort key",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending",
    )
    parser.add_argument(
        "--dedupe-ckpt",
        action="store_true",
        help="Keep only the newest report per checkpoint path",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    reports = load_reports(args.pattern, args.latest)
    rows = to_rows(reports, dedupe_ckpt=args.dedupe_ckpt)
    rows = sorted(rows, key=lambda r: r[args.sort_by], reverse=not args.ascending)

    print_table(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
