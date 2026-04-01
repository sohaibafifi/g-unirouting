from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.table import Table

from encoder.intervention_validation import (
    INTERVENTIONS,
    compute_intervention_validation_bundle,
    write_intervention_validation_report,
)
from encoder.encoder_probe import _json_ready
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare intervention-based validation of discovered directions across configs."
    )
    parser.add_argument("--config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument(
        "--pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument(
        "--sort-by",
        choices=[
            "pca_mean_directional_success",
            "ica_mean_directional_success",
            "pca_mean_aligned_delta_correlation",
            "ica_mean_aligned_delta_correlation",
            "concept_success_mean",
        ],
        default="ica_mean_directional_success",
    )
    parser.add_argument(
        "--output-json",
        default="logs/xai/encoder/intervention_validation/comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/xai/encoder/intervention_validation/comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default="logs/xai/encoder/intervention_validation/runs",
    )
    return parser


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _fmt(value: Any, ndigits: int = 4) -> str:
    cast = _safe_float(value)
    if cast is None:
        return "-"
    return f"{cast:.{ndigits}f}"


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_")


def _nested_get(payload: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summary_row(config_id: int, report: Dict[str, Any]) -> Dict[str, Any]:
    concept_successes = []
    pca_successes = []
    ica_successes = []
    pca_corrs = []
    ica_corrs = []
    for spec in INTERVENTIONS:
        name = str(spec["name"])
        concept_success = _nested_get(report, ["interventions", name, "concept_success_rate"])
        pca_success = _nested_get(report, ["interventions", name, "methods", "pca", "directional_success_rate"])
        ica_success = _nested_get(report, ["interventions", name, "methods", "ica", "directional_success_rate"])
        pca_corr = _nested_get(report, ["interventions", name, "methods", "pca", "aligned_delta_correlation"])
        ica_corr = _nested_get(report, ["interventions", name, "methods", "ica", "aligned_delta_correlation"])
        if _safe_float(concept_success) is not None:
            concept_successes.append(float(concept_success))
        if _safe_float(pca_success) is not None:
            pca_successes.append(float(pca_success))
        if _safe_float(ica_success) is not None:
            ica_successes.append(float(ica_success))
        if _safe_float(pca_corr) is not None:
            pca_corrs.append(float(pca_corr))
        if _safe_float(ica_corr) is not None:
            ica_corrs.append(float(ica_corr))

    return {
        "config_id": config_id,
        "model": report["config"]["config_repr"],
        "encoder": report["config"]["encoder_name"],
        "decoder": report["config"]["decoder_name"],
        "concept_success_mean": _mean(concept_successes),
        "pca_mean_directional_success": _mean(pca_successes),
        "ica_mean_directional_success": _mean(ica_successes),
        "pca_mean_aligned_delta_correlation": _mean(pca_corrs),
        "ica_mean_aligned_delta_correlation": _mean(ica_corrs),
    }


def _sort_key(row: Dict[str, Any], sort_by: str) -> float:
    value = _safe_float(row.get(sort_by))
    return float("-inf") if value is None else value


def _best_row(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if _safe_float(row.get(key)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _safe_float(row.get(key)) or float("-inf"))


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Intervention Validation Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("id", justify="right")
    table.add_column("model", style="bold", overflow="fold", max_width=48)
    table.add_column("concept", overflow="fold")
    table.add_column("pca", overflow="fold")
    table.add_column("ica", overflow="fold")

    for row in rows:
        table.add_row(
            str(row["config_id"]),
            str(row["model"]),
            f"success={_fmt(row['concept_success_mean'])}",
            "\n".join(
                [
                    f"dir={_fmt(row['pca_mean_directional_success'])}",
                    f"corr={_fmt(row['pca_mean_aligned_delta_correlation'])}",
                ]
            ),
            "\n".join(
                [
                    f"dir={_fmt(row['ica_mean_directional_success'])}",
                    f"corr={_fmt(row['ica_mean_aligned_delta_correlation'])}",
                ]
            ),
        )
    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace) -> str:
    lines = [
        "# Intervention Validation Comparison",
        "",
        "## Setup",
        "",
        f"- `num_samples`: `{args.num_samples}`",
        f"- `pooling`: `{args.pooling}`",
        f"- `seed`: `{args.seed}`",
        f"- `num_components`: `{args.num_components}`",
        f"- `top_components`: `{args.top_components}`",
        f"- `sort_by`: `{sort_by}`",
        "",
        "## Metrics Used",
        "",
        "- `concept_success_mean`: l'intervention a-t-elle bien fait bouger le concept cible dans la bonne direction ?",
        "- `pca_mean_directional_success` / `ica_mean_directional_success`: le score de la composante choisie bouge-t-il dans le sens attendu ?",
        "- `pca_mean_aligned_delta_correlation` / `ica_mean_aligned_delta_correlation`: la variation du concept et la variation de la composante sont-elles corrélées ?",
        "",
        "## Summary Table",
        "",
        "| id | model | concept_success | pca_dir | ica_dir | pca_corr | ica_corr |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config_id"]),
                    str(row["model"]),
                    _fmt(row["concept_success_mean"]),
                    _fmt(row["pca_mean_directional_success"]),
                    _fmt(row["ica_mean_directional_success"]),
                    _fmt(row["pca_mean_aligned_delta_correlation"]),
                    _fmt(row["ica_mean_aligned_delta_correlation"]),
                ]
            )
            + " |"
        )

    best_pca = _best_row(rows, "pca_mean_directional_success")
    best_ica = _best_row(rows, "ica_mean_directional_success")
    lines.extend(
        [
            "",
            "## Automatic Interpretation",
            "",
            (
                "- Best intervention-tracking config under PCA: "
                f"`{best_pca['model']}` with `pca_dir={_fmt(best_pca['pca_mean_directional_success'])}`."
                if best_pca is not None
                else "- Best intervention-tracking config under PCA: not available."
            ),
            (
                "- Best intervention-tracking config under ICA: "
                f"`{best_ica['model']}` with `ica_dir={_fmt(best_ica['ica_mean_directional_success'])}`."
                if best_ica is not None
                else "- Best intervention-tracking config under ICA: not available."
            ),
            "",
            "## Decision Notes",
            "",
            "- A high `concept_success_mean` validates the intervention design itself.",
            "- A high directional success but low delta correlation means the axis moves in the right direction, but not proportionally.",
            "- This phase is the closest we currently have to causal validation of discovered directions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    configs = Config.all()
    selected_ids = args.config_ids if args.config_ids else list(range(len(configs)))

    per_config_dir = Path(args.per_config_dir)
    per_config_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    all_reports: List[Dict[str, Any]] = []
    console = Console()
    for config_id in selected_ids:
        run_args = SimpleNamespace(
            config_id=config_id,
            checkpoint=None,
            device=args.device,
            graph_size=args.graph_size,
            problem=args.problem,
            num_samples=args.num_samples,
            pooling=args.pooling,
            seed=args.seed,
            num_components=args.num_components,
            top_components=args.top_components,
        )
        report = compute_intervention_validation_bundle(run_args)
        filename = _sanitize_filename(f"{config_id}_{report['config']['config_repr']}.json")
        write_intervention_validation_report(report, str(per_config_dir / filename))
        all_reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        console.print(
            f"[cyan]done[/cyan] config={config_id} model={report['config']['config_repr']} "
            f"pca_dir={_fmt(row['pca_mean_directional_success'])} "
            f"ica_dir={_fmt(row['ica_mean_directional_success'])}"
        )

    rows = sorted(rows, key=lambda row: _sort_key(row, args.sort_by), reverse=True)
    _render_console_table(rows, args.sort_by)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "sort_by": args.sort_by,
            "num_samples": args.num_samples,
            "pooling": args.pooling,
            "seed": args.seed,
            "num_components": args.num_components,
            "top_components": args.top_components,
            "config_ids": selected_ids,
        },
        "summary_rows": rows,
        "reports": all_reports,
    }
    output_json.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(rows, args.sort_by, args), encoding="utf-8")

    print(f"Wrote aggregate JSON: {output_json}")
    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote per-config reports to: {per_config_dir}")


if __name__ == "__main__":
    main()
