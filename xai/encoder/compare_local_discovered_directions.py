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

from encoder.local_discovered_directions import (
    compute_local_discovered_direction_bundle,
    write_local_discovered_direction_artifacts,
    write_local_discovered_direction_report,
)
from mavrp.configs.config import Config


def _build_parser(level: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run the discovered-direction {level}-level encoder analysis on all configs and compare "
            f"which PCA/ICA directions align most strongly with the {level} concept bank."
        )
    )
    parser.add_argument("--config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=256 if level == "node" else 128)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--inference-batch-size", type=int, default=64 if level == "node" else 32)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument(
        "--max-probe-nodes" if level == "node" else "--max-probe-edges",
        type=int,
        default=20000,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument(
        "--sort-by",
        choices=[
            "mean_best_abs_correlation_top_components",
            "best_abs_correlation_overall",
            "num_components_abs_correlation_ge_0_5",
            "ica_mean_best_abs_correlation_top_components",
            "ica_best_abs_correlation_overall",
            "ica_num_components_abs_correlation_ge_0_5",
            "top3_cumulative_explained_variance_ratio",
            "effective_rank_mean",
        ],
        default="mean_best_abs_correlation_top_components",
    )
    parser.add_argument(
        "--output-json",
        default=f"logs/xai/encoder/{level}/discovered_directions/comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default=f"logs/xai/encoder/{level}/discovered_directions/comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default=f"logs/xai/encoder/{level}/discovered_directions/runs",
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


def _summary_row(config_id: int, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config_id": config_id,
        "model": report["config"]["config_repr"],
        "encoder": report["config"]["encoder_name"],
        "decoder": report["config"]["decoder_name"],
        "mean_best_abs_correlation_top_components": _nested_get(
            report, ["discovered_directions", "summary", "mean_best_abs_correlation_top_components"]
        ),
        "best_abs_correlation_overall": _nested_get(
            report, ["discovered_directions", "summary", "best_abs_correlation_overall"]
        ),
        "num_components_abs_correlation_ge_0_3": _nested_get(
            report, ["discovered_directions", "summary", "num_components_abs_correlation_ge_0_3"]
        ),
        "num_components_abs_correlation_ge_0_5": _nested_get(
            report, ["discovered_directions", "summary", "num_components_abs_correlation_ge_0_5"]
        ),
        "ica_mean_best_abs_correlation_top_components": _nested_get(
            report,
            ["alternative_methods", "ica", "summary", "mean_best_abs_correlation_top_components"],
        ),
        "ica_best_abs_correlation_overall": _nested_get(
            report,
            ["alternative_methods", "ica", "summary", "best_abs_correlation_overall"],
        ),
        "ica_num_components_abs_correlation_ge_0_3": _nested_get(
            report,
            ["alternative_methods", "ica", "summary", "num_components_abs_correlation_ge_0_3"],
        ),
        "ica_num_components_abs_correlation_ge_0_5": _nested_get(
            report,
            ["alternative_methods", "ica", "summary", "num_components_abs_correlation_ge_0_5"],
        ),
        "top1_explained_variance_ratio": _nested_get(
            report, ["discovered_directions", "summary", "top1_explained_variance_ratio"]
        ),
        "top3_cumulative_explained_variance_ratio": _nested_get(
            report, ["discovered_directions", "summary", "top3_cumulative_explained_variance_ratio"]
        ),
        "top5_cumulative_explained_variance_ratio": _nested_get(
            report, ["discovered_directions", "summary", "top5_cumulative_explained_variance_ratio"]
        ),
        "strongest_concept_display": _nested_get(
            report, ["discovered_directions", "summary", "strongest_concept_display"]
        ),
        "strongest_concept_component_index": _nested_get(
            report, ["discovered_directions", "summary", "strongest_concept_component_index"]
        ),
        "ica_strongest_concept_display": _nested_get(
            report, ["alternative_methods", "ica", "summary", "strongest_concept_display"]
        ),
        "ica_strongest_concept_component_index": _nested_get(
            report,
            ["alternative_methods", "ica", "summary", "strongest_concept_component_index"],
        ),
        "effective_rank_mean": _nested_get(report, ["matrix_richness", "effective_rank_mean"]),
        "stable_rank_mean": _nested_get(report, ["matrix_richness", "stable_rank_mean"]),
    }


def _sort_key(row: Dict[str, Any], sort_by: str) -> float:
    value = _safe_float(row.get(sort_by))
    return float("-inf") if value is None else value


def _best_row(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if _safe_float(row.get(key)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _safe_float(row.get(key)) or float("-inf"))


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str, level: str) -> None:
    table = Table(
        title=f"{level.capitalize()} Discovered Directions Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("id", justify="right")
    table.add_column("model", style="bold", overflow="fold", max_width=48)
    table.add_column("pca", overflow="fold")
    table.add_column("ica", overflow="fold")
    table.add_column("variance", overflow="fold")
    table.add_column("strongest", overflow="fold")

    for row in rows:
        table.add_row(
            str(row["config_id"]),
            str(row["model"]),
            "\n".join(
                [
                    f"mean={_fmt(row['mean_best_abs_correlation_top_components'])}",
                    f"best={_fmt(row['best_abs_correlation_overall'])}",
                    f"n>=0.5={_fmt(row['num_components_abs_correlation_ge_0_5'], ndigits=0)}",
                ]
            ),
            "\n".join(
                [
                    f"mean={_fmt(row['ica_mean_best_abs_correlation_top_components'])}",
                    f"best={_fmt(row['ica_best_abs_correlation_overall'])}",
                    f"n>=0.5={_fmt(row['ica_num_components_abs_correlation_ge_0_5'], ndigits=0)}",
                ]
            ),
            "\n".join(
                [
                    f"pc1={_fmt(row['top1_explained_variance_ratio'])}",
                    f"pc3={_fmt(row['top3_cumulative_explained_variance_ratio'])}",
                    f"pc5={_fmt(row['top5_cumulative_explained_variance_ratio'])}",
                ]
            ),
            "\n".join(
                [
                    f"PCA: {row.get('strongest_concept_display') or '-'}",
                    f"ICA: {row.get('ica_strongest_concept_display') or '-'}",
                ]
            ),
        )
    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace, level: str) -> str:
    lines = [
        f"# {level.capitalize()} Discovered Directions Comparison",
        "",
        "## Setup",
        "",
        f"- `num_samples`: `{args.num_samples}`",
        f"- `decode_mode`: `{args.decode_mode}`",
        f"- `inference_batch_size`: `{args.inference_batch_size}`",
        f"- `seed`: `{args.seed}`",
        f"- `num_components`: `{args.num_components}`",
        f"- `top_components`: `{args.top_components}`",
        f"- `sort_by`: `{sort_by}`",
        "",
        "## Metrics Used",
        "",
        "- `mean_best_abs_correlation_top_components`: average best concept alignment over the first PCA directions.",
        "- `ica_mean_best_abs_correlation_top_components`: same measure for ICA directions.",
        "- `best_abs_correlation_overall`: strongest concept alignment reached by any PCA direction.",
        "- `ica_best_abs_correlation_overall`: strongest concept alignment reached by any ICA direction.",
        "- `num_components_abs_correlation_ge_0_5`: number of clearly interpretable directions.",
        "- `top3_cumulative_explained_variance_ratio`: how much PCA variance is concentrated in the first directions.",
        "",
        "## Summary Table",
        "",
        "| id | model | pca_mean | ica_mean | pca_best | ica_best | pca_n>=0.5 | ica_n>=0.5 | pc3 | strongest_pca | strongest_ica |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config_id"]),
                    str(row["model"]),
                    _fmt(row["mean_best_abs_correlation_top_components"]),
                    _fmt(row["ica_mean_best_abs_correlation_top_components"]),
                    _fmt(row["best_abs_correlation_overall"]),
                    _fmt(row["ica_best_abs_correlation_overall"]),
                    _fmt(row["num_components_abs_correlation_ge_0_5"], ndigits=0),
                    _fmt(row["ica_num_components_abs_correlation_ge_0_5"], ndigits=0),
                    _fmt(row["top3_cumulative_explained_variance_ratio"]),
                    str(row.get("strongest_concept_display") or "-"),
                    str(row.get("ica_strongest_concept_display") or "-"),
                ]
            )
            + " |"
        )

    best_pca = _best_row(rows, "mean_best_abs_correlation_top_components")
    best_ica = _best_row(rows, "ica_mean_best_abs_correlation_top_components")
    lines.extend(
        [
            "",
            "## Automatic Interpretation",
            "",
            (
                f"- Best {level}-level PCA config: "
                f"`{best_pca['model']}` with `mean_abs_corr={_fmt(best_pca['mean_best_abs_correlation_top_components'])}`."
                if best_pca is not None
                else f"- Best {level}-level PCA config: not available."
            ),
            (
                f"- Best {level}-level ICA config: "
                f"`{best_ica['model']}` with `mean_abs_corr={_fmt(best_ica['ica_mean_best_abs_correlation_top_components'])}`."
                if best_ica is not None
                else f"- Best {level}-level ICA config: not available."
            ),
            "",
            "## Decision Notes",
            "",
            "- If PCA is strong, the dominant variance axes are already interpretable.",
            "- If ICA is stronger, the latent may contain more separated factors than PCA suggests.",
            f"- This analysis complements the supervised {level}-level probes; it does not replace them.",
        ]
    )
    return "\n".join(lines) + "\n"


def main_for_level(level: str) -> None:
    args = _build_parser(level).parse_args()
    config_ids = set(args.config_ids) if args.config_ids else None
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    per_config_dir = Path(args.per_config_dir)
    per_config_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for config_id, config in enumerate(Config.all()):
        if config_ids is not None and config_id not in config_ids:
            continue
        run_args = SimpleNamespace(
            config_id=config_id,
            checkpoint=None,
            device=args.device,
            graph_size=args.graph_size,
            problem=args.problem,
            num_samples=args.num_samples,
            decode_mode=args.decode_mode,
            inference_batch_size=args.inference_batch_size,
            max_k=args.max_k,
            seed=args.seed,
            num_components=args.num_components,
            top_components=args.top_components,
            max_probe_nodes=getattr(args, "max_probe_nodes", None),
            max_probe_edges=getattr(args, "max_probe_edges", None),
        )
        report, artifacts = compute_local_discovered_direction_bundle(run_args, level=level)

        config_slug = _sanitize_filename(f"{config_id}_{repr(config)}")
        artifact_path = per_config_dir / f"{config_slug}.npz"
        report_path = per_config_dir / f"{config_slug}.json"
        written_artifact = write_local_discovered_direction_artifacts(artifacts, str(artifact_path))
        report["artifacts"] = {"discovered_directions_path": str(written_artifact)}
        write_local_discovered_direction_report(report, str(report_path))

        reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        print(
            f"done config={config_id} model={row['model']} "
            f"pca_mean={_fmt(row['mean_best_abs_correlation_top_components'])} "
            f"ica_mean={_fmt(row['ica_mean_best_abs_correlation_top_components'])}"
        )

    rows.sort(key=lambda row: _sort_key(row, args.sort_by), reverse=True)
    _render_console_table(rows, args.sort_by, level)

    payload = {
        "level": level,
        "sort_by": args.sort_by,
        "summary_rows": rows,
        "reports": reports,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(rows, args.sort_by, args, level), encoding="utf-8")

    print(f"Wrote aggregate JSON: {output_json}")
    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote per-config reports to: {per_config_dir}")


if __name__ == "__main__":
    raise SystemExit("Use compare_node_discovered_directions.py or compare_edge_discovered_directions.py")
