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

from encoder.discovered_directions import (
    compute_discovered_direction_bundle,
    write_discovered_direction_artifacts,
    write_discovered_direction_report,
)
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the discovered-direction encoder analysis on all configs and compare "
            "which principal directions align most strongly with the concept bank."
        )
    )
    parser.add_argument("--config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=1024)
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
            "mean_best_abs_correlation_top_components",
            "best_abs_correlation_overall",
            "num_components_abs_correlation_ge_0_5",
            "ica_mean_best_abs_correlation_top_components",
            "ica_best_abs_correlation_overall",
            "ica_num_components_abs_correlation_ge_0_5",
            "top3_cumulative_explained_variance_ratio",
            "effective_rank_mean",
            "distance_budget_best_abs_correlation",
            "combined_tension_best_abs_correlation",
        ],
        default="mean_best_abs_correlation_top_components",
    )
    parser.add_argument(
        "--output-json",
        default="logs/xai/encoder/discovered_directions/comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/xai/encoder/discovered_directions/comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default="logs/xai/encoder/discovered_directions/runs",
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
        "distance_budget_best_abs_correlation": _nested_get(
            report,
            [
                "discovered_directions",
                "concept_alignment",
                "distance_budget_pressure_state",
                "best_abs_correlation",
            ],
        ),
        "ica_distance_budget_best_abs_correlation": _nested_get(
            report,
            [
                "alternative_methods",
                "ica",
                "concept_alignment",
                "distance_budget_pressure_state",
                "best_abs_correlation",
            ],
        ),
        "combined_tension_best_abs_correlation": _nested_get(
            report,
            [
                "discovered_directions",
                "concept_alignment",
                "combined_constraint_tension_state",
                "best_abs_correlation",
            ],
        ),
        "ica_combined_tension_best_abs_correlation": _nested_get(
            report,
            [
                "alternative_methods",
                "ica",
                "concept_alignment",
                "combined_constraint_tension_state",
                "best_abs_correlation",
            ],
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


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Encoder Discovered Directions Comparison (sorted by {sort_by})",
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
    table.add_column("richness", overflow="fold")

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
                    f"PCA {row['strongest_concept_display'] or '-'} @pc={_fmt(row['strongest_concept_component_index'], ndigits=0)}",
                    f"ICA {row['ica_strongest_concept_display'] or '-'} @ic={_fmt(row['ica_strongest_concept_component_index'], ndigits=0)}",
                    f"PCA dist={_fmt(row['distance_budget_best_abs_correlation'])} / tens={_fmt(row['combined_tension_best_abs_correlation'])}",
                    f"ICA dist={_fmt(row['ica_distance_budget_best_abs_correlation'])} / tens={_fmt(row['ica_combined_tension_best_abs_correlation'])}",
                ]
            ),
            "\n".join(
                [
                    f"er={_fmt(row['effective_rank_mean'])}",
                    f"sr={_fmt(row['stable_rank_mean'])}",
                ]
            ),
        )
    Console().print(table)


def _render_interpretation(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Automatic Interpretation", ""]
    best_alignment = _best_row(rows, "mean_best_abs_correlation_top_components")
    best_clean_axes = _best_row(rows, "num_components_abs_correlation_ge_0_5")
    best_variance = _best_row(rows, "top3_cumulative_explained_variance_ratio")
    best_ica_alignment = _best_row(rows, "ica_mean_best_abs_correlation_top_components")

    if best_alignment is not None:
        lines.append(
            "- Best discovered-direction alignment with the concept bank: "
            f"`{best_alignment['model']}` with "
            f"`mean_abs_corr={_fmt(best_alignment['mean_best_abs_correlation_top_components'])}`. "
            "This config makes the first latent directions easiest to interpret after the fact."
        )
    if best_ica_alignment is not None:
        lines.append(
            "- Best ICA alignment with the concept bank: "
            f"`{best_ica_alignment['model']}` with "
            f"`ica_mean_abs_corr={_fmt(best_ica_alignment['ica_mean_best_abs_correlation_top_components'])}`. "
            "This is the strongest config if you want independent latent factors rather than variance-maximizing ones."
        )
    if best_clean_axes is not None:
        lines.append(
            "- Most clean discovered directions (`abs(corr) >= 0.5`): "
            f"`{best_clean_axes['model']}` with "
            f"`n_components={_fmt(best_clean_axes['num_components_abs_correlation_ge_0_5'], ndigits=0)}`. "
            "This favors configs whose leading components each align with a simple known concept."
        )
    if best_variance is not None:
        lines.append(
            "- Highest concentration of variance in the first three components: "
            f"`{best_variance['model']}` with "
            f"`top3_evr={_fmt(best_variance['top3_cumulative_explained_variance_ratio'])}`. "
            "High variance concentration is only useful if the aligned concepts are still interpretable."
        )
    if (
        best_alignment is not None
        and best_clean_axes is not None
        and best_alignment["model"] != best_clean_axes["model"]
    ):
        lines.append(
            "- The config with the strongest average alignment is not the same as the one with the most clean axes. "
            "This means one model can distribute interpretable information broadly, while another concentrates it in fewer but clearer components."
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Use `mean_abs_corr` to compare broad interpretability of the first discovered directions.",
            "- Use `n>=0.5` when you want a few very clear axes rather than many moderate ones.",
            "- Read the per-concept heatmap to see which instance properties become explicit directions: geometry, load, distance budget, or combined tension.",
            "- Keep `effective_rank` as a secondary signal: rich spaces are useful only when the discovered directions stay interpretable.",
        ]
    )
    return lines


def _render_recommendations(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Recommendations", ""]
    best_default = _best_row(rows, "mean_best_abs_correlation_top_components")
    best_xai = _best_row(rows, "num_components_abs_correlation_ge_0_5")
    best_ica = _best_row(rows, "ica_mean_best_abs_correlation_top_components")

    if best_default is not None:
        lines.append(
            "- Recommended default config for discovered latent analysis: "
            f"`{best_default['model']}`."
        )
    if best_xai is not None:
        lines.append(
            "- Recommended config when you want the clearest discovered axes: "
            f"`{best_xai['model']}`."
        )
    if best_ica is not None:
        lines.append(
            "- Recommended config under ICA analysis: "
            f"`{best_ica['model']}`."
        )
    if best_default is not None:
        lines.append(
            "- Conclusion: start from "
            f"`{best_default['model']}` when studying what the encoder organizes by itself, "
            "then inspect the per-component concept alignments to decide whether another config has cleaner specialized axes."
        )
    return lines


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace) -> str:
    headers = [
        "id",
        "model",
        "mean_abs_corr",
        "ica_mean_abs_corr",
        "best_abs_corr",
        "ica_best_abs_corr",
        "n_ge_0_5",
        "ica_n_ge_0_5",
        "pc1_evr",
        "pc3_evr",
        "strongest_concept",
        "ica_strongest_concept",
        "distance_budget_corr",
        "ica_distance_budget_corr",
        "combined_tension_corr",
        "ica_combined_tension_corr",
        "eff_rank",
        "stable_rank",
    ]
    lines = [
        "# Encoder Discovered Directions Comparison",
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
        "- `mean_abs_corr` / `ica_mean_abs_corr`: moyenne du meilleur `|corr|` des premières composantes avec les concepts connus. Plus c'est haut, plus les directions découvertes sont facilement interprétables.",
        "- `best_abs_corr` / `ica_best_abs_corr`: meilleure corrélation absolue trouvée entre une composante et un concept. Plus c'est haut, plus au moins une direction est très lisible.",
        "- `n_ge_0_5` / `ica_n_ge_0_5`: nombre de composantes dont la meilleure corrélation absolue atteint au moins `0.5`. Plus c'est haut, plus il existe d'axes clairs.",
        "- `pc1_evr` / `pc3_evr`: variance expliquée par la première composante PCA, puis cumul des trois premières. Ces métriques n'existent que pour PCA.",
        "- `distance_budget_corr` / `combined_tension_corr`: meilleure corrélation absolue atteinte pour ces concepts précis.",
        "- `eff_rank` / `stable_rank`: richesse globale du latent, à lire comme métriques secondaires.",
        "",
        "## Summary Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
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
                    _fmt(row["top1_explained_variance_ratio"]),
                    _fmt(row["top3_cumulative_explained_variance_ratio"]),
                    str(row["strongest_concept_display"] or "-"),
                    str(row["ica_strongest_concept_display"] or "-"),
                    _fmt(row["distance_budget_best_abs_correlation"]),
                    _fmt(row["ica_distance_budget_best_abs_correlation"]),
                    _fmt(row["combined_tension_best_abs_correlation"]),
                    _fmt(row["ica_combined_tension_best_abs_correlation"]),
                    _fmt(row["effective_rank_mean"]),
                    _fmt(row["stable_rank_mean"]),
                ]
            )
            + " |"
        )
    lines.extend([""] + _render_interpretation(rows))
    lines.extend([""] + _render_recommendations(rows))
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
            output=None,
            artifacts_output=None,
        )
        report, artifacts = compute_discovered_direction_bundle(run_args)
        config_repr = report["config"]["config_repr"]
        artifact_filename = _sanitize_filename(f"{config_id}_{config_repr}.npz")
        artifact_path = write_discovered_direction_artifacts(
            artifacts,
            str(per_config_dir / artifact_filename),
        )
        report["artifacts"] = {"discovered_directions_path": str(artifact_path)}
        filename = _sanitize_filename(f"{config_id}_{config_repr}.json")
        write_discovered_direction_report(report, str(per_config_dir / filename))
        all_reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        console.print(
            f"[cyan]done[/cyan] config={config_id} model={config_repr} "
            f"mean_abs_corr={_fmt(row['mean_best_abs_correlation_top_components'])} "
            f"best_abs_corr={_fmt(row['best_abs_correlation_overall'])}"
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
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(rows, args.sort_by, args), encoding="utf-8")

    print(f"Wrote aggregate JSON: {output_json}")
    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote per-config reports to: {per_config_dir}")


if __name__ == "__main__":
    main()
