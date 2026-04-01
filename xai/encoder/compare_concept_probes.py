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

from encoder.concept_probe import (
    compute_concept_probe_bundle,
    write_concept_probe_artifacts,
    write_concept_probe_report,
)
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the concept-focused encoder probe on all configs and print a readable comparison table."
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
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sort-by",
        choices=[
            "concept_signature_nmi",
            "concept_signature_ari",
            "concept_signature_macro_f1",
            "concept_macro_f1_mean",
            "compactness_f1",
            "clustering_f1",
            "outlier_f1",
            "outlier_auc",
            "load_concentration_f1",
            "lhbh_balance_f1",
            "capacity_prior_f1",
            "tw_density_f1",
            "tw_width_f1",
            "distance_budget_f1",
            "combined_tension_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="concept_signature_nmi",
    )
    parser.add_argument(
        "--output-json",
        default="logs/xai/encoder/graph/concepts/comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/xai/encoder/graph/concepts/comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default="logs/xai/encoder/graph/concepts/runs",
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
        "concept_signature_macro_f1": _nested_get(
            report, ["concept_signature_separation", "linear_probe", "macro_f1"]
        ),
        "concept_signature_nmi": _nested_get(
            report, ["concept_signature_separation", "kmeans_aligned", "nmi"]
        ),
        "concept_signature_ari": _nested_get(
            report, ["concept_signature_separation", "kmeans_aligned", "adjusted_rand"]
        ),
        "concept_macro_f1_mean": _nested_get(
            report, ["concept_bank", "concept_macro_f1_mean"]
        ),
        "compactness_f1": _nested_get(
            report,
            ["concept_state_separation", "instance_compactness_state", "linear_probe", "macro_f1"],
        ),
        "clustering_f1": _nested_get(
            report,
            ["concept_state_separation", "spatial_clustering_state", "linear_probe", "macro_f1"],
        ),
        "outlier_f1": _nested_get(
            report,
            ["concept_state_separation", "outlier_presence_state", "linear_probe", "macro_f1"],
        ),
        "outlier_auc": _nested_get(
            report,
            ["concept_state_separation", "outlier_presence_state", "linear_probe", "roc_auc"],
        ),
        "load_concentration_f1": _nested_get(
            report,
            ["concept_state_separation", "load_concentration_state", "linear_probe", "macro_f1"],
        ),
        "lhbh_balance_f1": _nested_get(
            report,
            ["concept_state_separation", "linehaul_backhaul_balance_state", "linear_probe", "macro_f1"],
        ),
        "capacity_prior_f1": _nested_get(
            report,
            ["concept_state_separation", "capacity_pressure_prior_state", "linear_probe", "macro_f1"],
        ),
        "tw_density_f1": _nested_get(
            report,
            ["concept_state_separation", "tw_density_state", "linear_probe", "macro_f1"],
        ),
        "tw_width_f1": _nested_get(
            report,
            ["concept_state_separation", "tw_width_profile_state", "linear_probe", "macro_f1"],
        ),
        "distance_budget_f1": _nested_get(
            report,
            ["concept_state_separation", "distance_budget_pressure_state", "linear_probe", "macro_f1"],
        ),
        "combined_tension_f1": _nested_get(
            report,
            ["concept_state_separation", "combined_constraint_tension_state", "linear_probe", "macro_f1"],
        ),
        "effective_rank_mean": _nested_get(report, ["matrix_richness", "effective_rank_mean"]),
        "stable_rank_mean": _nested_get(report, ["matrix_richness", "stable_rank_mean"]),
        "best_k": _nested_get(report, ["best_k_by_silhouette", "k"]),
        "best_silhouette": _nested_get(report, ["best_k_by_silhouette", "silhouette"]),
    }


def _sort_key(row: Dict[str, Any], sort_by: str) -> float:
    value = _safe_float(row.get(sort_by))
    return float("-inf") if value is None else value


def _best_row(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if _safe_float(row.get(key)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _safe_float(row.get(key)) or float("-inf"))


def _best_row_lexicographic(rows: List[Dict[str, Any]], keys: List[str]) -> Optional[Dict[str, Any]]:
    candidates = []
    for row in rows:
        values = [_safe_float(row.get(key)) for key in keys]
        if values[0] is None:
            continue
        candidates.append((row, values))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: tuple(float("-inf") if value is None else value for value in item[1]),
    )[0]


def _render_interpretation(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Automatic Interpretation", ""]
    best_signature = _best_row(rows, "concept_signature_nmi")
    best_mean = _best_row(rows, "concept_macro_f1_mean")
    best_rank = _best_row(rows, "effective_rank_mean")

    if best_signature is not None:
        lines.append(
            "- Best natural organization by core concept signatures: "
            f"`{best_signature['model']}` with "
            f"`sig_nmi={_fmt(best_signature['concept_signature_nmi'])}` and "
            f"`sig_ari={_fmt(best_signature['concept_signature_ari'])}`. "
            "This is the strongest candidate if you want a latent space that naturally clusters by instance-level concepts."
        )
    if best_mean is not None:
        lines.append(
            "- Best average readability across the concept bank: "
            f"`{best_mean['model']}` with "
            f"`concept_mean_f1={_fmt(best_mean['concept_macro_f1_mean'])}`. "
            "This is the strongest candidate if your goal is to recover many static instance concepts with a simple probe."
        )
    if best_rank is not None:
        lines.append(
            "- Richest latent representation by effective rank: "
            f"`{best_rank['model']}` with "
            f"`eff_rank={_fmt(best_rank['effective_rank_mean'])}` and "
            f"`stable_rank={_fmt(best_rank['stable_rank_mean'])}`. "
            "Richness is only useful if the concept metrics are also strong."
        )
    if (
        best_signature is not None
        and best_mean is not None
        and best_signature["model"] != best_mean["model"]
    ):
        lines.append(
            "- `concept_signature_nmi` and `concept_macro_f1_mean` do not select the same config. "
            "This means the most naturally organized concept space is not necessarily the one whose concepts are easiest to decode with supervision."
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- If your criterion is `natural organization of instance concepts`, prioritize `concept_signature_nmi` and `concept_signature_ari`.",
            "- If your criterion is `broad linear readability across concepts`, prioritize `concept_macro_f1_mean`.",
            "- Use the per-concept F1 scores to see which kinds of concepts each encoder captures best: geometry, demand, time, or global tension.",
            "- The table compares full training configs, not only encoders. The decoder used during training may also affect which concepts become explicit in the encoder latent space.",
        ]
    )
    return lines


def _render_recommendations(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Recommendations", ""]
    recommended_default = _best_row_lexicographic(
        rows,
        ["concept_signature_nmi", "concept_macro_f1_mean", "effective_rank_mean"],
    )
    recommended_discovery = _best_row_lexicographic(
        rows,
        ["concept_signature_nmi", "concept_signature_ari", "effective_rank_mean"],
    )
    recommended_supervised = _best_row_lexicographic(
        rows,
        ["concept_macro_f1_mean", "concept_signature_nmi", "concept_signature_ari"],
    )

    if recommended_default is not None:
        lines.append(
            "- Recommended default config: "
            f"`{recommended_default['model']}`. "
            "This recommendation prioritizes natural concept organization first, then broad concept readability, then latent richness."
        )
    if recommended_discovery is not None:
        lines.append(
            "- Recommended config for concept discovery: "
            f"`{recommended_discovery['model']}`. "
            "This recommendation prioritizes the config whose latent space is the most naturally aligned with the core concept signatures."
        )
    if recommended_supervised is not None:
        lines.append(
            "- Recommended config for supervised concept probing: "
            f"`{recommended_supervised['model']}`. "
            "This recommendation prioritizes average concept probe performance across the whole concept bank."
        )
    if recommended_discovery is not None:
        lines.extend(
            [
                "",
                "Conclusion: "
                f"for concept discovery, `{recommended_discovery['model']}` is the strongest choice; "
                f"for broad supervised concept readability, `{recommended_supervised['model']}` is the strongest choice.",
            ]
        )
    return lines


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Encoder Concept Probe Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("id", justify="right", style="bold")
    table.add_column("model", overflow="fold", ratio=3)
    table.add_column("signature", overflow="fold", ratio=2)
    table.add_column("geometry", overflow="fold", ratio=2)
    table.add_column("operations", overflow="fold", ratio=3)
    table.add_column("richness", overflow="fold", ratio=2)

    for row in rows:
        signature_cell = "\n".join(
            [
                f"mean={_fmt(row['concept_macro_f1_mean'])}",
                f"f1={_fmt(row['concept_signature_macro_f1'])}",
                f"nmi={_fmt(row['concept_signature_nmi'])}",
                f"ari={_fmt(row['concept_signature_ari'])}",
            ]
        )
        geometry_cell = "\n".join(
            [
                f"comp={_fmt(row['compactness_f1'])}",
                f"clus={_fmt(row['clustering_f1'])}",
                f"out={_fmt(row['outlier_f1'])}",
                f"out_auc={_fmt(row['outlier_auc'])}",
            ]
        )
        operations_cell = "\n".join(
            [
                f"load={_fmt(row['load_concentration_f1'])}",
                f"bal={_fmt(row['lhbh_balance_f1'])}",
                f"cap={_fmt(row['capacity_prior_f1'])}",
                f"tw_d={_fmt(row['tw_density_f1'])}",
                f"tw_w={_fmt(row['tw_width_f1'])}",
                f"dist={_fmt(row['distance_budget_f1'])}",
                f"tens={_fmt(row['combined_tension_f1'])}",
            ]
        )
        richness_cell = "\n".join(
            [
                f"er={_fmt(row['effective_rank_mean'])}",
                f"sr={_fmt(row['stable_rank_mean'])}",
                f"k={_fmt(row['best_k'], ndigits=0)}",
                f"sil={_fmt(row['best_silhouette'])}",
            ]
        )
        table.add_row(
            str(row["config_id"]),
            str(row["model"]),
            signature_cell,
            geometry_cell,
            operations_cell,
            richness_cell,
        )
    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace) -> str:
    lines = [
        "# Encoder Concept Probe Comparison",
        "",
        f"- sort_by: `{sort_by}`",
        f"- num_samples: `{args.num_samples}`",
        f"- pooling: `{args.pooling}`",
        f"- max_k: `{args.max_k}`",
        "",
        "| id | model | concept_mean_f1 | sig_f1 | sig_nmi | sig_ari | compact | cluster | outlier_f1 | outlier_auc | load | lh_bh | cap | tw_density | tw_width | dist_budget | tension | eff_rank | stable_rank | best_k | best_sil |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config_id"]),
                    str(row["model"]).replace("|", "\\|"),
                    _fmt(row["concept_macro_f1_mean"]),
                    _fmt(row["concept_signature_macro_f1"]),
                    _fmt(row["concept_signature_nmi"]),
                    _fmt(row["concept_signature_ari"]),
                    _fmt(row["compactness_f1"]),
                    _fmt(row["clustering_f1"]),
                    _fmt(row["outlier_f1"]),
                    _fmt(row["outlier_auc"]),
                    _fmt(row["load_concentration_f1"]),
                    _fmt(row["lhbh_balance_f1"]),
                    _fmt(row["capacity_prior_f1"]),
                    _fmt(row["tw_density_f1"]),
                    _fmt(row["tw_width_f1"]),
                    _fmt(row["distance_budget_f1"]),
                    _fmt(row["combined_tension_f1"]),
                    _fmt(row["effective_rank_mean"]),
                    _fmt(row["stable_rank_mean"]),
                    _fmt(row["best_k"], ndigits=0),
                    _fmt(row["best_silhouette"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metric Guide",
            "",
            "- `concept_mean_f1`: mean macro-F1 across the whole concept bank. Higher is better. It summarizes how readable the chosen instance concepts are on average with simple linear probes.",
            "- `sig_f1`: macro-F1 of a linear probe trained to predict the core concept signature. Higher is better. This tests whether a compact combination of representative concepts is jointly readable from the latent space. Rare signature singletons are collapsed into `other_rare` for this probe so that the metric stays estimable.",
            "- `sig_nmi`: normalized mutual information between unsupervised `k-means` clusters and the probeable core concept signatures. Higher is better. `0` means little alignment, `1` means perfect alignment. This is the main metric for asking whether concept structure appears naturally in the latent space.",
            "- `sig_ari`: adjusted Rand index between unsupervised clusters and the probeable core concept signatures. Higher is better. `0` is near chance and `1` is perfect. It is stricter than NMI.",
            "- `compact`, `cluster`, `outlier_f1`: macro-F1 for geometry concepts. Higher is better. Here `outlier` refers to a robust binary `has_remote_outlier / no_remote_outlier` concept, not to a fragile multi-class outlier count.",
            "- `outlier_auc`: ROC AUC for the binary `outlier_presence` concept. Higher is better. `0.5` is near chance and `1.0` is perfect separation. This complements `outlier_f1`: AUC tells you whether the latent ranks remote-outlier instances correctly even when the classification threshold is imperfect.",
            "- `load`, `lh_bh`, `cap`: macro-F1 for demand and load concepts. Higher is better. These scores tell you whether the encoder organizes customer demand structure, backhaul balance, and prior capacity pressure.",
            "- `tw_density`, `tw_width`, `dist_budget`, `tension`: macro-F1 for temporal and global difficulty concepts. Higher is better. They tell you whether the encoder keeps a readable notion of temporal structure, distance-budget pressure, and combined instance tension.",
            "- `eff_rank` and `stable_rank`: latent richness metrics. Higher usually means the latent uses more directions, but richness alone is not enough; concept metrics must also be strong.",
            "- `best_k` and `best_sil`: unsupervised clustering diagnostics. They describe geometry, not semantic alignment by themselves.",
            "",
            "## How To Read The Table",
            "",
            "- To compare natural concept organization, prioritize `sig_nmi` and `sig_ari`.",
            "- To compare broad supervised concept readability, prioritize `concept_mean_f1`.",
            "- To compare a specific kind of concept, read the relevant per-concept F1 columns directly.",
            "- Use `eff_rank` and `stable_rank` only together with the concept metrics.",
        ]
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
            max_k=args.max_k,
            seed=args.seed,
            output=None,
            artifacts_output=None,
        )
        report, artifacts = compute_concept_probe_bundle(run_args)
        config_repr = report["config"]["config_repr"]
        artifact_filename = _sanitize_filename(f"{config_id}_{config_repr}.npz")
        artifact_path = write_concept_probe_artifacts(artifacts, str(per_config_dir / artifact_filename))
        report["artifacts"] = {"pooled_features_path": str(artifact_path)}
        filename = _sanitize_filename(f"{config_id}_{config_repr}.json")
        write_concept_probe_report(report, str(per_config_dir / filename))
        all_reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        console.print(
            f"[cyan]done[/cyan] config={config_id} model={config_repr} "
            f"nmi={_fmt(row['concept_signature_nmi'])} "
            f"mean_f1={_fmt(row['concept_macro_f1_mean'])}"
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
            "max_k": args.max_k,
            "seed": args.seed,
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
