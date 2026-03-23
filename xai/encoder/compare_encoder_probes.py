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

from encoder_probe import compute_probe_bundle, write_probe_artifacts, write_probe_report
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the constraint-focused encoder probe on all configs and print a readable comparison table."
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
            "signature_nmi",
            "signature_ari",
            "signature_macro_f1",
            "route_structure_auc",
            "time_windows_auc",
            "distance_limit_auc",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="signature_nmi",
    )
    parser.add_argument(
        "--output-json",
        default="logs/xai/encoder_probe_comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/xai/encoder_probe_comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default="logs/xai/encoder_probe_runs",
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
        "signature_macro_f1": _nested_get(
            report, ["constraint_signature_separation", "linear_probe", "macro_f1"]
        ),
        "signature_nmi": _nested_get(
            report, ["constraint_signature_separation", "kmeans_aligned", "nmi"]
        ),
        "signature_ari": _nested_get(
            report, ["constraint_signature_separation", "kmeans_aligned", "adjusted_rand"]
        ),
        "route_structure_auc": _nested_get(
            report, ["constraint_group_separation", "route_structure", "linear_probe", "roc_auc"]
        ),
        "time_windows_auc": _nested_get(
            report, ["constraint_flag_separation", "time_windows", "linear_probe", "roc_auc"]
        ),
        "distance_limit_auc": _nested_get(
            report, ["constraint_flag_separation", "distance_limit", "linear_probe", "roc_auc"]
        ),
        "effective_rank_mean": _nested_get(
            report, ["matrix_richness", "effective_rank_mean"]
        ),
        "stable_rank_mean": _nested_get(
            report, ["matrix_richness", "stable_rank_mean"]
        ),
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


def _metric_spread(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values) - min(values)


def _render_interpretation(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = [
        "## Automatic Interpretation",
        "",
    ]

    best_nmi = _best_row(rows, "signature_nmi")
    best_f1 = _best_row(rows, "signature_macro_f1")
    best_rank = _best_row(rows, "effective_rank_mean")
    best_sil = _best_row(rows, "best_silhouette")

    if best_nmi is not None:
        lines.append(
            "- Best natural organization by full constraint signatures: "
            f"`{best_nmi['model']}` with "
            f"`sig_nmi={_fmt(best_nmi['signature_nmi'])}` and "
            f"`sig_ari={_fmt(best_nmi['signature_ari'])}`. "
            "This is the strongest candidate if you want a latent space that naturally clusters by active constraint combinations."
        )

    if best_f1 is not None:
        lines.append(
            "- Best supervised decodability of full constraint signatures: "
            f"`{best_f1['model']}` with "
            f"`sig_f1={_fmt(best_f1['signature_macro_f1'])}`. "
            "This is the strongest candidate if your goal is to recover the constraint signature with a simple linear probe, even if the clusters are not the cleanest."
        )

    if best_rank is not None:
        lines.append(
            "- Richest latent representation by effective rank: "
            f"`{best_rank['model']}` with "
            f"`eff_rank={_fmt(best_rank['effective_rank_mean'])}` and "
            f"`stable_rank={_fmt(best_rank['stable_rank_mean'])}`. "
            "This suggests a less collapsed representation, but richness should always be interpreted together with separation metrics."
        )

    if best_sil is not None:
        lines.append(
            "- Most compact unsupervised clustering geometry by silhouette: "
            f"`{best_sil['model']}` with "
            f"`best_sil={_fmt(best_sil['best_silhouette'])}`. "
            "This is only a geometric diagnostic; it does not guarantee that the clusters correspond to the true constraints."
        )

    if best_nmi is not None and best_sil is not None and best_nmi["model"] != best_sil["model"]:
        lines.append(
            "- `best_sil` and `sig_nmi` do not select the same config. "
            "This is evidence that compact clusters alone are not enough: the best clustering geometry is not necessarily the best alignment with the real constraint structure."
        )

    if best_nmi is not None and best_f1 is not None and best_nmi["model"] != best_f1["model"]:
        lines.append(
            "- `sig_nmi` and `sig_f1` do not select the same config. "
            "This means the best naturally organized latent space is not necessarily the one whose signatures are easiest to decode with supervision."
        )

    tw_spread = _metric_spread(rows, "time_windows_auc")
    route_spread = _metric_spread(rows, "route_structure_auc")
    dist_spread = _metric_spread(rows, "distance_limit_auc")

    if tw_spread is not None and tw_spread < 0.01:
        lines.append(
            "- `tw_auc` is almost constant across configs. "
            "Time-window information is therefore not very discriminative for ranking the methods in this experiment."
        )
    if route_spread is not None and route_spread < 0.01:
        lines.append(
            "- `route_auc` is almost constant across configs. "
            "Route-structure information is present in nearly all models and is not the main differentiator here."
        )
    if dist_spread is not None and dist_spread < 0.01:
        lines.append(
            "- `dist_auc` is almost constant across configs. "
            "Distance-limit information is easy to recover in all models, so it should not drive the ranking by itself."
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- If your criterion is `natural separation of constraint combinations`, prioritize the config with the best `sig_nmi`/`sig_ari`.",
            "- If your criterion is `linear readability of the full constraint signature`, prioritize the config with the best `sig_f1`.",
            "- If your criterion is `rich but still structured latent space`, look for a config that is simultaneously strong in `eff_rank` and in `sig_nmi`/`sig_ari`.",
            "- The table compares full configs, not only encoders. Differences may therefore come from both the encoder and the decoder used during training.",
        ]
    )

    return lines


def _render_recommendations(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = [
        "## Recommendations",
        "",
    ]

    recommended_default = _best_row_lexicographic(
        rows, ["signature_nmi", "effective_rank_mean", "signature_macro_f1"]
    )
    recommended_xai = _best_row_lexicographic(
        rows, ["signature_nmi", "signature_ari", "effective_rank_mean"]
    )
    recommended_supervised = _best_row_lexicographic(
        rows, ["signature_macro_f1", "signature_nmi", "signature_ari"]
    )

    if recommended_default is not None:
        lines.append(
            "- Recommended default config: "
            f"`{recommended_default['model']}`. "
            "This recommendation prioritizes natural organization by constraint signatures first, then latent richness, then supervised readability."
        )

    if recommended_xai is not None:
        lines.append(
            "- Recommended config for XAI: "
            f"`{recommended_xai['model']}`. "
            "This recommendation prioritizes `sig_nmi`, `sig_ari`, and `eff_rank`, so it favors a latent space that is both structured by constraints and sufficiently rich."
        )

    if recommended_supervised is not None:
        lines.append(
            "- Recommended config for supervised probing: "
            f"`{recommended_supervised['model']}`. "
            "This recommendation prioritizes `sig_f1`, then uses cluster-alignment metrics as tie-breakers."
        )

    conclusion_parts: List[str] = []
    if recommended_xai is not None:
        conclusion_parts.append(
            f"for XAI, `{recommended_xai['model']}` is the strongest choice"
        )
    if recommended_supervised is not None and (
        recommended_xai is None or recommended_supervised["model"] != recommended_xai["model"]
    ):
        conclusion_parts.append(
            f"for supervised probing, `{recommended_supervised['model']}` is the strongest choice"
        )

    if conclusion_parts:
        lines.extend(
            [
                "",
                "Conclusion: " + "; ".join(conclusion_parts) + ".",
            ]
        )

    return lines


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Encoder Constraint Probe Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("id", justify="right", style="bold")
    table.add_column("model", overflow="fold", ratio=3)
    table.add_column("signature", overflow="fold", ratio=2)
    table.add_column("flags", overflow="fold", ratio=2)
    table.add_column("richness", overflow="fold", ratio=2)
    table.add_column("clusters", overflow="fold", ratio=2)

    for row in rows:
        signature_cell = "\n".join(
            [
                f"f1={_fmt(row['signature_macro_f1'])}",
                f"nmi={_fmt(row['signature_nmi'])}",
                f"ari={_fmt(row['signature_ari'])}",
            ]
        )
        flags_cell = "\n".join(
            [
                f"route={_fmt(row['route_structure_auc'])}",
                f"tw={_fmt(row['time_windows_auc'])}",
                f"dist={_fmt(row['distance_limit_auc'])}",
            ]
        )
        richness_cell = "\n".join(
            [
                f"er={_fmt(row['effective_rank_mean'])}",
                f"sr={_fmt(row['stable_rank_mean'])}",
            ]
        )
        clusters_cell = "\n".join(
            [
                f"k={_fmt(row['best_k'], ndigits=0)}",
                f"sil={_fmt(row['best_silhouette'])}",
            ]
        )
        table.add_row(
            str(row["config_id"]),
            str(row["model"]),
            signature_cell,
            flags_cell,
            richness_cell,
            clusters_cell,
        )

    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace) -> str:
    lines = [
        "# Encoder Constraint Probe Comparison",
        "",
        f"- sort_by: `{sort_by}`",
        f"- num_samples: `{args.num_samples}`",
        f"- pooling: `{args.pooling}`",
        f"- max_k: `{args.max_k}`",
        "",
        "| id | model | sig_f1 | sig_nmi | sig_ari | route_auc | tw_auc | dist_auc | eff_rank | stable_rank | best_k | best_sil |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config_id"]),
                    str(row["model"]).replace("|", "\\|"),
                    _fmt(row["signature_macro_f1"]),
                    _fmt(row["signature_nmi"]),
                    _fmt(row["signature_ari"]),
                    _fmt(row["route_structure_auc"]),
                    _fmt(row["time_windows_auc"]),
                    _fmt(row["distance_limit_auc"]),
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
            "- `sig_f1`: macro-F1 of a linear probe trained to predict the full constraint signature from the latent representation. Higher is better. A high value means the combinations of active constraints are easy to decode with a simple supervised model. If it is low, the information may be entangled, incomplete, or too imbalanced across signatures.",
            "- `sig_nmi`: normalized mutual information between unsupervised `k-means` clusters and true constraint signatures. Higher is better. `0` means almost no alignment, `1` means perfect alignment. This is a good metric for asking whether the latent space naturally organizes itself by constraint combinations.",
            "- `sig_ari`: adjusted Rand index between `k-means` clusters and true constraint signatures. Higher is better. `0` is close to chance, `1` is perfect, and negative values mean worse than chance. It is stricter than NMI and more sensitive to over-fragmented clusterings.",
            "- `route_auc`: ROC AUC of a linear probe predicting the grouped `route_structure` signal from the latent representation. Higher is better. `0.5` is random guessing, values near `1.0` mean the representation separates route-structure-related constraints very clearly.",
            "- `tw_auc`: ROC AUC of a linear probe predicting the `time_windows` signal. Higher is better. `0.5` is random, `1.0` is perfect separation. If all models are at `1.0`, this constraint is probably too easy to discriminate and is not useful for ranking encoders.",
            "- `dist_auc`: ROC AUC of a linear probe predicting the `distance_limit` signal. Higher is better. `0.5` is random, values close to `1.0` mean the distance-limit constraint is explicitly present and easy to recover from the latent space.",
            "- `eff_rank`: effective rank of the output representation matrix. Higher usually means a richer latent space that uses more directions instead of collapsing to a small subspace. This is not a quality metric by itself: a high value is only useful if separation metrics are also good.",
            "- `stable_rank`: stable-rank-style compact richness measure. Higher usually means the latent representation is less dominated by only a few singular directions. Like `eff_rank`, it should be interpreted jointly with separation metrics rather than alone.",
            "- `best_k`: value of `k` that maximizes silhouette in the `k-means` sweep. This is descriptive, not a target to maximize by itself. A larger `best_k` does not automatically mean a better encoder; it may also reflect fragmentation.",
            "- `best_sil`: best silhouette score found during the `k-means` sweep. Higher is better for compact and separated clusters. Around `1` is very strong, around `0` means overlapping clusters, and negative values mean poor assignments. This metric must not be used alone because compact clusters are not necessarily aligned with the true constraints.",
            "",
            "## How To Read The Table",
            "",
            "- To compare natural organization by constraints, prioritize `sig_nmi` and `sig_ari`.",
            "- To compare how easy it is to decode constraint combinations with supervision, prioritize `sig_f1`.",
            "- To compare whether specific binary constraints are explicitly encoded, use `route_auc`, `tw_auc`, and `dist_auc`.",
            "- To study representation richness, use `eff_rank` and `stable_rank`, but only together with the separation metrics.",
            "- Do not rank encoders with `best_k` or `best_sil` alone. They are useful diagnostics for cluster geometry, not sufficient evidence that the latent space distinguishes the right constraints.",
            "",
            "## Practical Interpretation",
            "",
            "- High `sig_nmi` and high `sig_ari`: the encoder naturally clusters instances according to the true active constraint combinations.",
            "- High `sig_f1` but lower `sig_nmi`/`sig_ari`: the information is present and decodable, but not cleanly organized into natural clusters.",
            "- High `eff_rank` with weak separation metrics: the latent space is rich, but that richness is not specifically structured by constraints.",
            "- Very high `tw_auc` or `route_auc` for every model: these constraints are probably easy and not very discriminative for comparing encoders.",
            "- Strong `best_sil` with weak `sig_nmi`/`sig_ari`: the model forms compact clusters, but they do not correspond well to the real constraint structure.",
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
        report, artifacts = compute_probe_bundle(run_args)
        config_repr = report["config"]["config_repr"]
        artifact_filename = _sanitize_filename(f"{config_id}_{config_repr}.npz")
        artifact_path = write_probe_artifacts(artifacts, str(per_config_dir / artifact_filename))
        report["artifacts"] = {
            "pooled_features_path": str(artifact_path),
        }
        filename = _sanitize_filename(f"{config_id}_{config_repr}.json")
        write_probe_report(report, str(per_config_dir / filename))
        all_reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        console.print(
            f"[cyan]done[/cyan] config={config_id} model={config_repr} "
            f"nmi={_fmt(row['signature_nmi'])} eff_rank={_fmt(row['effective_rank_mean'])}"
            f"dist_auc={_fmt(row['distance_limit_auc'])} tw_auc={_fmt(row['time_windows_auc'])}"
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
