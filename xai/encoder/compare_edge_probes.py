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

from encoder.edge_probe import (
    compute_edge_probe_bundle,
    write_edge_probe_artifacts,
    write_edge_probe_report,
)
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the edge-level encoder probe on all configs and print a readable comparison table."
    )
    parser.add_argument("--config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-probe-edges", type=int, default=20000)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sort-by",
        choices=[
            "edge_signature_nmi",
            "edge_signature_ari",
            "edge_signature_macro_f1",
            "edge_macro_f1_mean",
            "same_route_f1",
            "same_route_auc",
            "solution_edge_f1",
            "solution_edge_auc",
            "local_edge_cost_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="edge_signature_nmi",
    )
    parser.add_argument("--output-json", default="logs/xai/encoder/edge_probes/comparison.json")
    parser.add_argument("--output-md", default="logs/xai/encoder/edge_probes/comparison.md")
    parser.add_argument("--per-config-dir", default="logs/xai/encoder/edge_probes/runs")
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
        "edge_signature_macro_f1": _nested_get(
            report, ["edge_concept_signature_separation", "linear_probe", "macro_f1"]
        ),
        "edge_signature_nmi": _nested_get(
            report, ["edge_concept_signature_separation", "kmeans_aligned", "nmi"]
        ),
        "edge_signature_ari": _nested_get(
            report, ["edge_concept_signature_separation", "kmeans_aligned", "adjusted_rand"]
        ),
        "edge_macro_f1_mean": _nested_get(report, ["edge_concept_bank", "concept_macro_f1_mean"]),
        "same_route_f1": _nested_get(
            report,
            ["edge_concept_state_separation", "same_route_state", "linear_probe", "macro_f1"],
        ),
        "same_route_auc": _nested_get(
            report,
            ["edge_concept_state_separation", "same_route_state", "linear_probe", "roc_auc"],
        ),
        "solution_edge_f1": _nested_get(
            report,
            ["edge_concept_state_separation", "edge_in_solution_state", "linear_probe", "macro_f1"],
        ),
        "solution_edge_auc": _nested_get(
            report,
            ["edge_concept_state_separation", "edge_in_solution_state", "linear_probe", "roc_auc"],
        ),
        "local_edge_cost_f1": _nested_get(
            report,
            ["edge_concept_state_separation", "local_edge_cost_contribution_state", "linear_probe", "macro_f1"],
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
    best_signature = _best_row(rows, "edge_signature_nmi")
    best_mean = _best_row(rows, "edge_macro_f1_mean")
    best_same = _best_row(rows, "same_route_auc")
    best_solution = _best_row(rows, "solution_edge_auc")
    best_cost = _best_row(rows, "local_edge_cost_f1")

    if best_signature is not None:
        lines.append(
            "- Best natural organization by edge-level concept signatures: "
            f"`{best_signature['model']}` with "
            f"`sig_nmi={_fmt(best_signature['edge_signature_nmi'])}` and "
            f"`sig_ari={_fmt(best_signature['edge_signature_ari'])}`."
        )
    if best_mean is not None:
        lines.append(
            "- Best average edge-level readability: "
            f"`{best_mean['model']}` with "
            f"`edge_mean_f1={_fmt(best_mean['edge_macro_f1_mean'])}`."
        )
    if best_same is not None:
        lines.append(
            "- Best config for predicting whether two customers end up on the same route: "
            f"`{best_same['model']}` with "
            f"`same_route_auc={_fmt(best_same['same_route_auc'])}`."
        )
    if best_solution is not None:
        lines.append(
            "- Best config for predicting whether a directed customer-to-customer edge belongs to the final solution: "
            f"`{best_solution['model']}` with "
            f"`solution_edge_auc={_fmt(best_solution['solution_edge_auc'])}`."
        )
    if best_cost is not None:
        lines.append(
            "- Best config for predicting the local edge-cost bucket: "
            f"`{best_cost['model']}` with "
            f"`local_edge_cost_f1={_fmt(best_cost['local_edge_cost_f1'])}`."
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- `same_route_auc` tests whether pair embeddings make route co-membership readable.",
            "- `solution_edge_auc` tests whether pair embeddings make actual route adjacency readable.",
            "- `local_edge_cost_f1` is a local edge-cost proxy on edges that actually appear in the solution.",
            "- `edge_signature_nmi` is the strongest metric if you care about naturally organized edge-level decision concepts.",
        ]
    )
    return lines


def _render_recommendations(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Recommendations", ""]
    recommended_default = _best_row_lexicographic(
        rows,
        ["edge_signature_nmi", "edge_macro_f1_mean", "effective_rank_mean"],
    )
    recommended_discovery = _best_row_lexicographic(
        rows,
        ["edge_signature_nmi", "edge_signature_ari", "effective_rank_mean"],
    )
    recommended_supervised = _best_row_lexicographic(
        rows,
        ["edge_macro_f1_mean", "solution_edge_auc", "same_route_auc"],
    )
    if recommended_default is not None:
        lines.append(f"- Recommended default config: `{recommended_default['model']}`.")
    if recommended_discovery is not None:
        lines.append(
            "- Recommended config for edge-level concept discovery: "
            f"`{recommended_discovery['model']}`."
        )
    if recommended_supervised is not None:
        lines.append(
            "- Recommended config for supervised edge-level probing: "
            f"`{recommended_supervised['model']}`."
        )
    if recommended_discovery is not None and recommended_supervised is not None:
        lines.extend(
            [
                "",
                "Conclusion: "
                f"for naturally organized edge-level concepts, `{recommended_discovery['model']}` is the strongest choice; "
                f"for supervised pair-readability, `{recommended_supervised['model']}` is the strongest choice.",
            ]
        )
    return lines


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Encoder Edge Probe Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("id", justify="right", style="bold")
    table.add_column("model", overflow="fold", ratio=3)
    table.add_column("signature", overflow="fold", ratio=2)
    table.add_column("edge concepts", overflow="fold", ratio=3)
    table.add_column("richness", overflow="fold", ratio=2)

    for row in rows:
        signature_cell = "\n".join(
            [
                f"mean={_fmt(row['edge_macro_f1_mean'])}",
                f"f1={_fmt(row['edge_signature_macro_f1'])}",
                f"nmi={_fmt(row['edge_signature_nmi'])}",
                f"ari={_fmt(row['edge_signature_ari'])}",
            ]
        )
        edge_cell = "\n".join(
            [
                f"same={_fmt(row['same_route_f1'])}",
                f"same_auc={_fmt(row['same_route_auc'])}",
                f"edge={_fmt(row['solution_edge_f1'])}",
                f"edge_auc={_fmt(row['solution_edge_auc'])}",
                f"cost={_fmt(row['local_edge_cost_f1'])}",
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
            edge_cell,
            richness_cell,
        )

    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str) -> str:
    headers = [
        "id",
        "model",
        "edge_mean_f1",
        "sig_f1",
        "sig_nmi",
        "sig_ari",
        "same_route_f1",
        "same_route_auc",
        "solution_edge_f1",
        "solution_edge_auc",
        "local_edge_cost_f1",
        "eff_rank",
        "stable_rank",
        "best_k",
        "best_sil",
    ]
    lines = [
        "# Encoder Edge Probe Comparison",
        "",
        f"Sorted by `{sort_by}`.",
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
                    f"`{row['model']}`",
                    _fmt(row["edge_macro_f1_mean"]),
                    _fmt(row["edge_signature_macro_f1"]),
                    _fmt(row["edge_signature_nmi"]),
                    _fmt(row["edge_signature_ari"]),
                    _fmt(row["same_route_f1"]),
                    _fmt(row["same_route_auc"]),
                    _fmt(row["solution_edge_f1"]),
                    _fmt(row["solution_edge_auc"]),
                    _fmt(row["local_edge_cost_f1"]),
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
            "- `edge_mean_f1`: average macro-F1 across the edge-level concept bank. Higher is better.",
            "- `sig_f1`: macro-F1 of a linear probe predicting the joint edge-level concept signature. Higher is better.",
            "- `sig_nmi`: alignment between k-means clusters and edge-level concept signatures. Higher is better.",
            "- `sig_ari`: same idea as `sig_nmi`, but corrected for chance. Higher is better.",
            "- `same_route_f1` / `same_route_auc`: can a simple probe predict whether two customers end up on the same route?",
            "- `solution_edge_f1` / `solution_edge_auc`: can a simple probe predict whether a directed customer-to-customer edge is used in the final solution?",
            "- `local_edge_cost_f1`: can a simple probe predict a low/medium/high local edge-cost bucket on used edges?",
            "- `eff_rank` and `stable_rank`: richness of the underlying node embedding matrices before building pair features.",
            "",
        ]
    )
    lines.extend(_render_interpretation(rows))
    lines.extend([""])
    lines.extend(_render_recommendations(rows))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    configs = Config.all()
    config_ids = args.config_ids if args.config_ids is not None else list(range(len(configs)))

    per_config_dir = Path(args.per_config_dir)
    per_config_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for config_id in config_ids:
        probe_args = SimpleNamespace(
            config_id=int(config_id),
            checkpoint=None,
            device=args.device,
            graph_size=args.graph_size,
            problem=args.problem,
            num_samples=args.num_samples,
            decode_mode=args.decode_mode,
            inference_batch_size=args.inference_batch_size,
            max_probe_edges=args.max_probe_edges,
            max_k=args.max_k,
            seed=args.seed,
            output="",
            artifacts_output="",
        )
        report, artifacts = compute_edge_probe_bundle(probe_args)
        safe_name = _sanitize_filename(report["config"]["config_repr"])
        report_path = per_config_dir / f"{config_id}_{safe_name}.json"
        artifact_path = per_config_dir / f"{config_id}_{safe_name}.npz"
        report["artifacts"] = {"edge_features_path": str(artifact_path)}
        report["paths"] = {"report_json": str(report_path)}
        write_edge_probe_report(report, str(report_path))
        write_edge_probe_artifacts(artifacts, str(artifact_path))
        reports.append(report)
        summary = _summary_row(int(config_id), report)
        summary_rows.append(summary)
        print(
            f"done config={config_id} model={summary['model']} "
            f"nmi={_fmt(summary['edge_signature_nmi'])} "
            f"same_auc={_fmt(summary['same_route_auc'])} "
            f"edge_auc={_fmt(summary['solution_edge_auc'])}"
        )

    ordered_rows = sorted(
        summary_rows,
        key=lambda row: (_sort_key(row, args.sort_by), -int(row["config_id"])),
        reverse=True,
    )

    aggregate = {
        "sort_by": str(args.sort_by),
        "rows": ordered_rows,
        "reports": reports,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        _render_markdown(ordered_rows, sort_by=args.sort_by),
        encoding="utf-8",
    )

    _render_console_table(ordered_rows, sort_by=args.sort_by)
    print(f"Wrote aggregate JSON: {output_json}")
    print(f"Wrote markdown summary: {output_md}")
    print(f"Wrote per-config reports to: {per_config_dir}")


if __name__ == "__main__":
    main()
