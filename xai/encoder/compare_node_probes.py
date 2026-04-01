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

from encoder.node_probe import (
    compute_node_probe_bundle,
    write_node_probe_artifacts,
    write_node_probe_report,
)
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the node-level encoder probe on all configs and print a readable comparison table."
    )
    parser.add_argument("--config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-probe-nodes", type=int, default=20000)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sort-by",
        choices=[
            "node_signature_nmi",
            "node_signature_ari",
            "node_signature_macro_f1",
            "node_macro_f1_mean",
            "service_order_f1",
            "route_role_f1",
            "local_cost_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="node_signature_nmi",
    )
    parser.add_argument("--output-json", default="logs/xai/encoder/node/probes/comparison.json")
    parser.add_argument("--output-md", default="logs/xai/encoder/node/probes/comparison.md")
    parser.add_argument("--per-config-dir", default="logs/xai/encoder/node/probes/runs")
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
        "node_signature_macro_f1": _nested_get(
            report, ["node_concept_signature_separation", "linear_probe", "macro_f1"]
        ),
        "node_signature_nmi": _nested_get(
            report, ["node_concept_signature_separation", "kmeans_aligned", "nmi"]
        ),
        "node_signature_ari": _nested_get(
            report, ["node_concept_signature_separation", "kmeans_aligned", "adjusted_rand"]
        ),
        "node_macro_f1_mean": _nested_get(
            report, ["node_concept_bank", "concept_macro_f1_mean"]
        ),
        "service_order_f1": _nested_get(
            report,
            ["node_concept_state_separation", "service_order_state", "linear_probe", "macro_f1"],
        ),
        "route_role_f1": _nested_get(
            report,
            ["node_concept_state_separation", "route_role_state", "linear_probe", "macro_f1"],
        ),
        "local_cost_f1": _nested_get(
            report,
            ["node_concept_state_separation", "local_cost_contribution_state", "linear_probe", "macro_f1"],
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
    best_signature = _best_row(rows, "node_signature_nmi")
    best_mean = _best_row(rows, "node_macro_f1_mean")
    best_rank = _best_row(rows, "effective_rank_mean")
    best_service = _best_row(rows, "service_order_f1")
    best_role = _best_row(rows, "route_role_f1")
    best_cost = _best_row(rows, "local_cost_f1")

    if best_signature is not None:
        lines.append(
            "- Best natural organization by node-level concept signatures: "
            f"`{best_signature['model']}` with "
            f"`sig_nmi={_fmt(best_signature['node_signature_nmi'])}` and "
            f"`sig_ari={_fmt(best_signature['node_signature_ari'])}`."
        )
    if best_mean is not None:
        lines.append(
            "- Best average node-level readability: "
            f"`{best_mean['model']}` with "
            f"`node_mean_f1={_fmt(best_mean['node_macro_f1_mean'])}`."
        )
    if best_service is not None:
        lines.append(
            "- Best config for predicting early/middle/late service order: "
            f"`{best_service['model']}` with "
            f"`service_order_f1={_fmt(best_service['service_order_f1'])}`."
        )
    if best_role is not None:
        lines.append(
            "- Best config for predicting the node route role (`start`, `middle`, `end`, `singleton`): "
            f"`{best_role['model']}` with "
            f"`route_role_f1={_fmt(best_role['route_role_f1'])}`."
        )
    if best_cost is not None:
        lines.append(
            "- Best config for predicting the local cost-impact proxy: "
            f"`{best_cost['model']}` with "
            f"`local_cost_f1={_fmt(best_cost['local_cost_f1'])}`."
        )
    if best_rank is not None:
        lines.append(
            "- Richest latent representation by effective rank: "
            f"`{best_rank['model']}` with "
            f"`eff_rank={_fmt(best_rank['effective_rank_mean'])}`."
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- `service_order_f1` tests whether the encoder already organizes which customers tend to be served early or late in the final solution.",
            "- `route_role_f1` tests whether the encoder makes route position readable at the node level.",
            "- `local_cost_f1` tests a local removal-cost proxy, not the exact counterfactual marginal contribution.",
            "- `node_signature_nmi` is the strongest metric if you care about naturally organized node-level decision concepts.",
        ]
    )
    return lines


def _render_recommendations(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Recommendations", ""]
    recommended_default = _best_row_lexicographic(
        rows,
        ["node_signature_nmi", "node_macro_f1_mean", "effective_rank_mean"],
    )
    recommended_discovery = _best_row_lexicographic(
        rows,
        ["node_signature_nmi", "node_signature_ari", "effective_rank_mean"],
    )
    recommended_supervised = _best_row_lexicographic(
        rows,
        ["node_macro_f1_mean", "node_signature_nmi", "node_signature_ari"],
    )
    if recommended_default is not None:
        lines.append(
            "- Recommended default config: "
            f"`{recommended_default['model']}`."
        )
    if recommended_discovery is not None:
        lines.append(
            "- Recommended config for node-level concept discovery: "
            f"`{recommended_discovery['model']}`."
        )
    if recommended_supervised is not None:
        lines.append(
            "- Recommended config for supervised node-level probing: "
            f"`{recommended_supervised['model']}`."
        )
    if recommended_discovery is not None and recommended_supervised is not None:
        lines.extend(
            [
                "",
                "Conclusion: "
                f"for naturally organized node-level concepts, `{recommended_discovery['model']}` is the strongest choice; "
                f"for broad supervised readability, `{recommended_supervised['model']}` is the strongest choice.",
            ]
        )
    return lines


def _render_console_table(rows: List[Dict[str, Any]], sort_by: str) -> None:
    table = Table(
        title=f"Encoder Node Probe Comparison (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("id", justify="right", style="bold")
    table.add_column("model", overflow="fold", ratio=3)
    table.add_column("signature", overflow="fold", ratio=2)
    table.add_column("node concepts", overflow="fold", ratio=3)
    table.add_column("richness", overflow="fold", ratio=2)

    for row in rows:
        signature_cell = "\n".join(
            [
                f"mean={_fmt(row['node_macro_f1_mean'])}",
                f"f1={_fmt(row['node_signature_macro_f1'])}",
                f"nmi={_fmt(row['node_signature_nmi'])}",
                f"ari={_fmt(row['node_signature_ari'])}",
            ]
        )
        node_cell = "\n".join(
            [
                f"order={_fmt(row['service_order_f1'])}",
                f"role={_fmt(row['route_role_f1'])}",
                f"cost={_fmt(row['local_cost_f1'])}",
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
            node_cell,
            richness_cell,
        )

    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str) -> str:
    headers = [
        "id",
        "model",
        "node_mean_f1",
        "sig_f1",
        "sig_nmi",
        "sig_ari",
        "service_order_f1",
        "route_role_f1",
        "local_cost_f1",
        "eff_rank",
        "stable_rank",
        "best_k",
        "best_sil",
    ]
    lines = [
        f"# Encoder Node Probe Comparison",
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
                    _fmt(row["node_macro_f1_mean"]),
                    _fmt(row["node_signature_macro_f1"]),
                    _fmt(row["node_signature_nmi"]),
                    _fmt(row["node_signature_ari"]),
                    _fmt(row["service_order_f1"]),
                    _fmt(row["route_role_f1"]),
                    _fmt(row["local_cost_f1"]),
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
            "- `node_mean_f1`: average macro-F1 across the node-level concept bank. Higher is better.",
            "- `sig_f1`: macro-F1 of a linear probe predicting the joint node-level concept signature. Higher is better.",
            "- `sig_nmi`: alignment between k-means clusters and node-level concept signatures. Higher is better.",
            "- `sig_ari`: same idea as `sig_nmi`, but corrected for chance. Higher is better.",
            "- `service_order_f1`: can a simple probe predict whether a customer is served early, in the middle, or late?",
            "- `route_role_f1`: can a simple probe predict whether a customer is a route start, middle, end, or singleton?",
            "- `local_cost_f1`: can a simple probe predict the local removal-cost proxy bucket (`low`, `medium`, `high`)?",
            "- `eff_rank` and `stable_rank`: how rich the node embedding matrices are before probing.",
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
            max_probe_nodes=args.max_probe_nodes,
            max_k=args.max_k,
            seed=args.seed,
            output="",
            artifacts_output="",
        )
        report, artifacts = compute_node_probe_bundle(probe_args)
        safe_name = _sanitize_filename(report["config"]["config_repr"])
        report_path = per_config_dir / f"{config_id}_{safe_name}.json"
        artifact_path = per_config_dir / f"{config_id}_{safe_name}.npz"
        report["artifacts"] = {"customer_features_path": str(artifact_path)}
        report["paths"] = {"report_json": str(report_path)}
        write_node_probe_report(report, str(report_path))
        write_node_probe_artifacts(artifacts, str(artifact_path))
        reports.append(report)
        summary = _summary_row(int(config_id), report)
        summary_rows.append(summary)
        print(
            f"done config={config_id} model={summary['model']} "
            f"nmi={_fmt(summary['node_signature_nmi'])} "
            f"order_f1={_fmt(summary['service_order_f1'])} "
            f"cost_f1={_fmt(summary['local_cost_f1'])}"
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
