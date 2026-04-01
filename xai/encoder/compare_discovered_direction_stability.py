from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
from rich.console import Console
from rich.table import Table

from encoder.concept_probe import compute_concept_probe_bundle
from encoder.discovered_directions import _fit_ica, _fit_pca, _pearson_corr, _summarize_method
from encoder.encoder_probe import _json_ready
from mavrp.configs.config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the stability of discovered latent directions across decomposition seeds "
            "for PCA and ICA, using the same sampled instances."
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
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument(
        "--sort-by",
        choices=[
            "pca_component_alignment_mean",
            "ica_component_alignment_mean",
            "pca_same_top_concept_ratio_mean",
            "ica_same_top_concept_ratio_mean",
            "pca_mean_abs_corr_mean",
            "ica_mean_abs_corr_mean",
        ],
        default="ica_component_alignment_mean",
    )
    parser.add_argument(
        "--output-json",
        default="logs/xai/encoder/discovered_direction_stability/comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/xai/encoder/discovered_direction_stability/comparison.md",
    )
    parser.add_argument(
        "--per-config-dir",
        default="logs/xai/encoder/discovered_direction_stability/runs",
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


def _match_components_by_score_correlation(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    top_components: int,
) -> Tuple[List[int], np.ndarray]:
    k = min(int(top_components), int(scores_a.shape[1]), int(scores_b.shape[1]))
    if k <= 0:
        return [], np.zeros((0, 0), dtype=np.float64)

    corr = np.full((k, k), np.nan, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            corr[i, j] = abs(_pearson_corr(scores_a[:, i], scores_b[:, j]))

    indices = list(range(k))
    best_perm = indices
    best_score = float("-inf")
    for perm in itertools.permutations(indices):
        score = 0.0
        valid = True
        for i, j in enumerate(perm):
            value = corr[i, j]
            if not np.isfinite(value):
                valid = False
                break
            score += float(value)
        if valid and score > best_score:
            best_score = score
            best_perm = list(perm)
    return best_perm, corr


def _pairwise_method_stability(
    method_runs: Sequence[Dict[str, Any]],
    top_components: int,
) -> Dict[str, Any]:
    pair_rows: List[Dict[str, Any]] = []
    if len(method_runs) < 2:
        return {
            "num_runs": len(method_runs),
            "num_pairs": 0,
            "pairwise_component_alignment_mean": None,
            "pairwise_component_alignment_std": None,
            "pairwise_min_component_alignment_mean": None,
            "pairwise_same_top_concept_ratio_mean": None,
            "pairwise_same_top_concept_ratio_std": None,
            "mean_abs_corr_mean": None,
            "mean_abs_corr_std": None,
            "best_abs_corr_mean": None,
            "best_abs_corr_std": None,
            "pair_rows": [],
        }

    mean_abs_corr_values = []
    best_abs_corr_values = []
    for run in method_runs:
        mean_abs = _nested_get(run, ["payload", "summary", "mean_best_abs_correlation_top_components"])
        best_abs = _nested_get(run, ["payload", "summary", "best_abs_correlation_overall"])
        if _safe_float(mean_abs) is not None:
            mean_abs_corr_values.append(float(mean_abs))
        if _safe_float(best_abs) is not None:
            best_abs_corr_values.append(float(best_abs))

    for idx_a in range(len(method_runs)):
        for idx_b in range(idx_a + 1, len(method_runs)):
            run_a = method_runs[idx_a]
            run_b = method_runs[idx_b]
            permutation, corr = _match_components_by_score_correlation(
                scores_a=np.asarray(run_a["scores"], dtype=np.float64),
                scores_b=np.asarray(run_b["scores"], dtype=np.float64),
                top_components=top_components,
            )
            if not permutation:
                continue
            matched_values = [float(corr[i, permutation[i]]) for i in range(len(permutation))]
            top_concepts_a = [
                str(item.get("best_concept_name", ""))
                for item in (run_a["payload"].get("top_components") or [])[: len(permutation)]
            ]
            top_concepts_b = [
                str(item.get("best_concept_name", ""))
                for item in (run_b["payload"].get("top_components") or [])
            ]
            matches = []
            for i, j in enumerate(permutation):
                concept_b = top_concepts_b[j] if j < len(top_concepts_b) else ""
                matches.append(1.0 if top_concepts_a[i] == concept_b and top_concepts_a[i] else 0.0)
            pair_rows.append(
                {
                    "seed_a": int(run_a["seed"]),
                    "seed_b": int(run_b["seed"]),
                    "mean_component_alignment": float(np.mean(matched_values)),
                    "min_component_alignment": float(np.min(matched_values)),
                    "same_top_concept_ratio": float(np.mean(matches)),
                }
            )

    def _mean_std(key: str) -> Tuple[Optional[float], Optional[float]]:
        values = [float(row[key]) for row in pair_rows if _safe_float(row.get(key)) is not None]
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    align_mean, align_std = _mean_std("mean_component_alignment")
    same_mean, same_std = _mean_std("same_top_concept_ratio")
    min_align_mean, _ = _mean_std("min_component_alignment")

    return {
        "num_runs": len(method_runs),
        "num_pairs": len(pair_rows),
        "pairwise_component_alignment_mean": align_mean,
        "pairwise_component_alignment_std": align_std,
        "pairwise_min_component_alignment_mean": min_align_mean,
        "pairwise_same_top_concept_ratio_mean": same_mean,
        "pairwise_same_top_concept_ratio_std": same_std,
        "mean_abs_corr_mean": float(np.mean(mean_abs_corr_values)) if mean_abs_corr_values else None,
        "mean_abs_corr_std": float(np.std(mean_abs_corr_values)) if mean_abs_corr_values else None,
        "best_abs_corr_mean": float(np.mean(best_abs_corr_values)) if best_abs_corr_values else None,
        "best_abs_corr_std": float(np.std(best_abs_corr_values)) if best_abs_corr_values else None,
        "pair_rows": pair_rows,
    }


def _run_single_method(
    method: str,
    features: np.ndarray,
    concept_names: List[str],
    concept_raw_values: Dict[str, np.ndarray],
    decomp_seed: int,
    num_components: int,
    top_components: int,
) -> Dict[str, Any]:
    if method == "pca":
        model, scores = _fit_pca(features, seed=decomp_seed, num_components=num_components)
        payload, _ = _summarize_method(
            method_name="pca",
            scores=scores,
            concept_names=concept_names,
            concept_raw_values=concept_raw_values,
            top_components=min(top_components, scores.shape[1]),
            component_vectors=np.asarray(model.components_, dtype=np.float32),
            explained_variance_ratio=np.asarray(model.explained_variance_ratio_, dtype=np.float64),
        )
    elif method == "ica":
        model, scores = _fit_ica(features, seed=decomp_seed, num_components=num_components)
        payload, _ = _summarize_method(
            method_name="ica",
            scores=scores,
            concept_names=concept_names,
            concept_raw_values=concept_raw_values,
            top_components=min(top_components, scores.shape[1]),
            component_vectors=np.asarray(model.components_, dtype=np.float32),
            explained_variance_ratio=None,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return {"seed": int(decomp_seed), "payload": payload, "scores": np.asarray(scores, dtype=np.float32)}


def compute_discovered_direction_stability_bundle(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    concept_args = SimpleNamespace(
        config_id=args.config_id,
        checkpoint=None,
        device=args.device,
        graph_size=args.graph_size,
        problem=args.problem,
        num_samples=args.num_samples,
        pooling=args.pooling,
        max_k=12,
        seed=args.data_seed,
        output=None,
        artifacts_output=None,
    )
    concept_report, concept_artifacts = compute_concept_probe_bundle(concept_args)
    features = np.asarray(concept_artifacts["pooled_features"], dtype=np.float32)
    concept_raw_values = {
        str(name): np.asarray(values, dtype=np.float32)
        for name, values in (concept_artifacts.get("concept_raw_values") or {}).items()
    }
    concept_names = sorted(concept_raw_values.keys())
    decomp_seeds = [int(args.seed_start) + idx for idx in range(int(args.num_runs))]

    method_runs: Dict[str, List[Dict[str, Any]]] = {"pca": [], "ica": []}
    for decomp_seed in decomp_seeds:
        method_runs["pca"].append(
            _run_single_method(
                "pca",
                features,
                concept_names,
                concept_raw_values,
                decomp_seed,
                int(args.num_components),
                int(args.top_components),
            )
        )
        method_runs["ica"].append(
            _run_single_method(
                "ica",
                features,
                concept_names,
                concept_raw_values,
                decomp_seed,
                int(args.num_components),
                int(args.top_components),
            )
        )

    stability = {
        "pca": _pairwise_method_stability(method_runs["pca"], int(args.top_components)),
        "ica": _pairwise_method_stability(method_runs["ica"], int(args.top_components)),
    }

    return {
        "config": {
            "problem": str(concept_report["config"]["problem"]),
            "graph_size": int(concept_report["config"]["graph_size"]),
            "encoder_name": str(concept_report["config"]["encoder_name"]),
            "decoder_name": str(concept_report["config"]["decoder_name"]),
            "config_repr": str(concept_report["config"]["config_repr"]),
            "checkpoint_path": str(concept_report["config"]["checkpoint_path"]),
            "pooling": str(args.pooling),
            "num_samples": int(args.num_samples),
            "data_seed": int(args.data_seed),
            "seed_start": int(args.seed_start),
            "num_runs": int(args.num_runs),
            "num_components": int(args.num_components),
            "top_components": int(args.top_components),
        },
        "dataset": dict(concept_report["dataset"]),
        "matrix_richness": dict(concept_report["matrix_richness"]),
        "decomposition_seeds": decomp_seeds,
        "stability": stability,
        "runs": {
            "pca": [{"seed": run["seed"], "payload": run["payload"]} for run in method_runs["pca"]],
            "ica": [{"seed": run["seed"], "payload": run["payload"]} for run in method_runs["ica"]],
        },
    }


def write_stability_report(report: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _summary_row(config_id: int, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config_id": config_id,
        "model": report["config"]["config_repr"],
        "encoder": report["config"]["encoder_name"],
        "decoder": report["config"]["decoder_name"],
        "pca_component_alignment_mean": _nested_get(
            report, ["stability", "pca", "pairwise_component_alignment_mean"]
        ),
        "pca_component_alignment_std": _nested_get(
            report, ["stability", "pca", "pairwise_component_alignment_std"]
        ),
        "pca_same_top_concept_ratio_mean": _nested_get(
            report, ["stability", "pca", "pairwise_same_top_concept_ratio_mean"]
        ),
        "pca_mean_abs_corr_mean": _nested_get(report, ["stability", "pca", "mean_abs_corr_mean"]),
        "ica_component_alignment_mean": _nested_get(
            report, ["stability", "ica", "pairwise_component_alignment_mean"]
        ),
        "ica_component_alignment_std": _nested_get(
            report, ["stability", "ica", "pairwise_component_alignment_std"]
        ),
        "ica_same_top_concept_ratio_mean": _nested_get(
            report, ["stability", "ica", "pairwise_same_top_concept_ratio_mean"]
        ),
        "ica_mean_abs_corr_mean": _nested_get(report, ["stability", "ica", "mean_abs_corr_mean"]),
        "effective_rank_mean": _nested_get(report, ["matrix_richness", "effective_rank_mean"]),
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
        title=f"Discovered Direction Stability (sorted by {sort_by})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("id", justify="right")
    table.add_column("model", style="bold", overflow="fold", max_width=48)
    table.add_column("pca", overflow="fold")
    table.add_column("ica", overflow="fold")
    table.add_column("richness", overflow="fold")

    for row in rows:
        table.add_row(
            str(row["config_id"]),
            str(row["model"]),
            "\n".join(
                [
                    f"align={_fmt(row['pca_component_alignment_mean'])}",
                    f"same concept={_fmt(row['pca_same_top_concept_ratio_mean'])}",
                    f"mean_abs={_fmt(row['pca_mean_abs_corr_mean'])}",
                ]
            ),
            "\n".join(
                [
                    f"align={_fmt(row['ica_component_alignment_mean'])}",
                    f"same concept={_fmt(row['ica_same_top_concept_ratio_mean'])}",
                    f"mean_abs={_fmt(row['ica_mean_abs_corr_mean'])}",
                ]
            ),
            f"er={_fmt(row['effective_rank_mean'])}",
        )
    Console().print(table)


def _render_markdown(rows: List[Dict[str, Any]], sort_by: str, args: argparse.Namespace) -> str:
    headers = [
        "id",
        "model",
        "pca_align",
        "ica_align",
        "pca_same_concept",
        "ica_same_concept",
        "pca_mean_abs",
        "ica_mean_abs",
        "eff_rank",
    ]
    lines = [
        "# Discovered Direction Stability",
        "",
        "## Setup",
        "",
        f"- `num_samples`: `{args.num_samples}`",
        f"- `pooling`: `{args.pooling}`",
        f"- `data_seed`: `{args.data_seed}`",
        f"- `seed_start`: `{args.seed_start}`",
        f"- `num_runs`: `{args.num_runs}`",
        f"- `num_components`: `{args.num_components}`",
        f"- `top_components`: `{args.top_components}`",
        f"- `sort_by`: `{sort_by}`",
        "",
        "## Metrics Used",
        "",
        "- `pca_align` / `ica_align`: moyenne des corrélations absolues après appariement optimal des composantes entre seeds. Plus c'est haut, plus les axes reviennent d'un run à l'autre.",
        "- `pca_same_concept` / `ica_same_concept`: proportion moyenne de composantes appariées qui gardent le même meilleur concept. Plus c'est haut, plus l'interprétation est stable.",
        "- `pca_mean_abs` / `ica_mean_abs`: qualité moyenne d'alignement avec le concept bank, déjà vue dans l'analyse `discovered_directions`.",
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
                    _fmt(row["pca_component_alignment_mean"]),
                    _fmt(row["ica_component_alignment_mean"]),
                    _fmt(row["pca_same_top_concept_ratio_mean"]),
                    _fmt(row["ica_same_top_concept_ratio_mean"]),
                    _fmt(row["pca_mean_abs_corr_mean"]),
                    _fmt(row["ica_mean_abs_corr_mean"]),
                    _fmt(row["effective_rank_mean"]),
                ]
            )
            + " |"
        )

    best_pca = _best_row(rows, "pca_component_alignment_mean")
    best_ica = _best_row(rows, "ica_component_alignment_mean")
    lines.extend(
        [
            "",
            "## Automatic Interpretation",
            "",
            (
                "- Most stable PCA directions: "
                f"`{best_pca['model']}` with `pca_align={_fmt(best_pca['pca_component_alignment_mean'])}`."
                if best_pca is not None
                else "- Most stable PCA directions: not available."
            ),
            (
                "- Most stable ICA directions: "
                f"`{best_ica['model']}` with `ica_align={_fmt(best_ica['ica_component_alignment_mean'])}`."
                if best_ica is not None
                else "- Most stable ICA directions: not available."
            ),
            "",
            "## Decision Notes",
            "",
            "- If `align` is high but `same_concept` is low, the geometry of the axes is stable but their interpretation still drifts.",
            "- If both are high, the discovered directions are good candidates for learned concepts worth validating by intervention.",
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
            data_seed=args.data_seed,
            seed_start=args.seed_start,
            num_runs=args.num_runs,
            num_components=args.num_components,
            top_components=args.top_components,
        )
        report = compute_discovered_direction_stability_bundle(run_args)
        filename = _sanitize_filename(f"{config_id}_{report['config']['config_repr']}.json")
        write_stability_report(report, str(per_config_dir / filename))
        all_reports.append(report)
        row = _summary_row(config_id, report)
        rows.append(row)
        console.print(
            f"[cyan]done[/cyan] config={config_id} model={report['config']['config_repr']} "
            f"pca_align={_fmt(row['pca_component_alignment_mean'])} "
            f"ica_align={_fmt(row['ica_component_alignment_mean'])}"
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
            "data_seed": args.data_seed,
            "seed_start": args.seed_start,
            "num_runs": args.num_runs,
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
