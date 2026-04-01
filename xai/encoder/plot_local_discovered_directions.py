from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cache_root = Path(tempfile.gettempdir()) / "codex-mpl-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _build_parser(level: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Generate plots for discovered latent directions in encoder {level}-level representations."
    )
    parser.add_argument(
        "--input-json",
        default=f"logs/xai/encoder/{level}/discovered_directions/comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default=f"logs/xai/encoder/{level}/discovered_directions/plots",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--top-components", type=int, default=5)
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


def _nested_get(payload: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _num_or_nan(value: Any) -> float:
    cast = _safe_float(value)
    return float("nan") if cast is None else cast


def _short_name(name: str, kind: str) -> str:
    encoder_map = {
        "SageEncoder": "SAGE",
        "AttentionEncoder": "Attention",
        "MixedScoresEncoder": "MatNet",
        "TransformerEncoder": "Transformer",
    }
    decoder_map = {
        "EndToEndDecoder": "E2E",
        "RecourseDecoder": "Recourse",
    }
    if kind == "encoder":
        return encoder_map.get(name, name.removesuffix("Encoder"))
    return decoder_map.get(name, name.removesuffix("Decoder"))


def _short_config_label(row: Dict[str, Any]) -> str:
    return f"{_short_name(str(row.get('encoder', '')), 'encoder')}/{_short_name(str(row.get('decoder', '')), 'decoder')}"


def _model_labels(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [_short_config_label(row) for row in rows]


def _report_by_model(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reports = payload.get("reports", []) or []
    return {str(report["config"]["config_repr"]): report for report in reports}


def _load_artifacts(report: Dict[str, Any]) -> Dict[str, Any]:
    artifact_path = _nested_get(report, ["artifacts", "discovered_directions_path"])
    if not artifact_path:
        raise FileNotFoundError(
            "Missing discovered-direction artifact path in report. Re-run the compare script."
        )
    loaded = np.load(Path(str(artifact_path)), allow_pickle=False)
    concept_raw_values: Dict[str, Any] = {}
    for key in loaded.files:
        if key.startswith("concept_value__"):
            concept_raw_values[key.split("__", 1)[1]] = loaded[key].astype(np.float32)
    return {
        "component_scores": loaded["component_scores"].astype(np.float32),
        "explained_variance_ratio": loaded["explained_variance_ratio"].astype(np.float32),
        "cumulative_explained_variance_ratio": loaded["cumulative_explained_variance_ratio"].astype(
            np.float32
        ),
        "correlation_matrix": loaded["correlation_matrix"].astype(np.float32),
        "ica_component_scores": loaded["ica_component_scores"].astype(np.float32),
        "ica_explained_variance_ratio": loaded["ica_explained_variance_ratio"].astype(np.float32),
        "ica_cumulative_explained_variance_ratio": loaded[
            "ica_cumulative_explained_variance_ratio"
        ].astype(np.float32),
        "ica_correlation_matrix": loaded["ica_correlation_matrix"].astype(np.float32),
        "concept_names": loaded["concept_names"].astype(str),
        "concept_raw_values": concept_raw_values,
    }


def _concept_names_and_labels(rows: Sequence[Dict[str, Any]], reports_by_model: Dict[str, Dict[str, Any]]) -> tuple[List[str], List[str]]:
    if not rows:
        return [], []
    reference_report = reports_by_model[str(rows[0]["model"])]
    artifacts = _load_artifacts(reference_report)
    concept_names = artifacts["concept_names"].astype(str).tolist()
    display_names = []
    for name in concept_names:
        display_names.append(
            str(
                _nested_get(reference_report, ["discovered_directions", "concept_alignment", name, "display_name"])
                or name
            )
        )
    return concept_names, display_names


def _annotate_heatmap(ax: plt.Axes, data: np.ndarray) -> None:
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            label = "-" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value >= 0.6 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=color)


def _plot_best_concept_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
    level: str,
    method: str,
) -> Path:
    concept_names, display_names = _concept_names_and_labels(rows, reports_by_model)
    labels = _model_labels(rows)
    matrix = np.full((len(labels), len(concept_names)), np.nan, dtype=float)

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        base_path = ["discovered_directions", "concept_alignment"]
        if method == "ica":
            base_path = ["alternative_methods", "ica", "concept_alignment"]
        for col_idx, concept_name in enumerate(concept_names):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, [*base_path, concept_name, "best_abs_correlation"])
            )

    fig, ax = plt.subplots(
        figsize=(max(10.5, 0.8 * max(len(concept_names), 1)), max(4.5, 1.1 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title(f"{level.capitalize()} Best Absolute Correlation by Concept ({method.upper()})")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_xticklabels(display_names, rotation=25, ha="right")
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    filename = "discovered_best_corr_heatmap.png" if method == "pca" else f"discovered_best_corr_heatmap_{method}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int, level: str) -> Path:
    labels = _model_labels(rows)
    positions = np.arange(len(labels), dtype=float)
    width = 0.22
    mean_corr = [_num_or_nan(row.get("mean_best_abs_correlation_top_components")) for row in rows]
    best_corr = [_num_or_nan(row.get("best_abs_correlation_overall")) for row in rows]
    top3_evr = [_num_or_nan(row.get("top3_cumulative_explained_variance_ratio")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.7 * len(labels)), 5.2), constrained_layout=True)
    ax.bar(positions - width, mean_corr, width=width, label="Mean |corr|")
    ax.bar(positions, best_corr, width=width, label="Best |corr|")
    ax.bar(positions + width, top3_evr, width=width, label="Top-3 EVR")
    ax.set_title(f"{level.capitalize()} Discovered Direction Summary (PCA)")
    ax.set_ylabel("score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    path = output_dir / "discovered_summary_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_method_comparison_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int, level: str) -> Path:
    labels = _model_labels(rows)
    positions = np.arange(len(labels), dtype=float)
    width = 0.18
    pca_mean = [_num_or_nan(row.get("mean_best_abs_correlation_top_components")) for row in rows]
    ica_mean = [_num_or_nan(row.get("ica_mean_best_abs_correlation_top_components")) for row in rows]
    pca_best = [_num_or_nan(row.get("best_abs_correlation_overall")) for row in rows]
    ica_best = [_num_or_nan(row.get("ica_best_abs_correlation_overall")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.8 * len(labels)), 5.4), constrained_layout=True)
    ax.bar(positions - 1.5 * width, pca_mean, width=width, label="PCA mean |corr|")
    ax.bar(positions - 0.5 * width, ica_mean, width=width, label="ICA mean |corr|")
    ax.bar(positions + 0.5 * width, pca_best, width=width, label="PCA best |corr|")
    ax.bar(positions + 1.5 * width, ica_best, width=width, label="ICA best |corr|")
    ax.set_title(f"{level.capitalize()} PCA vs ICA Alignment Summary")
    ax.set_ylabel("score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, ncols=2)

    path = output_dir / "discovered_method_comparison_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_alignment_scatter(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int, level: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    for row in rows:
        x_value = _safe_float(row.get("top3_cumulative_explained_variance_ratio"))
        y_value = _safe_float(row.get("mean_best_abs_correlation_top_components"))
        if x_value is None or y_value is None:
            continue
        label = _short_config_label(row)
        ax.scatter([x_value], [y_value], s=70)
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Top-3 cumulative explained variance")
    ax.set_ylabel("Mean best |corr|")
    ax.set_title(f"{level.capitalize()} Variance Concentration vs Direction Interpretability")
    ax.grid(alpha=0.25)

    path = output_dir / "discovered_alignment_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_component_heatmaps(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
    top_components: int,
    level: str,
    method: str,
) -> Path:
    n_panels = len(rows)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(12.0, 6.2 * ncols), max(3.6 * nrows, 4.0)),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.reshape(nrows, ncols)
    last_im = None

    for panel_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        artifacts = _load_artifacts(report)
        corr_key = "correlation_matrix" if method == "pca" else f"{method}_correlation_matrix"
        corr = np.asarray(artifacts[corr_key], dtype=np.float64)
        concept_names = artifacts["concept_names"].astype(str).tolist()
        n_use = min(top_components, corr.shape[0])
        corr = corr[:n_use]
        ax = axes.flat[panel_idx]
        last_im = ax.imshow(corr, aspect="auto", cmap=plt.cm.RdBu_r, vmin=-1.0, vmax=1.0)
        ax.set_title(_short_config_label(row))
        ax.set_yticks(np.arange(n_use))
        prefix = "PC" if method == "pca" else method.upper()
        ax.set_yticklabels([f"{prefix}{i+1}" for i in range(n_use)])
        ax.set_xticks(np.arange(len(concept_names)))
        ax.set_xticklabels(
            [
                str(
                    _nested_get(
                        report,
                        (
                            ["discovered_directions", "concept_alignment", name, "display_name"]
                            if method == "pca"
                            else ["alternative_methods", method, "concept_alignment", name, "display_name"]
                        ),
                    )
                    or name
                )
                for name in concept_names
            ],
            rotation=25,
            ha="right",
            fontsize=8,
        )

    for panel_idx in range(len(rows), nrows * ncols):
        axes.flat[panel_idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)

    filename = "discovered_component_heatmaps.png" if method == "pca" else f"discovered_component_heatmaps_{method}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_readme(output_dir: Path, generated: Sequence[Path], level: str) -> Path:
    lines = [
        f"# {level.capitalize()} Discovered Directions Plots",
        "",
        f"These figures summarize the unsupervised {level}-level latent analysis:",
        "- `discovered_best_corr_heatmap.png`: PCA best absolute correlation by concept",
        "- `discovered_best_corr_heatmap_ica.png`: ICA best absolute correlation by concept",
        "- `discovered_summary_bars.png`: PCA summary bars for alignment and explained variance concentration",
        "- `discovered_method_comparison_bars.png`: direct PCA vs ICA comparison on alignment metrics",
        "- `discovered_alignment_scatter.png`: tradeoff between variance concentration and average interpretability",
        "- `discovered_component_heatmaps.png`: signed PCA component correlations with the concept bank",
        "- `discovered_component_heatmaps_ica.png`: signed ICA component correlations with the concept bank",
        "",
        "Generated files:",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main_for_level(level: str) -> None:
    args = _build_parser(level).parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows", []) or []
    reports_by_model = _report_by_model(payload)
    if not rows:
        raise SystemExit(f"No summary rows found in {input_path}")

    generated = [
        _plot_best_concept_heatmap(rows, reports_by_model, output_dir, args.dpi, level, method="pca"),
        _plot_best_concept_heatmap(rows, reports_by_model, output_dir, args.dpi, level, method="ica"),
        _plot_summary_bars(rows, output_dir, args.dpi, level),
        _plot_method_comparison_bars(rows, output_dir, args.dpi, level),
        _plot_alignment_scatter(rows, output_dir, args.dpi, level),
        _plot_component_heatmaps(rows, reports_by_model, output_dir, args.dpi, args.top_components, level, method="pca"),
        _plot_component_heatmaps(rows, reports_by_model, output_dir, args.dpi, args.top_components, level, method="ica"),
    ]
    generated.append(_write_readme(output_dir, generated, level))

    print(f"Wrote plots to {output_dir}")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit("Use plot_node_discovered_directions.py or plot_edge_discovered_directions.py")
