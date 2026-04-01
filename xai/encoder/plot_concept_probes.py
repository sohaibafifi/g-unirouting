from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from encoder.concept_bank import CONCEPT_CLASS_ORDERS, CONCEPT_DISPLAY_NAMES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate human-readable plots from encoder concept-probe comparison JSON."
    )
    parser.add_argument(
        "--input-json",
        default="logs/xai/encoder/graph/concepts/comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/xai/encoder/graph/concepts/plots",
    )
    parser.add_argument(
        "--projection-methods",
        nargs="*",
        choices=["pca", "tsne"],
        default=["pca"],
    )
    parser.add_argument("--dpi", type=int, default=180)
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
        "GATEncoder": "GAT",
        "GATv2Encoder": "GATv2",
        "TransformerEncoder": "Transformer",
        "PerformerEncoder": "Performer",
        "MixedScoresEncoder": "MatNet",
        "GPSEncoder": "GPS",
    }
    decoder_map = {
        "EndToEndDecoder": "E2E",
        "RecourseDecoder": "Recourse",
    }
    if kind == "encoder":
        if name in encoder_map:
            return encoder_map[name]
        return name.removesuffix("Encoder")
    if name in decoder_map:
        return decoder_map[name]
    return name.removesuffix("Decoder")


def _short_config_label(row: Dict[str, Any], multiline: bool = False) -> str:
    encoder = _short_name(str(row.get("encoder", "")), "encoder")
    decoder = _short_name(str(row.get("decoder", "")), "decoder")
    sep = "/\n" if multiline else "/"
    return f"{encoder}{sep}{decoder}"


def _model_labels(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [_short_config_label(row) for row in rows]


def _bar_positions(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def _set_common_xlabels(ax: plt.Axes, labels: Sequence[str]) -> None:
    ax.set_xticks(_bar_positions(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")


def _report_by_model(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reports = payload.get("reports", []) or []
    return {str(report["config"]["config_repr"]): report for report in reports}


def _load_probe_artifacts(report: Dict[str, Any]) -> Dict[str, Any]:
    artifact_path = _nested_get(report, ["artifacts", "pooled_features_path"])
    if not artifact_path:
        raise FileNotFoundError(
            "Missing pooled feature artifact path in report. Re-run compare_concept_probes.py "
            "with the updated version to generate .npz sidecars."
        )
    path = Path(str(artifact_path))
    if not path.exists():
        raise FileNotFoundError(f"Probe artifact not found: {path}")

    loaded = np.load(path, allow_pickle=False)
    artifacts: Dict[str, Any] = {
        "pooled_features": loaded["pooled_features"],
        "core_concept_signatures": loaded["core_concept_signatures"].astype(str),
        "concept_states": {},
        "concept_raw_values": {},
    }
    for key in loaded.files:
        if key.startswith("concept_state__"):
            artifacts["concept_states"][key.split("__", 1)[1]] = loaded[key].astype(str)
        elif key.startswith("concept_value__"):
            artifacts["concept_raw_values"][key.split("__", 1)[1]] = loaded[key].astype(np.float32)
    return artifacts


def _annotate_heatmap(ax: plt.Axes, data: np.ndarray) -> None:
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            label = "-" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value >= 0.75 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=color)


def _project_features(features: np.ndarray, method: str) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(scaled)
    if method == "tsne":
        perplexity = min(30, max(5, features.shape[0] // 20))
        return TSNE(
            n_components=2,
            random_state=0,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(scaled)
    raise ValueError(f"Unsupported projection method: {method}")


def _concept_order() -> List[str]:
    return [
        "instance_compactness_state",
        "spatial_clustering_state",
        "outlier_presence_state",
        "load_concentration_state",
        "linehaul_backhaul_balance_state",
        "capacity_pressure_prior_state",
        "tw_density_state",
        "tw_width_profile_state",
        "distance_budget_pressure_state",
        "combined_constraint_tension_state",
    ]


def _concept_display_label(concept_name: str) -> str:
    return CONCEPT_DISPLAY_NAMES.get(concept_name, concept_name)


def _class_order(concept_name: str, classes: Sequence[str]) -> List[str]:
    preferred = CONCEPT_CLASS_ORDERS.get(concept_name, [])
    ordered = [label for label in preferred if label in classes]
    ordered.extend(sorted(label for label in classes if label not in ordered))
    return ordered


def _plot_concept_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    concept_names = _concept_order()
    labels = _model_labels(rows)
    matrix = np.full((len(labels), len(concept_names)), np.nan, dtype=float)

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        for col_idx, concept_name in enumerate(concept_names):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, ["concept_state_separation", concept_name, "linear_probe", "macro_f1"])
            )

    fig, ax = plt.subplots(
        figsize=(max(10.5, 1.25 * len(concept_names)), max(4.5, 1.1 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title("Concept Macro-F1")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_xticklabels(
        [_concept_display_label(name) for name in concept_names],
        rotation=25,
        ha="right",
    )
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)

    path = output_dir / "concept_macro_f1_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_binary_concept_auc_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Optional[Path]:
    concept_names = _concept_order()
    auc_columns = []
    for concept_name in concept_names:
        values = []
        for row in rows:
            report = reports_by_model[str(row["model"])]
            value = _nested_get(
                report,
                ["concept_state_separation", concept_name, "linear_probe", "roc_auc"],
            )
            values.append(_safe_float(value))
        if any(value is not None for value in values):
            auc_columns.append(concept_name)

    if not auc_columns:
        return None

    labels = _model_labels(rows)
    matrix = np.full((len(labels), len(auc_columns)), np.nan, dtype=float)
    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        for col_idx, concept_name in enumerate(auc_columns):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, ["concept_state_separation", concept_name, "linear_probe", "roc_auc"])
            )

    fig, ax = plt.subplots(
        figsize=(max(6.5, 1.4 * len(auc_columns)), max(4.0, 1.1 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title("Binary Concept ROC AUC")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(auc_columns)))
    ax.set_xticklabels(
        [_concept_display_label(name) for name in auc_columns],
        rotation=25,
        ha="right",
    )
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)

    path = output_dir / "binary_concept_auc_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_signature_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    labels = _model_labels(rows)
    x = _bar_positions(len(labels))
    metrics: List[Tuple[str, str, str]] = [
        ("concept_macro_f1_mean", "Concept Mean Macro-F1", "#72b7b2"),
        ("concept_signature_nmi", "Core Concept Signature NMI", "#4c78a8"),
        ("concept_signature_ari", "Core Concept Signature ARI", "#f58518"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (key, title, color) in zip(axes.flat, metrics):
        values = [_num_or_nan(row.get(key)) for row in rows]
        ax.bar(x, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(key)
        _set_common_xlabels(ax, labels)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    path = output_dir / "concept_signature_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_rank_vs_signature(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs = [_num_or_nan(row.get("effective_rank_mean")) for row in rows]
    ys = [_num_or_nan(row.get("concept_signature_nmi")) for row in rows]
    sizes = []
    for row in rows:
        concept_mean = _safe_float(row.get("concept_macro_f1_mean"))
        sizes.append(160 if concept_mean is None else 120 + 180 * concept_mean)

    ax.scatter(xs, ys, s=sizes, c="#4c78a8", alpha=0.8, edgecolors="black", linewidths=0.6)
    for row, x_value, y_value in zip(rows, xs, ys):
        if np.isfinite(x_value) and np.isfinite(y_value):
            ax.annotate(
                _short_config_label(row),
                (x_value, y_value),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
    ax.set_xlabel("Effective rank mean")
    ax.set_ylabel("Core concept signature NMI")
    ax.set_title("Latent Richness vs Concept-Signature Organization")
    ax.grid(alpha=0.25)

    path = output_dir / "rank_vs_concept_signature_nmi.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_clustering_curves(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for row in rows:
        report = reports_by_model[str(row["model"])]
        sweep = report.get("clustering_sweep", []) or []
        ks = [_num_or_nan(item.get("k")) for item in sweep]
        inertias = [_num_or_nan(item.get("inertia")) for item in sweep]
        silhouettes = [_num_or_nan(item.get("silhouette")) for item in sweep]
        axes[0].plot(ks, inertias, marker="o", linewidth=1.8, label=_short_config_label(row))
        axes[1].plot(ks, silhouettes, marker="o", linewidth=1.8, label=_short_config_label(row))

    axes[0].set_title("K-Means Inertia by k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("K-Means Silhouette by k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=9)

    path = output_dir / "concept_clustering_curves.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_signature_histogram(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    if not rows:
        raise ValueError("No rows available for signature histogram")
    report = reports_by_model[str(rows[0]["model"])]
    counts = report["dataset"].get("core_concept_signature_counts", {}) or {}
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    labels = [key for key, _ in items]
    values = [int(value) for _, value in items]

    fig_height = max(5.0, 0.32 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#72b7b2", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Core Concept Signature Frequency in the Probe Dataset")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value}", va="center", fontsize=8)

    path = output_dir / "core_concept_signature_histogram.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_projections(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
    method: str,
) -> Path:
    concept_names = _concept_order()
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=len(concept_names),
        figsize=(4.2 * len(concept_names), max(3.1 * nrows, 3.1)),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        artifacts = _load_probe_artifacts(report)
        coords = _project_features(artifacts["pooled_features"], method=method)

        for col_idx, concept_name in enumerate(concept_names):
            ax = axes[row_idx, col_idx]
            labels = artifacts["concept_states"][concept_name]
            classes = _class_order(concept_name, sorted(set(labels.tolist())))
            cmap = plt.get_cmap("tab10", max(len(classes), 3))
            for class_idx, class_name in enumerate(classes):
                mask = labels == class_name
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=10,
                    alpha=0.65,
                    color=cmap(class_idx),
                    label=class_name if row_idx == 0 else None,
                )
            if row_idx == 0:
                ax.set_title(_concept_display_label(concept_name))
                ax.legend(loc="best", fontsize=6, frameon=False)
            if col_idx == 0:
                ax.set_ylabel(_short_config_label(row, multiline=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.12)

    fig.suptitle(
        f"{method.upper()} projections colored by encoder concept states",
        fontsize=14,
    )
    path = output_dir / f"concept_projections_{method}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_manifest(output_dir: Path, payload: Dict[str, Any], files: Sequence[Path]) -> Path:
    lines = [
        "# Encoder Concept Probe Plots",
        "",
        f"- source_json: `{payload.get('meta', {}).get('source_json', '-')}`",
        f"- num_samples: `{payload.get('meta', {}).get('num_samples', '-')}`",
        f"- pooling: `{payload.get('meta', {}).get('pooling', '-')}`",
        "",
        "Generated files:",
    ]
    for path in files:
        lines.append(f"- `{path.name}`")
    manifest_path = output_dir / "README.md"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload.setdefault("meta", {})
    payload["meta"]["source_json"] = str(input_path)

    rows = payload.get("summary_rows", []) or []
    if not rows:
        raise RuntimeError("No summary_rows found in comparison JSON.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_by_model = _report_by_model(payload)

    generated = [
        _plot_concept_heatmap(rows, reports_by_model, output_dir, args.dpi),
        _plot_signature_bars(rows, output_dir, args.dpi),
        _plot_rank_vs_signature(rows, output_dir, args.dpi),
        _plot_clustering_curves(rows, reports_by_model, output_dir, args.dpi),
        _plot_signature_histogram(rows, reports_by_model, output_dir, args.dpi),
    ]
    binary_auc_heatmap = _plot_binary_concept_auc_heatmap(rows, reports_by_model, output_dir, args.dpi)
    if binary_auc_heatmap is not None:
        generated.append(binary_auc_heatmap)
    for method in args.projection_methods:
        generated.append(_plot_projections(rows, reports_by_model, output_dir, args.dpi, method))
    manifest = _write_manifest(output_dir, payload, generated)

    print(f"Wrote plots to {output_dir}")
    for path in generated:
        print(f"- {path}")
    print(f"- {manifest}")


if __name__ == "__main__":
    main()
