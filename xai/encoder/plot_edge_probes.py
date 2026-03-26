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

from encoder.edge_concept_bank import EDGE_CONCEPT_CLASS_ORDERS, EDGE_CONCEPT_DISPLAY_NAMES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate human-readable plots from encoder edge-probe comparison JSON."
    )
    parser.add_argument("--input-json", default="logs/xai/encoder/edge_probes/comparison.json")
    parser.add_argument("--output-dir", default="logs/xai/encoder/edge_probes/plots")
    parser.add_argument(
        "--projection-methods",
        nargs="*",
        choices=["pca", "tsne"],
        default=["pca"],
    )
    parser.add_argument("--max-points-per-panel", type=int, default=1200)
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


def _report_by_model(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reports = payload.get("reports", []) or []
    return {str(report["config"]["config_repr"]): report for report in reports}


def _load_probe_artifacts(report: Dict[str, Any]) -> Dict[str, Any]:
    artifact_path = _nested_get(report, ["artifacts", "edge_features_path"])
    if not artifact_path:
        raise FileNotFoundError(
            "Missing edge feature artifact path in report. Re-run compare_edge_probes.py "
            "with the updated version to generate .npz sidecars."
        )
    path = Path(str(artifact_path))
    if not path.exists():
        raise FileNotFoundError(f"Edge probe artifact not found: {path}")

    loaded = np.load(path, allow_pickle=False)
    artifacts: Dict[str, Any] = {
        "edge_features": loaded["edge_features"],
        "probe_core_edge_signatures": loaded["probe_core_edge_signatures"].astype(str),
        "edge_concept_states": {},
    }
    for key in loaded.files:
        if key.startswith("edge_concept_state__"):
            artifacts["edge_concept_states"][key.split("__", 1)[1]] = loaded[key].astype(str)
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
        perplexity = min(30, max(5, features.shape[0] // 25))
        return TSNE(
            n_components=2,
            random_state=0,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(scaled)
    raise ValueError(f"Unsupported projection method: {method}")


def _edge_concept_order() -> List[str]:
    return [
        "same_route_state",
        "edge_in_solution_state",
        "local_edge_cost_contribution_state",
    ]


def _edge_concept_display_label(concept_name: str) -> str:
    return EDGE_CONCEPT_DISPLAY_NAMES.get(concept_name, concept_name)


def _class_order(concept_name: str, classes: Sequence[str]) -> List[str]:
    preferred = EDGE_CONCEPT_CLASS_ORDERS.get(concept_name, [])
    ordered = [label for label in preferred if label in classes]
    ordered.extend(sorted(label for label in classes if label not in ordered))
    return ordered


def _plot_edge_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    concept_names = _edge_concept_order()
    labels = _model_labels(rows)
    matrix = np.full((len(labels), len(concept_names)), np.nan, dtype=float)

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        for col_idx, concept_name in enumerate(concept_names):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, ["edge_concept_state_separation", concept_name, "linear_probe", "macro_f1"])
            )

    fig, ax = plt.subplots(
        figsize=(8.5, max(4.2, 1.0 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title("Edge Concept Macro-F1")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_xticklabels(
        [_edge_concept_display_label(name) for name in concept_names],
        rotation=20,
        ha="right",
    )
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    path = output_dir / "edge_concept_macro_f1_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_binary_auc_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    concept_names = ["same_route_state", "edge_in_solution_state"]
    labels = _model_labels(rows)
    matrix = np.full((len(labels), len(concept_names)), np.nan, dtype=float)

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        for col_idx, concept_name in enumerate(concept_names):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, ["edge_concept_state_separation", concept_name, "linear_probe", "roc_auc"])
            )

    fig, ax = plt.subplots(
        figsize=(7.0, max(4.2, 1.0 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title("Binary Edge Concept ROC AUC")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_xticklabels(
        [_edge_concept_display_label(name) for name in concept_names],
        rotation=20,
        ha="right",
    )
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    path = output_dir / "binary_edge_concept_auc_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_signature_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    labels = _model_labels(rows)
    x = np.arange(len(labels), dtype=float)
    metrics: List[Tuple[str, str, str]] = [
        ("edge_macro_f1_mean", "Edge Mean Macro-F1", "#72b7b2"),
        ("edge_signature_nmi", "Edge Signature NMI", "#4c78a8"),
        ("edge_signature_ari", "Edge Signature ARI", "#f58518"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (key, title, color) in zip(axes.flat, metrics):
        values = [_num_or_nan(row.get(key)) for row in rows]
        ax.bar(x, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    path = output_dir / "edge_signature_bars.png"
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
    ys = [_num_or_nan(row.get("edge_signature_nmi")) for row in rows]
    sizes = []
    for row in rows:
        edge_mean = _safe_float(row.get("edge_macro_f1_mean"))
        sizes.append(160 if edge_mean is None else 120 + 180 * edge_mean)

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
    ax.set_ylabel("Edge concept signature NMI")
    ax.set_title("Latent Richness vs Edge-Concept Organization")
    ax.grid(alpha=0.25)

    path = output_dir / "rank_vs_edge_signature_nmi.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _subsample_points(
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= max_points or max_points <= 0:
        return features, labels

    per_class_limit = max(1, max_points // max(len(set(labels.tolist())), 1))
    keep_indices: List[int] = []
    rng = np.random.default_rng(0)
    for class_name in sorted(set(labels.tolist())):
        class_indices = np.flatnonzero(labels == class_name)
        if class_indices.size <= per_class_limit:
            keep_indices.extend(class_indices.tolist())
            continue
        sampled = rng.choice(class_indices, size=per_class_limit, replace=False)
        keep_indices.extend(sampled.tolist())
    keep = np.sort(np.asarray(keep_indices, dtype=np.int64))
    return features[keep], labels[keep]


def _plot_projection_grid(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    method: str,
    max_points_per_panel: int,
    dpi: int,
) -> Path:
    concept_names = _edge_concept_order()
    fig, axes = plt.subplots(
        len(rows),
        len(concept_names),
        figsize=(4.0 * len(concept_names), 3.2 * len(rows) + 0.8),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        artifacts = _load_probe_artifacts(report)
        features = artifacts["edge_features"]
        for col_idx, concept_name in enumerate(concept_names):
            ax = axes[row_idx, col_idx]
            labels = artifacts["edge_concept_states"][concept_name]
            sampled_features, sampled_labels = _subsample_points(
                features,
                labels,
                max_points=max_points_per_panel,
            )
            coords = _project_features(sampled_features, method=method)
            classes = _class_order(concept_name, sorted(set(sampled_labels.tolist())))
            cmap = plt.cm.get_cmap("tab10", len(classes))
            for class_idx, class_name in enumerate(classes):
                mask = sampled_labels == class_name
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=9,
                    alpha=0.55,
                    color=cmap(class_idx),
                    label=class_name,
                    linewidths=0.0,
                )
            if row_idx == 0:
                ax.set_title(_edge_concept_display_label(concept_name))
            if col_idx == 0:
                ax.set_ylabel(_short_config_label(row, multiline=True))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0 and col_idx == len(concept_names) - 1:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                    fontsize=8,
                )

    path = output_dir / f"edge_concept_projections_{method}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_readme(output_dir: Path, generated: Sequence[Path]) -> Path:
    lines = [
        "# Edge Probe Plots",
        "",
        "Generated figures:",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    readme = output_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    args = _build_parser().parse_args()
    input_json = Path(args.input_json)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) or []
    reports_by_model = _report_by_model(payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    generated.append(_plot_edge_heatmap(rows, reports_by_model, output_dir, args.dpi))
    generated.append(_plot_binary_auc_heatmap(rows, reports_by_model, output_dir, args.dpi))
    generated.append(_plot_signature_bars(rows, output_dir, args.dpi))
    generated.append(_plot_rank_vs_signature(rows, output_dir, args.dpi))
    for method in args.projection_methods:
        generated.append(
            _plot_projection_grid(
                rows,
                reports_by_model,
                output_dir,
                method=method,
                max_points_per_panel=args.max_points_per_panel,
                dpi=args.dpi,
            )
        )
    generated.append(_write_readme(output_dir, generated))

    print(f"Wrote plots to {output_dir}")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
