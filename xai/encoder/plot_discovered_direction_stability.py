from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots for discovered-direction stability across decomposition seeds."
    )
    parser.add_argument(
        "--input-json",
        default="logs/xai/encoder/discovered_direction_stability/comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/xai/encoder/discovered_direction_stability/plots",
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


def _num_or_nan(value: Any) -> float:
    cast = _safe_float(value)
    return float("nan") if cast is None else cast


def _short_name(name: str, kind: str) -> str:
    encoder_map = {"SageEncoder": "SAGE", "AttentionEncoder": "Attention", "MixedScoresEncoder": "MatNet"}
    decoder_map = {"EndToEndDecoder": "E2E", "RecourseDecoder": "Recourse"}
    if kind == "encoder":
        return encoder_map.get(name, name.removesuffix("Encoder"))
    return decoder_map.get(name, name.removesuffix("Decoder"))


def _short_config_label(row: dict[str, Any]) -> str:
    return f"{_short_name(str(row.get('encoder', '')), 'encoder')}/{_short_name(str(row.get('decoder', '')), 'decoder')}"


def _labels(rows: Sequence[dict[str, Any]]) -> List[str]:
    return [_short_config_label(row) for row in rows]


def _plot_alignment_bars(rows: Sequence[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    labels = _labels(rows)
    positions = np.arange(len(labels), dtype=float)
    width = 0.28
    pca = [_num_or_nan(row.get("pca_component_alignment_mean")) for row in rows]
    ica = [_num_or_nan(row.get("ica_component_alignment_mean")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(labels)), 5.0), constrained_layout=True)
    ax.bar(positions - width / 2, pca, width=width, label="PCA")
    ax.bar(positions + width / 2, ica, width=width, label="ICA")
    ax.set_title("Discovered Direction Stability")
    ax.set_ylabel("pairwise alignment")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    path = output_dir / "stability_alignment_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_concept_ratio_bars(rows: Sequence[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    labels = _labels(rows)
    positions = np.arange(len(labels), dtype=float)
    width = 0.28
    pca = [_num_or_nan(row.get("pca_same_top_concept_ratio_mean")) for row in rows]
    ica = [_num_or_nan(row.get("ica_same_top_concept_ratio_mean")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(labels)), 5.0), constrained_layout=True)
    ax.bar(positions - width / 2, pca, width=width, label="PCA")
    ax.bar(positions + width / 2, ica, width=width, label="ICA")
    ax.set_title("Concept Interpretation Stability")
    ax.set_ylabel("same top concept ratio")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    path = output_dir / "stability_same_concept_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_method_scatter(rows: Sequence[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for row in rows:
        x_value = _safe_float(row.get("pca_component_alignment_mean"))
        y_value = _safe_float(row.get("ica_component_alignment_mean"))
        if x_value is None or y_value is None:
            continue
        label = _short_config_label(row)
        ax.scatter([x_value], [y_value], s=70)
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("PCA stability")
    ax.set_ylabel("ICA stability")
    ax.set_title("PCA vs ICA Stability")
    ax.grid(alpha=0.25)

    path = output_dir / "stability_pca_vs_ica_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_readme(output_dir: Path, generated: Sequence[Path]) -> Path:
    lines = [
        "# Discovered Direction Stability Plots",
        "",
        "- `stability_alignment_bars.png`: pairwise component alignment across seeds for PCA and ICA",
        "- `stability_same_concept_bars.png`: stability of top-concept interpretation across seeds",
        "- `stability_pca_vs_ica_scatter.png`: direct config-wise comparison between PCA and ICA stability",
        "",
        "Generated files:",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows", []) or []
    if not rows:
        raise SystemExit(f"No summary rows found in {input_path}")

    generated = [
        _plot_alignment_bars(rows, output_dir, args.dpi),
        _plot_concept_ratio_bars(rows, output_dir, args.dpi),
        _plot_method_scatter(rows, output_dir, args.dpi),
    ]
    generated.append(_write_readme(output_dir, generated))

    print(f"Wrote plots to {output_dir}")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
