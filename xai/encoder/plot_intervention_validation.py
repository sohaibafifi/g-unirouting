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

from encoder.intervention_validation import INTERVENTIONS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots for intervention-based validation of discovered directions."
    )
    parser.add_argument(
        "--input-json",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/plots",
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


def _nested_get(payload: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _short_name(name: str, kind: str) -> str:
    encoder_map = {"SageEncoder": "SAGE", "AttentionEncoder": "Attention", "MixedScoresEncoder": "MatNet"}
    decoder_map = {"EndToEndDecoder": "E2E", "RecourseDecoder": "Recourse"}
    if kind == "encoder":
        return encoder_map.get(name, name.removesuffix("Encoder"))
    return decoder_map.get(name, name.removesuffix("Decoder"))


def _short_config_label(row: Dict[str, Any]) -> str:
    return f"{_short_name(str(row.get('encoder', '')), 'encoder')}/{_short_name(str(row.get('decoder', '')), 'decoder')}"


def _model_labels(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [_short_config_label(row) for row in rows]


def _report_by_model(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(report["config"]["config_repr"]): report for report in (payload.get("reports") or [])}


def _annotate_heatmap(ax: plt.Axes, data: np.ndarray) -> None:
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            label = "-" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value >= 0.75 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=color)


def _plot_heatmap(
    rows: Sequence[Dict[str, Any]],
    reports_by_model: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
    key_path_suffix: List[str],
    title: str,
    filename: str,
) -> Path:
    intervention_names = [str(spec["name"]) for spec in INTERVENTIONS]
    matrix = np.full((len(rows), len(intervention_names)), np.nan, dtype=float)
    for row_idx, row in enumerate(rows):
        report = reports_by_model[str(row["model"])]
        for col_idx, intervention_name in enumerate(intervention_names):
            matrix[row_idx, col_idx] = _num_or_nan(
                _nested_get(report, ["interventions", intervention_name, *key_path_suffix])
            )

    labels = _model_labels(rows)
    fig, ax = plt.subplots(
        figsize=(max(9.0, 1.4 * len(intervention_names)), max(4.5, 1.0 * len(labels) + 2)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(intervention_names)))
    ax.set_xticklabels(intervention_names, rotation=25, ha="right")
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    labels = _model_labels(rows)
    positions = np.arange(len(labels), dtype=float)
    width = 0.24
    concept = [_num_or_nan(row.get("concept_success_mean")) for row in rows]
    pca = [_num_or_nan(row.get("pca_mean_directional_success")) for row in rows]
    ica = [_num_or_nan(row.get("ica_mean_directional_success")) for row in rows]

    fig, ax = plt.subplots(figsize=(max(9.5, 1.8 * len(labels)), 5.2), constrained_layout=True)
    ax.bar(positions - width, concept, width=width, label="Concept success")
    ax.bar(positions, pca, width=width, label="PCA dir success")
    ax.bar(positions + width, ica, width=width, label="ICA dir success")
    ax.set_title("Intervention Validation Summary")
    ax.set_ylabel("score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    path = output_dir / "intervention_summary_bars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_readme(output_dir: Path, generated: Sequence[Path]) -> Path:
    lines = [
        "# Intervention Validation Plots",
        "",
        "## Purpose",
        "",
        "These plots test whether a discovered direction behaves as expected under controlled instance interventions.",
        "The logic is:",
        "",
        "1. modify the instance in a controlled way",
        "2. check whether the target concept moved in the intended direction",
        "3. check whether the latent direction associated with that concept also moved in the expected direction",
        "",
        "This is stronger than simple correlation because it asks whether the latent reacts coherently under intervention.",
        "",
        "## Interventions used",
        "",
        "- `geometry_spread`: spread customers away from the depot; target concept should move toward lower compactness",
        "- `demand_concentration`: concentrate demand on fewer customers; target concept should move toward higher load concentration",
        "- `tighten_time_windows`: shrink active time windows; target concept should move toward tighter TW profile",
        "- `tighten_distance_limit`: reduce the route distance budget; target concept should move toward tighter distance pressure",
        "",
        "## How to read each plot",
        "",
        "- `intervention_concept_success_heatmap.png`",
        "  - value = fraction of valid instances where the intervention actually changed the target concept in the intended direction",
        "  - high = the intervention itself is well designed",
        "  - low = do not over-interpret the latent plots, because the concept did not move reliably",
        "",
        "- `intervention_pca_directional_success_heatmap.png`",
        "  - for each intervention, select the PCA component most aligned with the target concept on the original data",
        "  - then measure how often that component moves in the direction predicted by the concept change",
        "  - high = the PCA axis behaves consistently with the intended concept intervention",
        "",
        "- `intervention_ica_directional_success_heatmap.png`",
        "  - same logic, but for ICA instead of PCA",
        "  - useful to compare whether PCA or ICA provides the more intervention-consistent axis",
        "",
        "- `intervention_summary_bars.png`",
        "  - compares average concept success, PCA directional success, and ICA directional success across interventions",
        "  - use it as a model-level summary, not as proof for one specific concept",
        "",
        "## Practical interpretation",
        "",
        "- `concept success` high and `PCA/ICA success` high",
        "  - strongest case: the intervention works and the latent direction follows it",
        "",
        "- `concept success` high but `PCA/ICA success` low",
        "  - the concept moved, but the discovered direction does not track it well",
        "",
        "- `concept success` low",
        "  - the intervention itself is weak or not applicable often enough; the latent results are less informative",
        "",
        "- `PCA success` > `ICA success`",
        "  - PCA gives a more intervention-consistent axis for these concepts",
        "",
        "- `ICA success` > `PCA success`",
        "  - ICA isolates a more intervention-faithful factor, even if it may be less stable overall",
        "",
        "## Important caveat",
        "",
        "A high directional success means the selected axis moves in the correct sign more often.",
        "It does not necessarily mean the magnitude of movement is proportional.",
        "For proportionality, consult the comparison markdown and the `aligned_delta_correlation` metrics.",
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
    reports_by_model = _report_by_model(payload)

    generated = [
        _plot_heatmap(
            rows,
            reports_by_model,
            output_dir,
            args.dpi,
            ["concept_success_rate"],
            "Intervention Concept Success",
            "intervention_concept_success_heatmap.png",
        ),
        _plot_heatmap(
            rows,
            reports_by_model,
            output_dir,
            args.dpi,
            ["methods", "pca", "directional_success_rate"],
            "PCA Directional Success",
            "intervention_pca_directional_success_heatmap.png",
        ),
        _plot_heatmap(
            rows,
            reports_by_model,
            output_dir,
            args.dpi,
            ["methods", "ica", "directional_success_rate"],
            "ICA Directional Success",
            "intervention_ica_directional_success_heatmap.png",
        ),
        _plot_summary_bars(rows, output_dir, args.dpi),
    ]
    generated.append(_write_readme(output_dir, generated))

    print(f"Wrote plots to {output_dir}")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
