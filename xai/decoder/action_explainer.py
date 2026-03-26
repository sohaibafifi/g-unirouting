"""Action-level explainability — thin CLI entry point."""
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L

from engine.attribution import GradientAttribution
from engine.deeplift_attribution import DeepLiftAttribution
from engine.explainer_engine import ExplainerEngine
from engine.ig_attribution import IGAttribution
from engine.model_io import ModelLoader, randomize_model_weights
from utils.text_utils import slugify


METHOD_ORDER = ["gradient", "integrated_gradients", "deeplift"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Action-level explainability for g-unirouting TransformerModel checkpoints."
    )
    parser.add_argument(
        "--config-id",
        type=int,
        default=None,
        help="Index in Config.all(). Optional if --checkpoint can be matched to a config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to baseline.pt or checkpoint.ckpt.",
    )
    parser.add_argument("--problem", default=None, help="Optional config.problem override.")
    parser.add_argument(
        "--graph-size", type=int, default=None, help="Optional config.graph_size override."
    )
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument(
        "--attribution-methods",
        default="gradient,integrated_gradients",
        help="Comma-separated: gradient, integrated_gradients, deeplift.",
    )
    parser.add_argument("--feasibility-weight", type=float, default=None)
    parser.add_argument("--feasibility-top-m", type=int, default=8)
    parser.add_argument("--feasibility-cost-weight", type=float, default=0.25)
    parser.add_argument("--attr-features", default="auto")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument(
        "--ig-baseline",
        choices=["mean-fill", "zero-with-customers-at-depot", "zero-all", "zero-with-current-locs"],
        default="mean-fill",
        help="Reference baseline used by integrated gradients and DeepLIFT.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize-weights", action="store_true")
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="logs/xai/decoder/reports")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-instance-traces", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _parse_attribution_methods(raw: str):
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        return ["gradient", "integrated_gradients"]
    normalized = []
    for token in tokens:
        if token in {"gradient", "grad", "saliency"}:
            key = "gradient"
        elif token in {"integrated_gradients", "ig"}:
            key = "integrated_gradients"
        elif token in {"deeplift", "deep_lift", "deep-lift", "dl"}:
            key = "deeplift"
        else:
            raise ValueError(
                f"Unsupported attribution method {token!r}. "
                "Use 'gradient', 'integrated_gradients', and/or 'deeplift'."
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def _run_single(
    args: argparse.Namespace,
    loader: ModelLoader,
    method: str,
) -> Path:
    """Run one attribution method and return the saved report path."""
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    if data_seed is not None:
        L.seed_everything(int(data_seed), workers=True)

    model = loader.load_model()
    if args.randomize_weights:
        randomize_model_weights(model)

    config, config_id, checkpoint_path = loader.resolve()
    node_features, global_features, variant_meta = loader.prepare_inputs(args.num_instances)

    if method == "gradient":
        attribution = GradientAttribution()
    elif method == "integrated_gradients":
        attribution = IGAttribution(
            ig_steps=int(args.ig_steps), ig_baseline=str(args.ig_baseline)
        )
    elif method == "deeplift":
        attribution = DeepLiftAttribution(baseline_mode=str(args.ig_baseline))
    else:
        raise ValueError(f"Unsupported attribution method: {method}")

    engine = ExplainerEngine(
        model=model,
        config=config,
        config_id=config_id,
        checkpoint_path=checkpoint_path,
        args=args,
        attribution=attribution,
    )
    report = engine.run(node_features, global_features, variant_meta)
    return engine.save_report(report, Path(args.output_dir))


def _build_bundle(
    output_dir: Path,
    report_paths: Dict[str, Optional[Path]],
) -> Optional[Path]:
    present = {
        key: path
        for key, path in report_paths.items()
        if path is not None
    }
    if len(present) <= 1:
        return None

    payload: Dict[str, Any] = {
        "kind": "xai_dual_bundle",
        "timestamp": int(time.time()),
        "methods": [method for method in METHOD_ORDER if method in present]
        + sorted(method for method in present if method not in METHOD_ORDER),
        "reports": {
            key: {"path": str(path), "path_resolved": str(path.resolve())}
            for key, path in present.items()
        },
    }

    model_slug = "bundle"
    config_summary: Dict[str, Any] = {}
    for key, path in present.items():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        cfg = data.get("config", {}) or {}
        if model_slug == "bundle":
            model_slug = str(cfg.get("model_slug", "")).strip() or model_slug
        config_summary[key] = {
            "model_label": cfg.get("model_label"),
            "attribution_method": cfg.get("attribution_method"),
            "checkpoint_path": cfg.get("checkpoint_path"),
            "checkpoint_path_resolved": cfg.get("checkpoint_path_resolved"),
        }
    if config_summary:
        payload["config"] = config_summary

    bundle_path = output_dir / f"xai_bundle_{slugify(model_slug)}_{int(time.time())}.json"
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved XAI bundle to {bundle_path}")
    return bundle_path


def main() -> None:
    ModelLoader.preflight_check()

    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        L.seed_everything(int(args.seed), workers=True)

    methods = _parse_attribution_methods(args.attribution_methods)
    loader = ModelLoader(args)
    loader.resolve()

    out = {method: None for method in methods}
    for method in methods:
        out[method] = _run_single(args, loader, method)

    _build_bundle(
        output_dir=Path(args.output_dir),
        report_paths=out,
    )


if __name__ == "__main__":
    main()
