"""DeepLIFT explainability — standalone CLI entry point."""
from __future__ import annotations

import argparse
import sys

from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L

from engine.deeplift_attribution import DeepLiftAttribution
from engine.explainer_engine import ExplainerEngine
from engine.ig_attribution import IG_BASELINE_MODES
from engine.model_io import ModelLoader, randomize_model_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepLIFT action-level explainability for g-unirouting checkpoints."
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--graph-size", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument("--attr-features", default="auto")
    parser.add_argument(
        "--deeplift-baseline",
        choices=IG_BASELINE_MODES,
        default="mean-fill",
    )
    parser.add_argument(
        "--ig-baseline",
        dest="deeplift_baseline",
        choices=IG_BASELINE_MODES,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-instance-traces", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--feasibility-weight", type=float, default=None)
    parser.add_argument("--feasibility-top-m", type=int, default=8)
    parser.add_argument("--feasibility-cost-weight", type=float, default=0.25)
    parser.add_argument("--randomize-weights", action="store_true", default=False)
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.seed is not None:
        L.seed_everything(int(args.seed), workers=True)

    loader = ModelLoader(args)
    config, config_id, checkpoint_path = loader.resolve()
    model = loader.load_model()
    if getattr(args, "randomize_weights", False):
        randomize_model_weights(model)

    data_seed = args.data_seed if args.data_seed is not None else args.seed
    if data_seed is not None:
        L.seed_everything(int(data_seed), workers=True)

    node_features, global_features, variant_meta = loader.prepare_inputs(args.num_instances)

    attribution = DeepLiftAttribution(baseline_mode=str(args.deeplift_baseline))
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


def main() -> None:
    ModelLoader.preflight_check()
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
