"""ModelLoader: resolves config, loads checkpoint, prepares inputs."""
from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mavrp.configs.config import Config
    from mavrp.env.models import TransformerModel

from engine.decode_types import DecodeState, DecodeCache


class ModelLoader:
    """Encapsulates config resolution and model loading logic.

    Usage::

        loader = ModelLoader(args)
        config, config_id, ckpt_path = loader.resolve()
        model = loader.load_model()
        node_features, global_features, variant_meta = loader.prepare_inputs(n)
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._config: Optional["Config"] = None
        self._config_id: Optional[int] = None
        self._ckpt_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self) -> Tuple["Config", Optional[int], Path]:
        """Resolve config and checkpoint path from args."""
        from mavrp.configs.config import Config

        args = self._args
        configs = Config.all()
        selected_idx: Optional[int] = None

        if args.config_id is not None:
            if args.config_id < 0 or args.config_id >= len(configs):
                raise IndexError(f"config-id out of range: {args.config_id}")
            config = configs[args.config_id]
            selected_idx = int(args.config_id)
        elif args.checkpoint:
            ckpt_path = Path(args.checkpoint).resolve()
            folder_name = ckpt_path.parent.name
            graph_size = (
                ckpt_path.parent.parent.name if len(ckpt_path.parents) >= 2 else None
            )
            problem = (
                ckpt_path.parent.parent.parent.name
                if len(ckpt_path.parents) >= 3
                else None
            )
            matches = [
                (i, conf)
                for i, conf in enumerate(configs)
                if repr(conf) == folder_name
                and (graph_size is None or str(conf.graph_size) == str(graph_size))
                and (problem is None or conf.problem == problem)
            ]
            if not matches:
                raise ValueError(
                    "Unable to infer config from checkpoint path. "
                    "Pass --config-id explicitly."
                )
            selected_idx, config = matches[0]
        else:
            raise ValueError("Pass either --config-id or --checkpoint")

        if getattr(args, "graph_size", None) is not None:
            config.graph_size = int(args.graph_size)
        if getattr(args, "problem", None) is not None:
            config.problem = str(args.problem)
        if getattr(args, "device", None) is not None:
            config.device = str(args.device)

        if args.checkpoint:
            ckpt_path = Path(args.checkpoint).resolve()
        else:
            ckpt_path = (
                Path(config.working_dir)
                / config.problem
                / str(config.graph_size)
                / repr(config)
                / "baseline.pt"
            )
            ckpt_path = ckpt_path.resolve()

        self._config = config
        self._config_id = selected_idx
        self._ckpt_path = ckpt_path
        return config, selected_idx, ckpt_path

    def load_model(self) -> "TransformerModel":
        """Load the model from the resolved checkpoint."""
        if self._config is None or self._ckpt_path is None:
            raise RuntimeError("Call resolve() before load_model()")
        return _load_model(self._config, self._ckpt_path)

    def prepare_inputs(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Generate a random dataset and return (node_features, global_features, variant_meta)."""
        if self._config is None:
            raise RuntimeError("Call resolve() before prepare_inputs()")
        from engine.decode_ops import variant_metadata_from_inputs

        config = self._config
        dataset = config.get_problem().dataset(
            graph_size=config.graph_size,
            num_samples=n,
            device=config.device,
        )
        node_features = dataset.node_features.to(config.device)
        global_features = dataset.global_features.to(config.device)
        variant_meta = variant_metadata_from_inputs(node_features, global_features)
        return node_features, global_features, variant_meta

    @staticmethod
    def preflight_check() -> None:
        """Raise RuntimeError if torch_geometric is not importable."""
        probe = subprocess.run(
            [sys.executable, "-c", "import torch_geometric"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "The current g-unirouting environment cannot import torch_geometric "
                "(or one of its native dependencies such as torch_cluster). "
                "Fix the target .venv before running action_explainer."
            )


# ---------------------------------------------------------------------------
# Module-level helpers (also called by ExplainerEngine)
# ---------------------------------------------------------------------------


def _load_model(config: "Config", checkpoint_path: Path) -> "TransformerModel":
    from mavrp.env.models import TransformerModel

    model = TransformerModel(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".ckpt":
        model.load_from_ckpt(str(checkpoint_path), baseline=False)
    else:
        state = torch.load(
            checkpoint_path, map_location=config.device, weights_only=True
        )
        model.load_state_dict(state, strict=True, assign=True)
    model = model.to(config.device)
    model.eval()
    return model


def randomize_model_weights(model: "TransformerModel") -> None:
    """Reset all model parameters in-place (for XAI sanity checks)."""

    def _reset(module: torch.nn.Module) -> None:
        if module is model:
            return
        reset_fn = getattr(module, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    model.apply(_reset)
    model.eval()
