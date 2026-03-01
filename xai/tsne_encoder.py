from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE


def _resolve_config(config_id: int | None, checkpoint: str | None) -> tuple["Config", Path]:
    from mavrp.configs.config import Config

    configs = Config.all()
    if config_id is not None:
        if config_id < 0 or config_id >= len(configs):
            raise IndexError(f"config-id out of range: {config_id}")
        config = configs[config_id]
    elif checkpoint:
        ckpt_path = Path(checkpoint).resolve()
        folder_name = ckpt_path.parent.name
        matches = [conf for conf in configs if repr(conf) == folder_name]
        if not matches:
            raise ValueError("Unable to infer config from checkpoint path. Pass --config-id.")
        config = matches[0]
    else:
        raise ValueError("Pass either --config-id or --checkpoint")

    ckpt_path = (
        Path(checkpoint).resolve()
        if checkpoint
        else (Path(config.working_dir) / config.problem / str(config.graph_size) / repr(config) / "baseline.pt").resolve()
    )
    return config, ckpt_path


def _load_model(config: "Config", checkpoint_path: Path) -> "TransformerModel":
    from mavrp.env.models import TransformerModel

    model = TransformerModel(config)
    if checkpoint_path.suffix == ".ckpt":
        model.load_from_ckpt(str(checkpoint_path), baseline=False)
    else:
        state = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(state, strict=True, assign=True)
    return model.to(config.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="t-SNE view of encoder representations across MAVRP variants."
    )
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples-per-variant", type=int, default=100)
    parser.add_argument("--output", default="logs/xai/tsne_encoder.png")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    probe = subprocess.run(
        [sys.executable, "-c", "import torch_geometric"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "The current g-unirouting environment cannot import torch_geometric. "
            "Fix the target .venv before running tsne_encoder."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config, checkpoint_path = _resolve_config(args.config_id, args.checkpoint)
    if args.device:
        config.device = str(args.device)
    model = _load_model(config, checkpoint_path)
    problem = config.get_problem()

    variants: List[str] = problem.get_variants()
    encodings = []
    labels: List[str] = []
    for variant in variants:
        dataset = problem.dataset(
            graph_size=config.graph_size,
            num_samples=args.samples_per_variant,
            variant=variant,
            device=config.device,
        )
        node_features = dataset.node_features.to(config.device)
        global_features = dataset.global_features.to(config.device)
        with torch.inference_mode():
            encoded = list(model.encoder((node_features, global_features)))
            node_embeddings = encoded[0]
            mean_embeddings = node_embeddings.mean(dim=1).detach().cpu().numpy()
        encodings.append(mean_embeddings)
        labels.extend([variant] * mean_embeddings.shape[0])

    if not encodings:
        raise RuntimeError("No encodings collected.")

    features = np.concatenate(encodings, axis=0)
    tsne = TSNE(n_components=2, random_state=args.seed)
    coords = tsne.fit_transform(features)

    plt.figure(figsize=(11, 8))
    palette = plt.cm.tab20(np.linspace(0, 1, len(variants)))
    labels_np = np.array(labels)
    for i, variant in enumerate(variants):
        mask = labels_np == variant
        points = coords[mask]
        plt.scatter(points[:, 0], points[:, 1], s=18, alpha=0.65, label=variant.upper(), color=palette[i])

    plt.title(f"Encoder t-SNE | {config.problem} | {config.graph_size} | {config.encoder.__name__}")
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=8, ncol=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved t-SNE figure to {output_path}")


if __name__ == "__main__":
    main()
