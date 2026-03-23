"""IGAttribution: Integrated Gradients attribution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mavrp.env.models import TransformerModel

from engine.attribution import AttributionBase
from engine.decode_ops import (
    build_common,
    encode_inputs,
    extract_feature_grads,
    grad_to_node_scores,
    step_logits_and_mask,
)
from engine.decode_types import DecodeState


IG_BASELINE_MODES = (
    "mean-fill",
    "zero-with-customers-at-depot",
    "zero-all",
    "zero-with-current-locs",
)


def build_ig_baseline(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the IG baseline tensors for the given mode."""
    mode = str(mode).strip()
    if mode not in IG_BASELINE_MODES:
        raise ValueError(
            f"Unsupported --ig-baseline={mode!r}. "
            f"Expected one of: {', '.join(IG_BASELINE_MODES)}"
        )

    node_base = torch.zeros_like(node_features)
    global_base = torch.zeros_like(global_features)

    if mode == "zero-all":
        return node_base, global_base

    if mode == "zero-with-customers-at-depot":
        depot_locs = node_features[:, :1, :2].detach()
        node_base[:, :, :2] = depot_locs.expand(-1, node_features.size(1), -1)
        return node_base, global_base

    if mode == "zero-with-current-locs":
        node_base[:, :, :2] = node_features[:, :, :2].detach()
        return node_base, global_base

    # mean-fill: average over batch
    node_mean = node_features.detach().mean(dim=0, keepdim=True)
    global_mean = global_features.detach().mean(dim=0, keepdim=True)
    return node_mean.expand_as(node_features).clone(), global_mean.expand_as(global_features).clone()


def _safe_score_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return torch.where(torch.isfinite(gathered), gathered, torch.zeros_like(gathered))


class IGAttribution(AttributionBase):
    """Integrated Gradients attribution.

    Args:
        ig_steps: Number of interpolation steps.
        ig_baseline: Baseline mode string (see IG_BASELINE_MODES).
    """

    def __init__(self, ig_steps: int, ig_baseline: str) -> None:
        self.ig_steps = ig_steps
        self.ig_baseline = ig_baseline
        self._base_node: Optional[torch.Tensor] = None
        self._base_global: Optional[torch.Tensor] = None

    def method_key(self) -> str:
        return "integrated_gradients"

    def uses_reference_baseline(self) -> bool:
        return True

    def needs_local_counterfactual_grads(self) -> bool:
        return True

    def set_baseline(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        """Compute and cache the baseline for the given input batch."""
        self._base_node, self._base_global = build_ig_baseline(
            node_features, global_features, self.ig_baseline
        )

    def prepare_inputs(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        self.set_baseline(node_features, global_features)

    def compute_step(
        self,
        model: "TransformerModel",
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        state: DecodeState,
        action: torch.Tensor,
        alt_action: torch.Tensor,
        has_alt: torch.Tensor,
        selected_features: Sequence[str],
    ) -> Tuple[
        Dict[str, Optional[torch.Tensor]],
        Dict[str, Optional[torch.Tensor]],
        torch.Tensor,
    ]:
        if self._base_node is None or self._base_global is None:
            raise RuntimeError("Call set_baseline() before compute_step()")

        batch_size, num_nodes = node_features.shape[:2]
        device = node_features.device

        (ig_node, ig_global), (ig_cf_node, ig_cf_global) = _compute_integrated_grads(
            model=model,
            state=state,
            node_features=node_features,
            global_features=global_features,
            base_node_features=self._base_node,
            base_global_features=self._base_global,
            action=action,
            alt_action=alt_action,
            has_alt=has_alt,
            ig_steps=self.ig_steps,
        )

        grad_by_feature = extract_feature_grads(ig_node, ig_global, selected_features)
        contrastive_grad_by_feature = extract_feature_grads(
            ig_cf_node, ig_cf_global, selected_features
        )

        decision_node_scores = torch.zeros(
            (batch_size, num_nodes), dtype=node_features.dtype, device=device
        )
        for name in selected_features:
            node_score = grad_to_node_scores(grad_by_feature.get(name), num_nodes)
            if node_score is not None:
                decision_node_scores = decision_node_scores + node_score

        return grad_by_feature, contrastive_grad_by_feature, decision_node_scores

    def summary(self) -> Dict[str, Any]:
        return {
            "attribution_method": self.method_key(),
            "reference_baseline": self.ig_baseline,
            "ig_steps": self.ig_steps,
            "ig_baseline": self.ig_baseline,
        }


def _compute_integrated_grads(
    model: "TransformerModel",
    state: DecodeState,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    base_node_features: torch.Tensor,
    base_global_features: torch.Tensor,
    action: torch.Tensor,
    alt_action: torch.Tensor,
    has_alt: torch.Tensor,
    ig_steps: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    node_delta = node_features - base_node_features
    global_delta = global_features - base_global_features

    acc_node = torch.zeros_like(node_features)
    acc_global = torch.zeros_like(global_features)
    acc_contrastive_node = torch.zeros_like(node_features)
    acc_contrastive_global = torch.zeros_like(global_features)

    alphas = torch.linspace(
        1.0 / float(ig_steps),
        1.0,
        steps=int(ig_steps),
        device=node_features.device,
        dtype=node_features.dtype,
    )

    for alpha in alphas:
        node_interp = (base_node_features + alpha * node_delta).detach().requires_grad_(True)
        global_interp = (base_global_features + alpha * global_delta).detach().requires_grad_(True)

        common = build_common(node_interp, global_interp)
        cache = encode_inputs(model, node_interp, global_interp)
        logits, _, _, _, _ = step_logits_and_mask(model, cache, common, state)

        selected_logit = _safe_score_gather(logits, action)
        grads = torch.autograd.grad(
            outputs=selected_logit.sum(),
            inputs=[node_interp, global_interp],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        acc_node = acc_node + (grads[0] if grads[0] is not None else torch.zeros_like(node_interp))
        acc_global = acc_global + (
            grads[1] if grads[1] is not None else torch.zeros_like(global_interp)
        )

        if bool(has_alt.any().item()):
            alt_logit = _safe_score_gather(logits, alt_action)
            contrastive_target = (selected_logit - alt_logit) * has_alt.float()
            contrastive_grads = torch.autograd.grad(
                outputs=contrastive_target.sum(),
                inputs=[node_interp, global_interp],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            acc_contrastive_node = acc_contrastive_node + (
                contrastive_grads[0]
                if contrastive_grads[0] is not None
                else torch.zeros_like(node_interp)
            )
            acc_contrastive_global = acc_contrastive_global + (
                contrastive_grads[1]
                if contrastive_grads[1] is not None
                else torch.zeros_like(global_interp)
            )

    inv_steps = 1.0 / float(ig_steps)
    ig_node = node_delta * (acc_node * inv_steps)
    ig_global = global_delta * (acc_global * inv_steps)
    ig_contrastive_node = node_delta * (acc_contrastive_node * inv_steps)
    ig_contrastive_global = global_delta * (acc_contrastive_global * inv_steps)
    return (ig_node, ig_global), (ig_contrastive_node, ig_contrastive_global)


def compute_local_contrastive_feature_grads(
    model: "TransformerModel",
    state: DecodeState,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    action: torch.Tensor,
    alt_action: torch.Tensor,
    has_alt: torch.Tensor,
    selected_features: Sequence[str],
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute local (non-IG) contrastive gradients for counterfactual proposals."""
    if not bool(has_alt.any().item()):
        return {}

    node_inputs = node_features.detach().clone().requires_grad_(True)
    global_inputs = global_features.detach().clone().requires_grad_(True)
    common = build_common(node_inputs, global_inputs)
    cache = encode_inputs(model, node_inputs, global_inputs)
    logits, _, _, _, _ = step_logits_and_mask(model, cache, common, state)
    selected_logit = _safe_score_gather(logits, action)
    alt_logit = _safe_score_gather(logits, alt_action)
    contrastive_target = (selected_logit - alt_logit) * has_alt.float()
    grads = torch.autograd.grad(
        outputs=contrastive_target.sum(),
        inputs=[node_inputs, global_inputs],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return extract_feature_grads(grads[0], grads[1], selected_features)
