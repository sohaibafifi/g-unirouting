"""AttributionBase and GradientAttribution."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mavrp.env.models import TransformerModel

from engine.decode_ops import (
    build_common,
    encode_inputs,
    extract_feature_grads,
    grad_to_instance_scores,
    grad_to_node_scores,
    step_logits_and_mask,
)
from engine.decode_types import DecodeState


class AttributionBase:
    """Abstract base for attribution methods."""

    def method_key(self) -> str:
        raise NotImplementedError

    def prepare_inputs(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        """Optional hook to cache method-specific state before the step loop."""
        return None

    def uses_reference_baseline(self) -> bool:
        return False

    def needs_local_counterfactual_grads(self) -> bool:
        return False

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
        """Return (grad_by_feature, contrastive_grad_by_feature, node_scores)."""
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        """Return any attribution-method-specific summary fields."""
        return {"attribution_method": self.method_key()}


class GradientAttribution(AttributionBase):
    """Vanilla gradient (saliency) attribution."""

    def method_key(self) -> str:
        return "gradient"

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
        batch_size, num_nodes = node_features.shape[:2]
        device = node_features.device

        node_inputs = node_features.detach().clone().requires_grad_(True)
        global_inputs = global_features.detach().clone().requires_grad_(True)

        common = build_common(node_inputs, global_inputs)
        cache = encode_inputs(model, node_inputs, global_inputs)
        logits, _, _, logprobs, _ = step_logits_and_mask(model, cache, common, state)

        selected_logit = logits.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        grads = torch.autograd.grad(
            outputs=selected_logit.sum(),
            inputs=[node_inputs, global_inputs],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        contrastive_target = (selected_logit - logits.gather(-1, alt_action.unsqueeze(-1)).squeeze(-1)) * has_alt.float()
        contrastive_grads = torch.autograd.grad(
            outputs=contrastive_target.sum(),
            inputs=[node_inputs, global_inputs],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_by_feature = extract_feature_grads(grads[0], grads[1], selected_features)
        contrastive_grad_by_feature = extract_feature_grads(
            contrastive_grads[0], contrastive_grads[1], selected_features
        )

        decision_node_scores = torch.zeros(
            (batch_size, num_nodes), dtype=node_inputs.dtype, device=device
        )
        for name in selected_features:
            node_score = grad_to_node_scores(grad_by_feature.get(name), num_nodes)
            if node_score is not None:
                decision_node_scores = decision_node_scores + node_score

        return grad_by_feature, contrastive_grad_by_feature, decision_node_scores
