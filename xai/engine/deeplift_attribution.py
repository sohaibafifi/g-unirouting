"""DeepLIFT attribution with optional Captum dependency."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from torch import nn

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
from engine.ig_attribution import IG_BASELINE_MODES, build_ig_baseline


def _safe_score_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return torch.where(torch.isfinite(gathered), gathered, torch.zeros_like(gathered))


def _load_captum_deeplift():
    try:
        from captum.attr import DeepLift
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DeepLIFT requires the optional 'captum' dependency. "
            "Install it in the project environment, then re-run the explainer."
        ) from exc
    return DeepLift


def _expand_batch_tensor(
    tensor: Optional[torch.Tensor],
    base_batch: int,
    target_batch: int,
    *,
    batch_dim: int = 0,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[batch_dim] == target_batch:
        return tensor
    if tensor.shape[batch_dim] != base_batch:
        raise RuntimeError(
            f"Unexpected batch size on dim {batch_dim}: got {tensor.shape[batch_dim]}, "
            f"expected {base_batch} or {target_batch}."
        )
    if target_batch % base_batch != 0:
        raise RuntimeError(
            f"Cannot expand tensor from batch {base_batch} to {target_batch}."
        )
    repeats = target_batch // base_batch
    return torch.cat([tensor] * repeats, dim=batch_dim)


def _expand_decode_state(
    state: DecodeState,
    base_batch: int,
    target_batch: int,
) -> DecodeState:
    if target_batch == base_batch:
        return state
    return DecodeState(
        current_node=_expand_batch_tensor(state.current_node, base_batch, target_batch),
        not_served=_expand_batch_tensor(state.not_served, base_batch, target_batch),
        leave_time=_expand_batch_tensor(state.leave_time, base_batch, target_batch),
        deliveries=_expand_batch_tensor(state.deliveries, base_batch, target_batch),
        pickups=_expand_batch_tensor(state.pickups, base_batch, target_batch),
        distance=_expand_batch_tensor(state.distance, base_batch, target_batch),
        total_distance=_expand_batch_tensor(state.total_distance, base_batch, target_batch),
        is_depot=_expand_batch_tensor(state.is_depot, base_batch, target_batch),
        hidden=_expand_batch_tensor(
            state.hidden,
            base_batch,
            target_batch,
            batch_dim=1,
        ),
    )


class _StepForwardModule(nn.Module):
    """Captum-compatible wrapper around one decoder step objective."""

    def __init__(
        self,
        model: "TransformerModel",
        state: DecodeState,
        action: torch.Tensor,
        alt_action: torch.Tensor,
        has_alt: torch.Tensor,
        mode: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.state = state
        self.action = action
        self.alt_action = alt_action
        self.has_alt = has_alt
        self.mode = mode
        self.base_batch = int(action.shape[0])

    def forward(
        self,
        node_inputs: torch.Tensor,
        global_inputs: torch.Tensor,
    ) -> torch.Tensor:
        target_batch = int(node_inputs.shape[0])
        state = _expand_decode_state(self.state, self.base_batch, target_batch)
        action = _expand_batch_tensor(self.action, self.base_batch, target_batch)
        alt_action = _expand_batch_tensor(self.alt_action, self.base_batch, target_batch)
        has_alt = _expand_batch_tensor(self.has_alt, self.base_batch, target_batch)

        common = build_common(node_inputs, global_inputs)
        cache = encode_inputs(self.model, node_inputs, global_inputs)
        logits, _, _, _, _ = step_logits_and_mask(self.model, cache, common, state)
        selected_logit = _safe_score_gather(logits, action)
        if self.mode == "selected":
            return selected_logit
        if self.mode == "contrastive":
            alt_logit = _safe_score_gather(logits, alt_action)
            return (selected_logit - alt_logit) * has_alt.float()
        raise ValueError(f"Unsupported DeepLIFT step mode: {self.mode}")


class DeepLiftAttribution(AttributionBase):
    """DeepLIFT attribution using the same baseline family as IG."""

    def __init__(self, baseline_mode: str) -> None:
        if baseline_mode not in IG_BASELINE_MODES:
            raise ValueError(
                f"Unsupported DeepLIFT baseline {baseline_mode!r}. "
                f"Expected one of: {', '.join(IG_BASELINE_MODES)}"
            )
        self._deep_lift_cls = _load_captum_deeplift()
        self.baseline_mode = str(baseline_mode)
        self._base_node: Optional[torch.Tensor] = None
        self._base_global: Optional[torch.Tensor] = None

    def method_key(self) -> str:
        return "deeplift"

    def uses_reference_baseline(self) -> bool:
        return True

    def needs_local_counterfactual_grads(self) -> bool:
        return True

    def set_baseline(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        self._base_node, self._base_global = build_ig_baseline(
            node_features=node_features,
            global_features=global_features,
            mode=self.baseline_mode,
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
            raise RuntimeError("Call prepare_inputs() before compute_step()")

        batch_size, num_nodes = node_features.shape[:2]
        device = node_features.device

        base_node = self._base_node.to(device=device, dtype=node_features.dtype)
        base_global = self._base_global.to(device=device, dtype=global_features.dtype)

        selected_module = _StepForwardModule(
            model=model,
            state=state,
            action=action,
            alt_action=alt_action,
            has_alt=has_alt,
            mode="selected",
        )
        deep_lift = self._deep_lift_cls(selected_module)
        attr_node, attr_global = deep_lift.attribute(
            inputs=(node_features, global_features),
            baselines=(base_node, base_global),
            return_convergence_delta=False,
        )

        if bool(has_alt.any().item()):
            contrastive_module = _StepForwardModule(
                model=model,
                state=state,
                action=action,
                alt_action=alt_action,
                has_alt=has_alt,
                mode="contrastive",
            )
            deep_lift_cf = self._deep_lift_cls(contrastive_module)
            cf_node, cf_global = deep_lift_cf.attribute(
                inputs=(node_features, global_features),
                baselines=(base_node, base_global),
                return_convergence_delta=False,
            )
        else:
            cf_node = torch.zeros_like(node_features)
            cf_global = torch.zeros_like(global_features)

        grad_by_feature = extract_feature_grads(attr_node, attr_global, selected_features)
        contrastive_grad_by_feature = extract_feature_grads(
            cf_node, cf_global, selected_features
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
            "reference_baseline": self.baseline_mode,
            "deeplift_baseline": self.baseline_mode,
        }
