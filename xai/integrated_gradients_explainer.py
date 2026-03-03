from __future__ import annotations

import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch

IG_BASELINE_MODES = (
    "mean-fill",
    "zero-with-customers-at-depot",
    "zero-all",
    "zero-with-current-locs",
)


def _safe_score_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return torch.where(torch.isfinite(gathered), gathered, torch.zeros_like(gathered))


def _build_ig_baseline(
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    # Average over the batch dimension for both tensors so the baseline
    # represents the mean instance in this batch consistently.
    node_mean = node_features.detach().mean(dim=0, keepdim=True)
    global_mean = global_features.detach().mean(dim=0, keepdim=True)
    return node_mean.expand_as(node_features).clone(), global_mean.expand_as(global_features).clone()



def _compute_integrated_grads(
    model: Any,
    grad_base_module: Any,
    state: Any,
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

        common = grad_base_module._build_common(node_interp, global_interp)
        cache = grad_base_module._encode_inputs(model, node_interp, global_interp)
        logits, _, _, _, _ = grad_base_module._step_logits_and_mask(
            model, cache, common, state
        )

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


def _compute_local_contrastive_feature_grads(
    model: Any,
    grad_base_module: Any,
    state: Any,
    node_features: torch.Tensor,
    global_features: torch.Tensor,
    action: torch.Tensor,
    alt_action: torch.Tensor,
    has_alt: torch.Tensor,
    selected_features: Sequence[str],
) -> Dict[str, Optional[torch.Tensor]]:
    if not bool(has_alt.any().item()):
        return {}

    node_inputs = node_features.detach().clone().requires_grad_(True)
    global_inputs = global_features.detach().clone().requires_grad_(True)
    common = grad_base_module._build_common(node_inputs, global_inputs)
    cache = grad_base_module._encode_inputs(model, node_inputs, global_inputs)
    logits, _, _, _, _ = grad_base_module._step_logits_and_mask(
        model, cache, common, state
    )
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
    return grad_base_module._extract_feature_grads(grads[0], grads[1], selected_features)


def _simple_step_record(
    step: int,
    action: torch.Tensor,
    selected_logit: torch.Tensor,
    selected_logprob: torch.Tensor,
    topk_metrics: Dict[int, Dict[str, float]],
    top_features: List[Dict[str, float]],
    top_constraints: List[Dict[str, float]],
    contrastive_alt_available_rate: float,
    contrastive_logit_gap_mean: float,
    contrastive_logprob_gap_mean: float,
    contrastive_top_constraints: List[Dict[str, float]],
    recourse_flags: torch.Tensor,
    recourse_cost_est: torch.Tensor,
    action_feasible: torch.Tensor,
    done_before: torch.Tensor,
    done_after: torch.Tensor,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "actions": action.detach().cpu().tolist(),
        "selected_logit_mean": float(selected_logit.mean().item()),
        "selected_logprob_mean": float(selected_logprob.mean().item()),
        "top_features": top_features,
        "top_constraints": top_constraints,
        "contrastive": {
            "alt_available_rate": float(contrastive_alt_available_rate),
            "mean_logit_gap": float(contrastive_logit_gap_mean),
            "mean_logprob_gap": float(contrastive_logprob_gap_mean),
            "top_constraints": contrastive_top_constraints,
        },
        "recourse_rate": float(recourse_flags.float().mean().item()),
        "recourse_cost_est_mean": float(recourse_cost_est.mean().item()),
        "chosen_action_feasible_rate": float(action_feasible.float().mean().item()),
        "done_before": done_before.detach().cpu().tolist(),
        "done_after": done_after.detach().cpu().tolist(),
        "deletion": {str(k): m for k, m in topk_metrics.items()},
    }


def run(args: argparse.Namespace, grad_base_module: Any | None = None) -> Path:
    if grad_base_module is None:
        XAI_DIR = Path(__file__).resolve().parent
        if str(XAI_DIR) not in sys.path:
            sys.path.insert(0, str(XAI_DIR))
        import action_explainer as grad_base_module  # type: ignore

    grad_base = grad_base_module
    if args.seed is not None:
        L.seed_everything(int(args.seed), workers=True)

    config, config_id, checkpoint_path = grad_base._resolve_config(args)
    model = grad_base._load_model(config, checkpoint_path)
    if bool(getattr(args, "randomize_weights", False)):
        grad_base._randomize_model_weights(model)
    recourse_enabled = grad_base._is_recourse_decoder(model)

    topk_list = grad_base._parse_topk_nodes(args.topk_nodes)
    max_attr_k = max(topk_list)
    selected_features = grad_base._parse_attr_features(args.attr_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_seed = args.data_seed if args.data_seed is not None else args.seed
    if data_seed is not None:
        L.seed_everything(int(data_seed), workers=True)

    node_features, global_features, variant_meta = grad_base._prepare_inputs(
        config, args.num_instances
    )
    base_node_features, base_global_features = _build_ig_baseline(
        node_features,
        global_features,
        args.ig_baseline,
    )

    batch_size, num_nodes = node_features.shape[:2]
    num_store = (
        min(args.num_instances, args.max_instances_to_store)
        if args.save_instance_traces
        else 0
    )
    instance_traces = (
        grad_base._init_instance_traces(node_features, num_store, variant_meta)
        if num_store > 0
        else []
    )

    per_feature_attr: Dict[str, List[float]] = defaultdict(list)
    per_feature_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
    per_constraint_attr: Dict[str, List[float]] = defaultdict(list)
    per_constraint_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
    per_k_logit_drop: Dict[int, List[float]] = defaultdict(list)
    per_k_logprob_drop: Dict[int, List[float]] = defaultdict(list)
    per_k_flip_rate: Dict[int, List[float]] = defaultdict(list)
    contrastive_alt_available_history: List[float] = []
    contrastive_logit_gap_history: List[float] = []
    contrastive_logprob_gap_history: List[float] = []
    recourse_rate_history: List[float] = []
    recourse_cost_est_history: List[float] = []
    chosen_feasible_rate_history: List[float] = []
    counterfactual_available_history: List[float] = []
    counterfactual_switch_history: List[float] = []
    counterfactual_make_feasible_history: List[float] = []
    counterfactual_approximate_history: List[float] = []
    counterfactual_relative_delta_history: List[float] = []
    counterfactual_by_feature: Dict[str, List[float]] = defaultdict(list)

    step_records: List[Dict[str, Any]] = []
    state = grad_base._init_state(grad_base._build_common(node_features, global_features))
    top_k_effective = 0
    executed_steps = 0

    for step in range(args.max_steps):
        if not bool(state.not_served.any().item()):
            break
        executed_steps += 1
        done_before = ~state.not_served.any(dim=1)

        common = grad_base._build_common(node_features, global_features)
        cache = grad_base._encode_inputs(model, node_features, global_features)
        logits, policy_mask, full_mask, logprobs, potential_distance = grad_base._step_logits_and_mask(
            model, cache, common, state
        )

        action = logprobs.argmax(dim=-1)
        selected_logit = _safe_score_gather(logits, action)
        selected_logprob = _safe_score_gather(logprobs, action)

        (
            alt_action_policy,
            has_alt_policy,
            alt_logit_policy,
            alt_logprob_policy,
        ) = grad_base._select_best_alternative(
            logits=logits,
            logprobs=logprobs,
            chosen_action=action,
            action_mask=~policy_mask,
        )
        (
            alt_action,
            has_alt,
            alt_logit,
            alt_logprob,
        ) = grad_base._select_best_alternative(
            logits=logits,
            logprobs=logprobs,
            chosen_action=action,
            action_mask=~full_mask,
        )
        contrastive_logit_gap = selected_logit - _safe_score_gather(logits, alt_action)
        contrastive_logprob_gap = selected_logprob - _safe_score_gather(logprobs, alt_action)
        contrastive_logit_gap_policy = selected_logit - alt_logit_policy
        contrastive_logprob_gap_policy = selected_logprob - alt_logprob_policy
        contrastive_alt_available_history.append(float(has_alt.float().mean().item()))
        contrastive_logit_gap_history.append(
            grad_base._masked_mean(contrastive_logit_gap.detach(), has_alt)
        )
        contrastive_logprob_gap_history.append(
            grad_base._masked_mean(contrastive_logprob_gap.detach(), has_alt)
        )

        (
            recourse_flags,
            recourse_cost_est,
            action_feasible,
        ) = grad_base._infer_recourse_for_action(
            common,
            action.detach(),
            full_mask.detach(),
            recourse_enabled,
        )
        recourse_rate_history.append(float(recourse_flags.float().mean().item()))
        recourse_cost_est_history.append(float(recourse_cost_est.mean().item()))
        chosen_feasible_rate_history.append(float(action_feasible.float().mean().item()))
        alt_action_feasible = (
            ~full_mask.gather(1, alt_action.unsqueeze(-1)).squeeze(-1)
        ) & has_alt
        alt_recourse_flags = has_alt & recourse_enabled & (alt_action != 0) & (~alt_action_feasible)
        alt_action_policy_feasible = (
            ~full_mask.gather(1, alt_action_policy.unsqueeze(-1)).squeeze(-1)
        ) & has_alt_policy
        alt_recourse_flags_policy = (
            has_alt_policy
            & recourse_enabled
            & (alt_action_policy != 0)
            & (~alt_action_policy_feasible)
        )

        grads, contrastive_grads = _compute_integrated_grads(
            model=model,
            grad_base_module=grad_base,
            state=state,
            node_features=node_features,
            global_features=global_features,
            base_node_features=base_node_features,
            base_global_features=base_global_features,
            action=action.detach(),
            alt_action=alt_action.detach(),
            has_alt=has_alt.detach(),
            ig_steps=int(args.ig_steps),
        )

        grad_by_feature = grad_base._extract_feature_grads(
            grads[0], grads[1], selected_features
        )
        contrastive_grad_by_feature = grad_base._extract_feature_grads(
            contrastive_grads[0], contrastive_grads[1], selected_features
        )

        step_feature_attr_mean: Dict[str, float] = {}
        step_contrastive_feature_attr_mean: Dict[str, float] = {}
        instance_feature_attr: Dict[str, torch.Tensor] = {}
        instance_contrastive_feature_attr: Dict[str, torch.Tensor] = {}
        node_scores = torch.zeros(
            (batch_size, num_nodes), dtype=node_features.dtype, device=node_features.device
        )

        for name in selected_features:
            grad = grad_by_feature.get(name)
            inst_score = grad_base._grad_to_instance_scores(grad)
            if inst_score is None:
                inst_score = torch.zeros(
                    batch_size, dtype=node_features.dtype, device=node_features.device
                )
                mean_score = 0.0
            else:
                mean_score = float(inst_score.mean().item())
            step_feature_attr_mean[name] = mean_score
            instance_feature_attr[name] = inst_score
            per_feature_attr[name].append(mean_score)

            node_score = grad_base._grad_to_node_scores(grad, num_nodes)
            if node_score is not None:
                node_scores = node_scores + node_score

            contrastive_grad = contrastive_grad_by_feature.get(name)
            contrastive_inst_score = grad_base._grad_to_instance_scores(contrastive_grad)
            if contrastive_inst_score is None:
                contrastive_inst_score = torch.zeros(
                    batch_size, dtype=node_features.dtype, device=node_features.device
                )
                contrastive_mean_score = 0.0
            else:
                contrastive_mean_score = float(contrastive_inst_score.mean().item())
            step_contrastive_feature_attr_mean[name] = contrastive_mean_score
            instance_contrastive_feature_attr[name] = contrastive_inst_score
            per_feature_contrastive_attr[name].append(contrastive_mean_score)

        node_scores[:, 0] = 0.0
        rank_candidate_mask = state.not_served.clone()
        rank_candidate_mask[:, 0] = True
        (
            top_nodes_all,
            top_scores_all,
            top_nodes_valid_all,
            top_k_effective,
        ) = grad_base._top_nodes_and_scores(node_scores, rank_candidate_mask, max_attr_k)

        top_features_payload = grad_base._top_feature_payload(step_feature_attr_mean, top_n=3)
        step_constraint_attr_mean = grad_base._aggregate_constraint_scores(step_feature_attr_mean)
        top_constraints_payload = grad_base._top_constraint_payload(
            step_constraint_attr_mean, top_n=None
        )
        step_contrastive_constraint_attr_mean = grad_base._aggregate_constraint_scores(
            step_contrastive_feature_attr_mean
        )
        top_contrastive_constraints_payload = grad_base._top_constraint_payload(
            step_contrastive_constraint_attr_mean, top_n=None
        )
        for group_name, score in step_constraint_attr_mean.items():
            per_constraint_attr[group_name].append(float(score))
        for group_name, score in step_contrastive_constraint_attr_mean.items():
            per_constraint_contrastive_attr[group_name].append(float(score))

        customer_candidate_mask = state.not_served.clone()
        customer_candidate_mask[:, 0] = False
        topk_metrics: Dict[int, Dict[str, float]] = {}
        with torch.no_grad():
            for k in topk_list:
                node_perturbed, _, k_eff = grad_base._perturb_topk_nodes(
                    node_features,
                    grad_base._normalize_node_scores(node_scores.detach()),
                    customer_candidate_mask,
                    k,
                )
                common_del = grad_base._build_common(node_perturbed, global_features)
                cache_del = grad_base._encode_inputs(model, node_perturbed, global_features)
                logits_del, _, _, logprobs_del, _ = grad_base._step_logits_and_mask(
                    model, cache_del, common_del, state
                )
                sel_logit_del = _safe_score_gather(logits_del, action.detach())
                sel_logprob_del = _safe_score_gather(logprobs_del, action.detach())
                action_del = logprobs_del.argmax(dim=-1)

                logit_drop = selected_logit.detach() - sel_logit_del
                logprob_drop = selected_logprob.detach() - sel_logprob_del
                flip_rate = (action_del != action.detach()).float().mean()

                logit_drop_mean = float(logit_drop.mean().item())
                logprob_drop_mean = float(logprob_drop.mean().item())
                flip_rate_mean = float(flip_rate.item())

                per_k_logit_drop[k].append(logit_drop_mean)
                per_k_logprob_drop[k].append(logprob_drop_mean)
                per_k_flip_rate[k].append(flip_rate_mean)

                topk_metrics[k] = {
                    "k_effective": int(k_eff),
                    "logit_drop_mean": logit_drop_mean,
                    "mean_logit_drop": logit_drop_mean,
                    "logprob_drop_mean": logprob_drop_mean,
                    "mean_logprob_drop": logprob_drop_mean,
                    "action_flip_rate": flip_rate_mean,
                    "mean_action_flip_rate": flip_rate_mean,
                }

        next_state = grad_base._step_update_state(
            common,
            state,
            action.detach(),
            potential_distance.detach(),
            full_mask=full_mask.detach(),
            recourse_enabled=recourse_enabled,
        )
        done_after = ~next_state.not_served.any(dim=1)

        if args.save_step_records:
            step_records.append(
                _simple_step_record(
                    step=step,
                    action=action.detach(),
                    selected_logit=selected_logit.detach(),
                    selected_logprob=selected_logprob.detach(),
                    topk_metrics=topk_metrics,
                    top_features=top_features_payload,
                    top_constraints=top_constraints_payload,
                    contrastive_alt_available_rate=float(has_alt.float().mean().item()),
                    contrastive_logit_gap_mean=grad_base._masked_mean(
                        contrastive_logit_gap.detach(), has_alt
                    ),
                    contrastive_logprob_gap_mean=grad_base._masked_mean(
                        contrastive_logprob_gap.detach(), has_alt
                    ),
                    contrastive_top_constraints=top_contrastive_constraints_payload,
                    recourse_flags=recourse_flags.detach(),
                    recourse_cost_est=recourse_cost_est.detach(),
                    action_feasible=action_feasible.detach(),
                    done_before=done_before.detach(),
                    done_after=done_after.detach(),
                )
            )

        if num_store > 0:
            step_top_nodes = top_nodes_all[:num_store].detach().cpu().tolist()
            step_top_scores = top_scores_all[:num_store].detach().cpu().tolist()
            step_top_valid = top_nodes_valid_all[:num_store].detach().cpu().tolist()
            alt_action_store = alt_action[:num_store].detach().cpu().tolist()
            has_alt_store = has_alt[:num_store].detach().cpu().tolist()
            alt_action_policy_store = (
                alt_action_policy[:num_store].detach().cpu().tolist()
            )
            has_alt_policy_store = has_alt_policy[:num_store].detach().cpu().tolist()
            alt_action_policy_feasible_store = (
                alt_action_policy_feasible[:num_store].detach().cpu().tolist()
            )
            alt_recourse_policy_store = (
                alt_recourse_flags_policy[:num_store].detach().cpu().tolist()
            )
            contrastive_logit_gap_policy_store = (
                contrastive_logit_gap_policy[:num_store].detach().cpu().tolist()
            )
            contrastive_logprob_gap_policy_store = (
                contrastive_logprob_gap_policy[:num_store].detach().cpu().tolist()
            )
            alt_feasible_store = alt_action_feasible[:num_store].detach().cpu().tolist()
            alt_recourse_store = alt_recourse_flags[:num_store].detach().cpu().tolist()
            contrastive_logit_gap_store = (
                contrastive_logit_gap[:num_store].detach().cpu().tolist()
            )
            contrastive_logprob_gap_store = (
                contrastive_logprob_gap[:num_store].detach().cpu().tolist()
            )
            action_feasible_store = action_feasible[:num_store].detach().cpu().tolist()
            recourse_store = recourse_flags[:num_store].detach().cpu().tolist()
            recourse_cost_store = recourse_cost_est[:num_store].detach().cpu().tolist()

            for i in range(num_store):
                inst_feat_scores = {
                    key: float(instance_feature_attr[key][i].item())
                    for key in instance_feature_attr
                }
                inst_top_features = grad_base._top_feature_payload(inst_feat_scores, top_n=3)
                inst_constraint_scores = grad_base._aggregate_constraint_scores(inst_feat_scores)
                inst_top_constraints = grad_base._top_constraint_payload(
                    inst_constraint_scores, top_n=None
                )

                inst_contrastive_feat_scores = {
                    key: float(instance_contrastive_feature_attr[key][i].item())
                    for key in instance_contrastive_feature_attr
                }
                inst_contrastive_constraint_scores = grad_base._aggregate_constraint_scores(
                    inst_contrastive_feat_scores
                )
                inst_top_contrastive_constraints = grad_base._top_constraint_payload(
                    inst_contrastive_constraint_scores, top_n=None
                )

                instance_traces[i]["done_before"].append(bool(done_before[i].item()))
                instance_traces[i]["done_after"].append(bool(done_after[i].item()))
                instance_traces[i]["actions"].append(int(action[i].item()))
                instance_traces[i]["top_nodes"].append(
                    [
                        int(node)
                        for node, valid in zip(step_top_nodes[i], step_top_valid[i])
                        if bool(valid)
                    ]
                )
                instance_traces[i]["top_scores"].append(
                    [
                        float(score)
                        for score, valid in zip(step_top_scores[i], step_top_valid[i])
                        if bool(valid)
                    ]
                )
                instance_traces[i]["top_nodes_decision"].append(
                    instance_traces[i]["top_nodes"][-1]
                )
                instance_traces[i]["top_scores_decision"].append(
                    instance_traces[i]["top_scores"][-1]
                )
                instance_traces[i]["top_nodes_feasibility"].append([])
                instance_traces[i]["top_scores_feasibility"].append([])
                instance_traces[i]["top_features"].append(inst_top_features)
                instance_traces[i]["top_constraints"].append(inst_top_constraints)
                if has_alt_policy_store[i]:
                    instance_traces[i]["contrastive_policy_alt_action"].append(
                        int(alt_action_policy_store[i])
                    )
                    instance_traces[i]["contrastive_policy_alt_feasible"].append(
                        bool(alt_action_policy_feasible_store[i])
                    )
                    instance_traces[i]["contrastive_policy_alt_recourse"].append(
                        bool(alt_recourse_policy_store[i])
                    )
                    instance_traces[i]["contrastive_policy_logit_gap"].append(
                        float(contrastive_logit_gap_policy_store[i])
                    )
                    instance_traces[i]["contrastive_policy_logprob_gap"].append(
                        float(contrastive_logprob_gap_policy_store[i])
                    )
                else:
                    instance_traces[i]["contrastive_policy_alt_action"].append(-1)
                    instance_traces[i]["contrastive_policy_alt_feasible"].append(False)
                    instance_traces[i]["contrastive_policy_alt_recourse"].append(False)
                    instance_traces[i]["contrastive_policy_logit_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_policy_logprob_gap"].append(float("nan"))
                if has_alt_store[i]:
                    instance_traces[i]["contrastive_feasible_alt_action"].append(
                        int(alt_action_store[i])
                    )
                    instance_traces[i]["contrastive_feasible_alt_feasible"].append(
                        bool(alt_feasible_store[i])
                    )
                    instance_traces[i]["contrastive_feasible_alt_recourse"].append(
                        bool(alt_recourse_store[i])
                    )
                    instance_traces[i]["contrastive_feasible_logit_gap"].append(
                        float(contrastive_logit_gap_store[i])
                    )
                    instance_traces[i]["contrastive_feasible_logprob_gap"].append(
                        float(contrastive_logprob_gap_store[i])
                    )
                else:
                    instance_traces[i]["contrastive_feasible_alt_action"].append(-1)
                    instance_traces[i]["contrastive_feasible_alt_feasible"].append(False)
                    instance_traces[i]["contrastive_feasible_alt_recourse"].append(False)
                    instance_traces[i]["contrastive_feasible_logit_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_feasible_logprob_gap"].append(float("nan"))
                if has_alt_store[i]:
                    instance_traces[i]["contrastive_alt_action"].append(int(alt_action_store[i]))
                    instance_traces[i]["contrastive_alt_source"].append("full_feasible")
                    instance_traces[i]["contrastive_alt_feasible"].append(
                        bool(alt_feasible_store[i])
                    )
                    instance_traces[i]["contrastive_alt_recourse"].append(
                        bool(alt_recourse_store[i])
                    )
                    instance_traces[i]["contrastive_logit_gap"].append(
                        float(contrastive_logit_gap_store[i])
                    )
                    instance_traces[i]["contrastive_logprob_gap"].append(
                        float(contrastive_logprob_gap_store[i])
                    )
                    instance_traces[i]["contrastive_top_constraints"].append(
                        inst_top_contrastive_constraints
                    )
                else:
                    instance_traces[i]["contrastive_alt_action"].append(-1)
                    instance_traces[i]["contrastive_alt_source"].append("none")
                    instance_traces[i]["contrastive_alt_feasible"].append(False)
                    instance_traces[i]["contrastive_alt_recourse"].append(False)
                    instance_traces[i]["contrastive_logit_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_logprob_gap"].append(float("nan"))
                    instance_traces[i]["contrastive_top_constraints"].append([])
                instance_traces[i]["chosen_feasible"].append(bool(action_feasible_store[i]))
                instance_traces[i]["recourse_triggered"].append(bool(recourse_store[i]))
                instance_traces[i]["recourse_cost_est"].append(float(recourse_cost_store[i]))
                state_slice = grad_base._slice_state(state, i)
                local_cf_grads = _compute_local_contrastive_feature_grads(
                    model=model,
                    grad_base_module=grad_base,
                    state=state_slice,
                    node_features=node_features[i : i + 1],
                    global_features=global_features[i : i + 1],
                    action=action[i : i + 1].detach(),
                    alt_action=alt_action[i : i + 1].detach(),
                    has_alt=has_alt[i : i + 1].detach(),
                    selected_features=selected_features,
                )
                counterfactual_payload = grad_base._propose_counterfactual(
                    model=model,
                    node_features=node_features[i : i + 1],
                    global_features=global_features[i : i + 1],
                    state=state_slice,
                    batch_index=0,
                    alt_action=int(alt_action_store[i]),
                    has_alt=bool(has_alt_store[i]),
                    contrastive_logit_gap=float(contrastive_logit_gap_store[i]),
                    contrastive_grad_by_feature=local_cf_grads,
                    full_mask=full_mask[i : i + 1].detach(),
                    recourse_enabled=recourse_enabled,
                )
                if counterfactual_payload is None:
                    counterfactual_available_history.append(0.0)
                    counterfactual_switch_history.append(0.0)
                    counterfactual_make_feasible_history.append(0.0)
                    counterfactual_approximate_history.append(0.0)
                else:
                    status = str(counterfactual_payload.get("status", "approximate"))
                    feature_name = str(counterfactual_payload.get("feature", "unknown"))
                    counterfactual_available_history.append(1.0)
                    counterfactual_switch_history.append(1.0 if status == "switch" else 0.0)
                    counterfactual_make_feasible_history.append(
                        1.0 if status == "make_feasible" else 0.0
                    )
                    counterfactual_approximate_history.append(
                        1.0 if status == "approximate" else 0.0
                    )
                    rel_delta = counterfactual_payload.get("relative_delta", None)
                    try:
                        rel_delta_val = float(rel_delta)
                    except (TypeError, ValueError):
                        rel_delta_val = float("nan")
                    if math.isfinite(rel_delta_val):
                        counterfactual_relative_delta_history.append(rel_delta_val)
                    counterfactual_by_feature[feature_name].append(1.0)
                instance_traces[i]["counterfactuals"].append(counterfactual_payload)

        state = next_state

    final_costs = state.total_distance.squeeze(1)
    trajectory_summary = grad_base._summarize_trajectory(instance_traces)

    summary = {
        "method": "integrated_gradients",
        "num_instances": int(args.num_instances),
        "num_steps": int(executed_steps),
        "done_all": not bool(state.not_served.any().item()),
        "mean_final_reward": float((-final_costs).mean().item()) if final_costs.numel() > 0 else None,
        "mean_final_cost": float(final_costs.mean().item()) if final_costs.numel() > 0 else None,
        "chosen_action_feasible_rate": grad_base._safe_mean(chosen_feasible_rate_history),
        "recourse_event_rate": grad_base._safe_mean(recourse_rate_history),
        "recourse_cost_est_mean": grad_base._safe_mean(recourse_cost_est_history),
        "deletion_faithfulness": {
            str(k): {
                "mean_logit_drop": grad_base._safe_mean(per_k_logit_drop[k]),
                "mean_logprob_drop": grad_base._safe_mean(per_k_logprob_drop[k]),
                "mean_action_flip_rate": grad_base._safe_mean(per_k_flip_rate[k]),
            }
            for k in topk_list
        },
        "feature_importance_mean": {
            key: grad_base._safe_mean(per_feature_attr[key])
            for key in sorted(per_feature_attr.keys())
        },
        "constraint_importance_mean": {
            key: grad_base._safe_mean(per_constraint_attr[key])
            for key in sorted(per_constraint_attr.keys())
        },
        "contrastive": {
            "alt_available_rate": grad_base._safe_mean(contrastive_alt_available_history),
            "mean_logit_gap": grad_base._safe_mean(contrastive_logit_gap_history),
            "mean_logprob_gap": grad_base._safe_mean(contrastive_logprob_gap_history),
            "feature_importance_mean": {
                key: grad_base._safe_mean(per_feature_contrastive_attr[key])
                for key in sorted(per_feature_contrastive_attr.keys())
            },
            "constraint_importance_mean": {
                key: grad_base._safe_mean(per_constraint_contrastive_attr[key])
                for key in sorted(per_constraint_contrastive_attr.keys())
            },
        },
        "counterfactuals": {
            "available_rate": grad_base._safe_mean(counterfactual_available_history),
            "switch_rate": grad_base._safe_mean(counterfactual_switch_history),
            "make_feasible_rate": grad_base._safe_mean(
                counterfactual_make_feasible_history
            ),
            "approximate_rate": grad_base._safe_mean(counterfactual_approximate_history),
            "mean_relative_delta": grad_base._safe_mean(
                counterfactual_relative_delta_history
            ),
            "feature_frequency": {
                key: grad_base._safe_mean(counterfactual_by_feature[key])
                for key in sorted(counterfactual_by_feature.keys())
            },
            "feature_share": {},
        },
        "trajectory": trajectory_summary,
        "instance_variant_counts": {
            key: sum(1 for meta in variant_meta if meta["code"] == key)
            for key in sorted({meta["code"] for meta in variant_meta})
        },
    }

    feat_total = sum(summary["feature_importance_mean"].values())
    summary["feature_importance_share"] = {
        key: (value / feat_total if feat_total > 0 else 0.0)
        for key, value in summary["feature_importance_mean"].items()
    }
    constraint_total = sum(summary["constraint_importance_mean"].values())
    summary["constraint_importance_share"] = {
        key: (value / constraint_total if constraint_total > 0 else 0.0)
        for key, value in summary["constraint_importance_mean"].items()
    }
    contrastive_feature_total = sum(summary["contrastive"]["feature_importance_mean"].values())
    summary["contrastive"]["feature_importance_share"] = {
        key: (value / contrastive_feature_total if contrastive_feature_total > 0 else 0.0)
        for key, value in summary["contrastive"]["feature_importance_mean"].items()
    }
    contrastive_constraint_total = sum(
        summary["contrastive"]["constraint_importance_mean"].values()
    )
    summary["contrastive"]["constraint_importance_share"] = {
        key: (value / contrastive_constraint_total if contrastive_constraint_total > 0 else 0.0)
        for key, value in summary["contrastive"]["constraint_importance_mean"].items()
    }
    cf_total = sum(summary["counterfactuals"]["feature_frequency"].values())
    summary["counterfactuals"]["feature_share"] = {
        key: (value / cf_total if cf_total > 0 else 0.0)
        for key, value in summary["counterfactuals"]["feature_frequency"].items()
    }

    run_name = repr(config)
    run_group = f"{config.problem}/{config.graph_size}"
    model_label_base = f"{run_group}/{run_name}"
    baseline_tag = str(args.ig_baseline)
    model_label = f"{model_label_base} [IG:{baseline_tag}]"
    model_slug = grad_base._slugify(f"{model_label_base}-ig-{baseline_tag}")

    report = {
        "timestamp": int(time.time()),
        "config": {
            "seed": int(args.seed) if args.seed is not None else None,
            "data_seed": int(data_seed) if data_seed is not None else None,
            "device": str(config.device),
            "num_instances": int(args.num_instances),
            "max_steps": int(args.max_steps),
            "topk_nodes": [int(k) for k in topk_list],
            "attr_features": list(selected_features),
            "constraint_groups": sorted(
                {grad_base._feature_to_constraint_group(name) for name in selected_features}
            ),
            "model_target": "mavrp.env.models.TransformerModel",
            "encoder_target": f"{config.encoder.__module__}.{config.encoder.__name__}",
            "decoder_target": f"{config.decoder.__module__}.{config.decoder.__name__}",
            "run_name": run_name,
            "run_group": run_group,
            "model_label_base": model_label_base,
            "model_label": model_label,
            "model_slug": model_slug,
            "config_id": config_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_path_resolved": str(checkpoint_path.resolve()),
            "checkpoint_kind": checkpoint_path.name,
            "model_state": "trained",
            "attribution_method": "integrated_gradients",
            "node_importance_mode": "integrated-gradients",
            "ig_steps": int(args.ig_steps),
            "ig_baseline": baseline_tag,
            "problem": config.problem,
            "graph_size": int(config.graph_size),
            "config_repr": run_name,
        },
        "summary": summary,
    }

    if args.save_step_records:
        report["steps"] = step_records
    if num_store > 0:
        for trace in instance_traces:
            done_step = None
            for i, flag in enumerate(trace["done_after"]):
                if flag:
                    done_step = i
                    break
            trace["done_step"] = done_step
            trace["top_k_effective"] = int(top_k_effective)
        report["instances"] = instance_traces

    output_path = output_dir / f"action_explainer_ig_{model_slug}_{int(time.time())}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved IG XAI report to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated-Gradients action-level explainability for g-unirouting checkpoints."
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
        help="Path to baseline.pt or checkpoint.ckpt. If omitted, uses the resolved config default baseline.pt path.",
    )
    parser.add_argument("--problem", default=None, help="Optional config.problem override.")
    parser.add_argument(
        "--graph-size", type=int, default=None, help="Optional config.graph_size override."
    )
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument(
        "--attr-features",
        default="auto",
        help="Comma-separated feature names or 'auto'.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Number of interpolation points used by Integrated Gradients.",
    )
    parser.add_argument(
        "--ig-baseline",
        choices=IG_BASELINE_MODES,
        default="mean-fill",
        help="Baseline used by Integrated Gradients.",
    )
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, mps).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Optional seed used specifically for dataset generation. "
            "Defaults to --seed. Use this to align exactly the same sampled instances "
            "across different model architectures."
        ),
    )
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument(
        "--save-step-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-step summary records.",
    )
    parser.add_argument(
        "--save-instance-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-instance traces used by plotting/text scripts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be >= 2")
    grad_base._preflight_check()
    run(args)


if __name__ == "__main__":
    main()
