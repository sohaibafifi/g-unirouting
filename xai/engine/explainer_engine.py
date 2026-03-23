"""ExplainerEngine: unified step-loop for attribution methods."""
from __future__ import annotations

import json
import math
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mavrp.configs.config import Config
    from mavrp.env.models import TransformerModel

from engine.attribution import AttributionBase
from engine.ig_attribution import compute_local_contrastive_feature_grads
from engine.decode_types import DecodeState
from engine.decode_ops import (
    build_common,
    encode_inputs,
    grad_to_instance_scores,
    grad_to_node_scores,
    init_instance_traces,
    init_state,
    infer_recourse_for_action,
    is_recourse_decoder,
    normalize_node_scores,
    parse_attr_features,
    parse_feasibility_cost_weight,
    parse_feasibility_top_m,
    parse_feasibility_weight,
    parse_topk_nodes,
    perturb_topk_nodes,
    propose_counterfactual,
    select_best_alternative,
    slice_state,
    step_logits_and_mask,
    step_update_state,
    top_nodes_and_scores,
    compute_feasibility_node_scores,
)
from domain.score_ops import (
    aggregate_constraint_scores,
    aggregate_decoder_dynamic_state_scores,
    aggregate_decoder_state_constraint_scores,
    feature_to_constraint_group as _feature_to_constraint_group,
    top_constraint_payload,
    top_decoder_dynamic_state_payload,
    top_feature_payload,
    top_state_payload,
)
from utils.math_utils import masked_mean, safe_mean
from utils.text_utils import slugify


class ExplainerEngine:
    """Runs the XAI step loop for a single attribution method.

    Args:
        model: Loaded TransformerModel (eval mode).
        config: Config object resolved from checkpoint.
        config_id: Optional index into Config.all().
        checkpoint_path: Path to the loaded checkpoint file.
        args: Parsed CLI namespace (used for num_instances, max_steps, etc.).
        attribution: Attribution method implementation.
    """

    def __init__(
        self,
        model: "TransformerModel",
        config: "Config",
        config_id: Optional[int],
        checkpoint_path: Path,
        args: Any,
        attribution: AttributionBase,
    ) -> None:
        self.model = model
        self.config = config
        self.config_id = config_id
        self.checkpoint_path = checkpoint_path
        self.args = args
        self.attribution = attribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        variant_meta: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the full step loop and return the report dict."""
        args = self.args
        model = self.model
        config = self.config
        method_key = self.attribution.method_key()
        uses_reference_baseline = self.attribution.uses_reference_baseline()
        needs_local_counterfactual_grads = (
            self.attribution.needs_local_counterfactual_grads()
        )

        self.attribution.prepare_inputs(node_features, global_features)

        topk_list = parse_topk_nodes(args.topk_nodes)
        selected_features = parse_attr_features(args.attr_features)
        recourse_enabled = is_recourse_decoder(model)

        feasibility_weight = parse_feasibility_weight(
            args.feasibility_weight, recourse_enabled
        )
        feasibility_top_m = parse_feasibility_top_m(args.feasibility_top_m)
        feasibility_cost_weight = parse_feasibility_cost_weight(
            args.feasibility_cost_weight
        )
        use_feasibility_importance = recourse_enabled and feasibility_weight > 0.0
        if method_key == "gradient":
            importance_mode = (
                "decision+feasibility" if use_feasibility_importance else "decision-only"
            )
        else:
            method_label = method_key.replace("_", "-")
            importance_mode = (
                f"{method_label}+feasibility"
                if use_feasibility_importance
                else method_label
            )

        batch_size, num_nodes = node_features.shape[:2]
        max_attr_k = max(topk_list)
        num_store = (
            min(args.num_instances, args.max_instances_to_store)
            if args.save_instance_traces
            else 0
        )
        instance_traces = (
            init_instance_traces(node_features, global_features, num_store, variant_meta)
            if num_store > 0
            else []
        )

        # Accumulators
        per_k_logit_drop: Dict[int, List[float]] = defaultdict(list)
        per_k_logprob_drop: Dict[int, List[float]] = defaultdict(list)
        per_k_flip_rate: Dict[int, List[float]] = defaultdict(list)
        per_feature_attr: Dict[str, List[float]] = defaultdict(list)
        per_feature_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
        per_constraint_attr: Dict[str, List[float]] = defaultdict(list)
        per_constraint_contrastive_attr: Dict[str, List[float]] = defaultdict(list)
        per_decoder_state_constraint_attr: Dict[str, List[float]] = defaultdict(list)
        recourse_rate_history: List[float] = []
        recourse_cost_est_history: List[float] = []
        chosen_feasible_rate_history: List[float] = []
        counterfactual_available_history: List[float] = []
        counterfactual_switch_history: List[float] = []
        counterfactual_make_feasible_history: List[float] = []
        counterfactual_approximate_history: List[float] = []
        counterfactual_relative_delta_history: List[float] = []
        counterfactual_by_feature: Dict[str, List[float]] = defaultdict(list)
        contrastive_alt_available_history: List[float] = []
        contrastive_logit_gap_history: List[float] = []
        contrastive_logprob_gap_history: List[float] = []
        decision_node_attr_history: List[float] = []
        feasibility_node_attr_history: List[float] = []

        step_records: List[Dict[str, Any]] = []
        state = init_state(build_common(node_features, global_features))
        top_k_effective = 0
        executed_steps = 0

        for step in range(args.max_steps):
            if not bool(state.not_served.any().item()):
                break
            executed_steps += 1
            done_before = ~state.not_served.any(dim=1)

            # ----------------------------------------------------------
            # Forward pass — build common/logits for action selection
            # (separate from the attribution forward pass)
            # ----------------------------------------------------------
            with torch.no_grad():
                common = build_common(node_features, global_features)
                cache = encode_inputs(model, node_features, global_features)
                logits_nograd, policy_mask, full_mask, logprobs_nograd, potential_distance = (
                    step_logits_and_mask(model, cache, common, state)
                )

            action = logprobs_nograd.argmax(dim=-1)
            selected_logit = logits_nograd.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            selected_logprob = logprobs_nograd.gather(-1, action.unsqueeze(-1)).squeeze(-1)

            # ----------------------------------------------------------
            # Alternative action selection
            # ----------------------------------------------------------
            (
                alt_action_policy,
                has_alt_policy,
                alt_logit_policy,
                alt_logprob_policy,
            ) = select_best_alternative(
                logits=logits_nograd,
                logprobs=logprobs_nograd,
                chosen_action=action,
                action_mask=~policy_mask,
            )
            (
                alt_action_full,
                has_alt_full,
                alt_logit_full,
                alt_logprob_full,
            ) = select_best_alternative(
                logits=logits_nograd,
                logprobs=logprobs_nograd,
                chosen_action=action,
                action_mask=~full_mask,
            )
            has_alt = has_alt_full | has_alt_policy
            alt_action = torch.where(has_alt_full, alt_action_full, alt_action_policy)
            alt_logit = torch.where(has_alt_full, alt_logit_full, alt_logit_policy)
            alt_logprob = torch.where(has_alt_full, alt_logprob_full, alt_logprob_policy)
            contrastive_source_code = torch.zeros_like(action)
            contrastive_source_code = torch.where(
                has_alt_policy, torch.ones_like(contrastive_source_code), contrastive_source_code
            )
            contrastive_source_code = torch.where(
                has_alt_full,
                torch.full_like(contrastive_source_code, 2),
                contrastive_source_code,
            )
            contrastive_logit_gap = selected_logit - alt_logit
            contrastive_logprob_gap = selected_logprob - alt_logprob
            contrastive_alt_available_history.append(float(has_alt.float().mean().item()))
            contrastive_logit_gap_history.append(
                masked_mean(contrastive_logit_gap.detach(), has_alt)
            )
            contrastive_logprob_gap_history.append(
                masked_mean(contrastive_logprob_gap.detach(), has_alt)
            )

            # ----------------------------------------------------------
            # Recourse / feasibility
            # ----------------------------------------------------------
            (
                recourse_flags,
                recourse_cost_est,
                action_feasible,
            ) = infer_recourse_for_action(
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
            alt_recourse_flags = (
                has_alt & recourse_enabled & (alt_action != 0) & (~alt_action_feasible)
            )
            alt_action_policy_feasible = (
                ~full_mask.gather(1, alt_action_policy.unsqueeze(-1)).squeeze(-1)
            ) & has_alt_policy
            alt_recourse_flags_policy = (
                has_alt_policy
                & recourse_enabled
                & (alt_action_policy != 0)
                & (~alt_action_policy_feasible)
            )
            alt_action_full_feasible = (
                ~full_mask.gather(1, alt_action_full.unsqueeze(-1)).squeeze(-1)
            ) & has_alt_full
            alt_recourse_flags_full = (
                has_alt_full
                & recourse_enabled
                & (alt_action_full != 0)
                & (~alt_action_full_feasible)
            )
            contrastive_logit_gap_policy = selected_logit - alt_logit_policy
            contrastive_logprob_gap_policy = selected_logprob - alt_logprob_policy
            contrastive_logit_gap_full = selected_logit - alt_logit_full
            contrastive_logprob_gap_full = selected_logprob - alt_logprob_full

            # ----------------------------------------------------------
            # Attribution
            # ----------------------------------------------------------
            grad_by_feature, contrastive_grad_by_feature, decision_node_scores = (
                self.attribution.compute_step(
                    model=model,
                    node_features=node_features,
                    global_features=global_features,
                    state=state,
                    action=action.detach(),
                    alt_action=alt_action.detach(),
                    has_alt=has_alt.detach(),
                    selected_features=selected_features,
                )
            )

            # Per-feature accumulators + instance scores
            step_feature_attr_mean: Dict[str, float] = {}
            step_contrastive_feature_attr_mean: Dict[str, float] = {}
            instance_feature_attr: Dict[str, torch.Tensor] = {}
            instance_contrastive_feature_attr: Dict[str, torch.Tensor] = {}

            for name in selected_features:
                grad = grad_by_feature.get(name)
                inst_score = grad_to_instance_scores(grad)
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

                contrastive_grad = contrastive_grad_by_feature.get(name)
                contrastive_inst_score = grad_to_instance_scores(contrastive_grad)
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

            decision_node_attr_history.append(float(decision_node_scores.mean().item()))

            # ----------------------------------------------------------
            # Feasibility importance (gradient-only)
            # ----------------------------------------------------------
            feasibility_node_scores = torch.zeros_like(decision_node_scores)
            if use_feasibility_importance:
                feasibility_node_scores = compute_feasibility_node_scores(
                    node_features=node_features.detach(),
                    global_features=global_features.detach(),
                    state=state,
                    action=action.detach(),
                    base_feasible=action_feasible.detach(),
                    base_recourse_cost=recourse_cost_est.detach(),
                    decision_node_scores=decision_node_scores.detach(),
                    top_m=feasibility_top_m,
                    cost_weight=feasibility_cost_weight,
                    recourse_enabled=recourse_enabled,
                )
                node_scores = (
                    normalize_node_scores(decision_node_scores.detach())
                    + feasibility_weight * feasibility_node_scores
                )
            else:
                node_scores = decision_node_scores

            feasibility_node_attr_history.append(
                float(feasibility_node_scores.mean().item())
            )

            # ----------------------------------------------------------
            # Top-node ranking
            # ----------------------------------------------------------
            rank_candidate_mask = state.not_served.clone()
            rank_candidate_mask[:, 0] = True
            customer_candidate_mask = state.not_served.clone()
            customer_candidate_mask[:, 0] = False

            masked_node_scores = torch.where(
                rank_candidate_mask,
                node_scores,
                torch.full_like(node_scores, float("-inf")),
            )
            top1_nodes = masked_node_scores.argmax(dim=-1)
            top_nodes_all, top_scores_all, top_nodes_valid_all, top_k_effective = (
                top_nodes_and_scores(node_scores, rank_candidate_mask, max_attr_k)
            )
            (
                top_nodes_decision_all,
                top_scores_decision_all,
                top_nodes_decision_valid_all,
                _,
            ) = top_nodes_and_scores(decision_node_scores, rank_candidate_mask, max_attr_k)
            if use_feasibility_importance:
                (
                    top_nodes_feasibility_all,
                    top_scores_feasibility_all,
                    top_nodes_feasibility_valid_all,
                    _,
                ) = top_nodes_and_scores(
                    feasibility_node_scores, rank_candidate_mask, max_attr_k
                )
            else:
                top_nodes_feasibility_all = None
                top_scores_feasibility_all = None
                top_nodes_feasibility_valid_all = None

            # ----------------------------------------------------------
            # Constraint aggregation
            # ----------------------------------------------------------
            step_solution_features = _compute_step_solution_features(
                common=common,
                state=state,
                action=action.detach(),
            )
            top_features_payload = top_feature_payload(step_feature_attr_mean, top_n=3)
            step_constraint_attr_mean = aggregate_constraint_scores(step_feature_attr_mean)
            step_contrastive_constraint_attr_mean = aggregate_constraint_scores(
                step_contrastive_feature_attr_mean
            )
            top_contrastive_constraints_payload = top_constraint_payload(
                step_contrastive_constraint_attr_mean, top_n=None
            )
            step_decoder_state_constraint_scores: List[Dict[str, float]] = []
            step_decoder_state_constraint_mean_terms: Dict[str, List[float]] = defaultdict(list)
            step_decoder_dynamic_state_mean_terms: Dict[str, List[float]] = defaultdict(list)
            for i in range(batch_size):
                inst_solution_features = {
                    key: float(values[i].detach().item())
                    for key, values in step_solution_features.items()
                }
                inst_decoder_state_scores = aggregate_decoder_state_constraint_scores(
                    solution_features=inst_solution_features,
                    variant_flags=variant_meta[i].get("flags", {}),
                )
                step_decoder_state_constraint_scores.append(inst_decoder_state_scores)
                for group_name, score in inst_decoder_state_scores.items():
                    step_decoder_state_constraint_mean_terms[group_name].append(float(score))
                inst_decoder_dynamic_state_scores = aggregate_decoder_dynamic_state_scores(
                    constraint_scores=inst_decoder_state_scores,
                    solution_features=inst_solution_features,
                    variant_flags=variant_meta[i].get("flags", {}),
                )
                for state_name, score in inst_decoder_dynamic_state_scores.items():
                    step_decoder_dynamic_state_mean_terms[state_name].append(
                        float(score)
                    )
            top_constraints_payload = top_constraint_payload(step_constraint_attr_mean, top_n=None)
            top_constraint_states_payload: List[Dict[str, float]] = []
            step_decoder_state_constraint_mean = {
                key: safe_mean(values)
                for key, values in sorted(step_decoder_state_constraint_mean_terms.items())
            }
            top_decoder_state_constraints_payload = top_constraint_payload(
                step_decoder_state_constraint_mean,
                top_n=None,
            )
            step_decoder_dynamic_state_mean = {
                key: safe_mean(values)
                for key, values in sorted(
                    step_decoder_dynamic_state_mean_terms.items()
                )
            }
            top_decoder_dynamic_states_payload = top_state_payload(
                step_decoder_dynamic_state_mean,
                top_n=None,
            )
            for group_name, score in step_constraint_attr_mean.items():
                per_constraint_attr[group_name].append(float(score))
            for group_name, score in step_contrastive_constraint_attr_mean.items():
                per_constraint_contrastive_attr[group_name].append(float(score))
            for group_name, score in step_decoder_state_constraint_mean.items():
                per_decoder_state_constraint_attr[group_name].append(float(score))

            # ----------------------------------------------------------
            # Deletion faithfulness
            # ----------------------------------------------------------
            topk_metrics: Dict[int, Dict[str, float]] = {}
            with torch.no_grad():
                for k in topk_list:
                    node_perturbed, _, k_eff = perturb_topk_nodes(
                        node_features,
                        normalize_node_scores(node_scores.detach()),
                        customer_candidate_mask,
                        k,
                    )
                    common_del = build_common(node_perturbed, global_features)
                    cache_del = encode_inputs(model, node_perturbed, global_features)
                    logits_del, _, _, logprobs_del, _ = step_logits_and_mask(
                        model, cache_del, common_del, state
                    )
                    sel_logit_del = logits_del.gather(
                        -1, action.detach().unsqueeze(-1)
                    ).squeeze(-1)
                    sel_logprob_del = logprobs_del.gather(
                        -1, action.detach().unsqueeze(-1)
                    ).squeeze(-1)
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

            # ----------------------------------------------------------
            # State transition
            # ----------------------------------------------------------
            next_state = step_update_state(
                common,
                state,
                action.detach(),
                potential_distance.detach(),
                full_mask=full_mask.detach(),
                recourse_enabled=recourse_enabled,
            )
            done_after = ~next_state.not_served.any(dim=1)
            done_ratio = float(done_after.float().mean().item())

            # ----------------------------------------------------------
            # Step record
            # ----------------------------------------------------------
            if args.save_step_records:
                step_records.append(
                    _build_step_record(
                        step=step,
                        action=action.detach(),
                        selected_logit=selected_logit.detach(),
                        selected_logprob=selected_logprob.detach(),
                        node_scores=node_scores.detach(),
                        decision_node_scores=decision_node_scores.detach(),
                        feasibility_node_scores=feasibility_node_scores.detach(),
                        use_feasibility_importance=use_feasibility_importance,
                        topk_metrics=topk_metrics,
                        top1_nodes=top1_nodes.detach(),
                        done_ratio=done_ratio,
                        top_features=top_features_payload,
                        top_constraints=top_constraints_payload,
                        top_constraint_states=top_constraint_states_payload,
                        decoder_state_constraints=top_decoder_state_constraints_payload,
                        decoder_dynamic_states=top_decoder_dynamic_states_payload,
                        decoder_state_constraint_states=(
                            top_decoder_dynamic_states_payload
                        ),
                        feature_attr_mean=step_feature_attr_mean,
                        constraint_attr_mean=step_constraint_attr_mean,
                        contrastive_alt_available_rate=float(has_alt.float().mean().item()),
                        contrastive_logit_gap_mean=masked_mean(
                            contrastive_logit_gap.detach(), has_alt
                        ),
                        contrastive_logprob_gap_mean=masked_mean(
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

            # ----------------------------------------------------------
            # Instance traces
            # ----------------------------------------------------------
            if num_store > 0:
                step_top_nodes = top_nodes_all[:num_store].detach().cpu().tolist()
                step_top_scores = top_scores_all[:num_store].detach().cpu().tolist()
                step_top_valid = top_nodes_valid_all[:num_store].detach().cpu().tolist()
                step_top_nodes_decision = (
                    top_nodes_decision_all[:num_store].detach().cpu().tolist()
                )
                step_top_scores_decision = (
                    top_scores_decision_all[:num_store].detach().cpu().tolist()
                )
                step_top_valid_decision = (
                    top_nodes_decision_valid_all[:num_store].detach().cpu().tolist()
                )
                if use_feasibility_importance:
                    step_top_nodes_feasibility = (
                        top_nodes_feasibility_all[:num_store].detach().cpu().tolist()
                        if top_nodes_feasibility_all is not None
                        else []
                    )
                    step_top_scores_feasibility = (
                        top_scores_feasibility_all[:num_store].detach().cpu().tolist()
                        if top_scores_feasibility_all is not None
                        else []
                    )
                    step_top_valid_feasibility = (
                        top_nodes_feasibility_valid_all[:num_store].detach().cpu().tolist()
                        if top_nodes_feasibility_valid_all is not None
                        else []
                    )
                else:
                    step_top_nodes_feasibility = []
                    step_top_scores_feasibility = []
                    step_top_valid_feasibility = []

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
                alt_action_full_store = alt_action_full[:num_store].detach().cpu().tolist()
                has_alt_full_store = has_alt_full[:num_store].detach().cpu().tolist()
                alt_action_full_feasible_store = (
                    alt_action_full_feasible[:num_store].detach().cpu().tolist()
                )
                alt_recourse_full_store = (
                    alt_recourse_flags_full[:num_store].detach().cpu().tolist()
                )
                contrastive_logit_gap_full_store = (
                    contrastive_logit_gap_full[:num_store].detach().cpu().tolist()
                )
                contrastive_logprob_gap_full_store = (
                    contrastive_logprob_gap_full[:num_store].detach().cpu().tolist()
                )
                contrastive_logit_gap_store = (
                    contrastive_logit_gap[:num_store].detach().cpu().tolist()
                )
                contrastive_logprob_gap_store = (
                    contrastive_logprob_gap[:num_store].detach().cpu().tolist()
                )
                alt_source_store = contrastive_source_code[:num_store].detach().cpu().tolist()
                alt_feasible_store = alt_action_feasible[:num_store].detach().cpu().tolist()
                alt_recourse_store = alt_recourse_flags[:num_store].detach().cpu().tolist()
                action_feasible_store = action_feasible[:num_store].detach().cpu().tolist()
                recourse_store = recourse_flags[:num_store].detach().cpu().tolist()
                recourse_cost_store = recourse_cost_est[:num_store].detach().cpu().tolist()
                step_solution_feature_store = {
                    key: values[:num_store].detach().cpu().tolist()
                    for key, values in step_solution_features.items()
                }

                for i in range(num_store):
                    inst_feat_scores = {
                        key: float(instance_feature_attr[key][i].item())
                        for key in instance_feature_attr
                    }
                    inst_top_features = top_feature_payload(inst_feat_scores, top_n=3)
                    inst_constraint_scores = aggregate_constraint_scores(inst_feat_scores)
                    inst_top_constraints = top_constraint_payload(
                        inst_constraint_scores, top_n=None
                    )
                    inst_solution_features = {
                        key: float(values[i])
                        for key, values in step_solution_feature_store.items()
                    }
                    inst_top_constraint_states: List[Dict[str, float]] = []
                    inst_contrastive_feat_scores = {
                        key: float(instance_contrastive_feature_attr[key][i].item())
                        for key in instance_contrastive_feature_attr
                    }
                    inst_contrastive_constraint_scores = aggregate_constraint_scores(
                        inst_contrastive_feat_scores
                    )
                    inst_top_contrastive_constraints = top_constraint_payload(
                        inst_contrastive_constraint_scores, top_n=None
                    )
                    inst_decoder_state_constraints = top_constraint_payload(
                        step_decoder_state_constraint_scores[i],
                        top_n=None,
                    )
                    inst_decoder_dynamic_states = top_decoder_dynamic_state_payload(
                        constraint_scores=step_decoder_state_constraint_scores[i],
                        solution_features=inst_solution_features,
                        variant_flags=variant_meta[i].get("flags", {}),
                        top_n=None,
                    )

                    # Counterfactual grads: local non-IG for IG, batch slice for gradient
                    if needs_local_counterfactual_grads:
                        state_slice = slice_state(state, i)
                        cf_grads = compute_local_contrastive_feature_grads(
                            model=model,
                            state=state_slice,
                            node_features=node_features[i : i + 1],
                            global_features=global_features[i : i + 1],
                            action=action[i : i + 1].detach(),
                            alt_action=alt_action[i : i + 1].detach(),
                            has_alt=has_alt[i : i + 1].detach(),
                            selected_features=selected_features,
                        )
                        cf_batch_index = 0
                        cf_node_features = node_features[i : i + 1]
                        cf_global_features = global_features[i : i + 1]
                        cf_state = state_slice
                        cf_full_mask = full_mask[i : i + 1].detach()
                    else:
                        cf_grads = contrastive_grad_by_feature
                        cf_batch_index = i
                        cf_node_features = node_features
                        cf_global_features = global_features
                        cf_state = state
                        cf_full_mask = full_mask.detach()

                    counterfactual_payload = propose_counterfactual(
                        model=model,
                        node_features=cf_node_features,
                        global_features=cf_global_features,
                        state=cf_state,
                        batch_index=cf_batch_index,
                        alt_action=int(alt_action_store[i]),
                        has_alt=bool(has_alt_store[i]),
                        contrastive_logit_gap=float(contrastive_logit_gap_store[i]),
                        contrastive_grad_by_feature=cf_grads,
                        full_mask=cf_full_mask,
                        recourse_enabled=recourse_enabled,
                    )

                    _accumulate_counterfactual(
                        counterfactual_payload,
                        counterfactual_available_history,
                        counterfactual_switch_history,
                        counterfactual_make_feasible_history,
                        counterfactual_approximate_history,
                        counterfactual_relative_delta_history,
                        counterfactual_by_feature,
                    )

                    _update_instance_trace(
                        trace=instance_traces[i],
                        step=step,
                        action=action,
                        done_before=done_before,
                        done_after=done_after,
                        action_feasible_store=action_feasible_store,
                        recourse_store=recourse_store,
                        recourse_cost_store=recourse_cost_store,
                        step_top_nodes=step_top_nodes,
                        step_top_scores=step_top_scores,
                        step_top_valid=step_top_valid,
                        step_top_nodes_decision=step_top_nodes_decision,
                        step_top_scores_decision=step_top_scores_decision,
                        step_top_valid_decision=step_top_valid_decision,
                        step_top_nodes_feasibility=step_top_nodes_feasibility,
                        step_top_scores_feasibility=step_top_scores_feasibility,
                        step_top_valid_feasibility=step_top_valid_feasibility,
                        use_feasibility_importance=use_feasibility_importance,
                        inst_top_features=inst_top_features,
                        inst_top_constraints=inst_top_constraints,
                        inst_top_constraint_states=inst_top_constraint_states,
                        inst_decoder_state_constraints=inst_decoder_state_constraints,
                        inst_decoder_dynamic_states=inst_decoder_dynamic_states,
                        inst_decoder_state_constraint_states=(
                            inst_decoder_dynamic_states
                        ),
                        has_alt_policy_store=has_alt_policy_store,
                        alt_action_policy_store=alt_action_policy_store,
                        alt_action_policy_feasible_store=alt_action_policy_feasible_store,
                        alt_recourse_policy_store=alt_recourse_policy_store,
                        contrastive_logit_gap_policy_store=contrastive_logit_gap_policy_store,
                        contrastive_logprob_gap_policy_store=contrastive_logprob_gap_policy_store,
                        has_alt_full_store=has_alt_full_store,
                        alt_action_full_store=alt_action_full_store,
                        alt_action_full_feasible_store=alt_action_full_feasible_store,
                        alt_recourse_full_store=alt_recourse_full_store,
                        contrastive_logit_gap_full_store=contrastive_logit_gap_full_store,
                        contrastive_logprob_gap_full_store=contrastive_logprob_gap_full_store,
                        has_alt_store=has_alt_store,
                        alt_action_store=alt_action_store,
                        alt_source_store=alt_source_store,
                        alt_feasible_store=alt_feasible_store,
                        alt_recourse_store=alt_recourse_store,
                        contrastive_logit_gap_store=contrastive_logit_gap_store,
                        contrastive_logprob_gap_store=contrastive_logprob_gap_store,
                        inst_top_contrastive_constraints=inst_top_contrastive_constraints,
                        counterfactual_payload=counterfactual_payload,
                        step_solution_feature_store=step_solution_feature_store,
                        idx=i,
                    )

            state = next_state

        # ------------------------------------------------------------------
        # Build summary + report
        # ------------------------------------------------------------------
        from engine.decode_ops import summarize_trajectory

        final_costs = state.total_distance.squeeze(1)
        done_all = not bool(state.not_served.any().item())
        trajectory_summary = summarize_trajectory(instance_traces)

        summary = _build_summary(
            args=args,
            executed_steps=executed_steps,
            done_all=done_all,
            final_costs=final_costs,
            chosen_feasible_rate_history=chosen_feasible_rate_history,
            recourse_rate_history=recourse_rate_history,
            recourse_cost_est_history=recourse_cost_est_history,
            per_k_logit_drop=per_k_logit_drop,
            per_k_logprob_drop=per_k_logprob_drop,
            per_k_flip_rate=per_k_flip_rate,
            per_feature_attr=per_feature_attr,
            per_constraint_attr=per_constraint_attr,
            per_decoder_state_constraint_attr=per_decoder_state_constraint_attr,
            contrastive_alt_available_history=contrastive_alt_available_history,
            contrastive_logit_gap_history=contrastive_logit_gap_history,
            contrastive_logprob_gap_history=contrastive_logprob_gap_history,
            per_feature_contrastive_attr=per_feature_contrastive_attr,
            per_constraint_contrastive_attr=per_constraint_contrastive_attr,
            importance_mode=importance_mode,
            feasibility_weight=feasibility_weight,
            feasibility_top_m=feasibility_top_m,
            feasibility_cost_weight=feasibility_cost_weight,
            decision_node_attr_history=decision_node_attr_history,
            feasibility_node_attr_history=feasibility_node_attr_history,
            counterfactual_available_history=counterfactual_available_history,
            counterfactual_switch_history=counterfactual_switch_history,
            counterfactual_make_feasible_history=counterfactual_make_feasible_history,
            counterfactual_approximate_history=counterfactual_approximate_history,
            counterfactual_relative_delta_history=counterfactual_relative_delta_history,
            counterfactual_by_feature=counterfactual_by_feature,
            trajectory_summary=trajectory_summary,
            variant_meta=variant_meta,
            topk_list=topk_list,
            method_key=method_key,
        )

        report = self._build_report(
            args=args,
            summary=summary,
            step_records=step_records,
            instance_traces=instance_traces,
            top_k_effective=top_k_effective,
            num_store=num_store,
            selected_features=selected_features,
            topk_list=topk_list,
            importance_mode=importance_mode,
            feasibility_weight=feasibility_weight,
            feasibility_top_m=feasibility_top_m,
            feasibility_cost_weight=feasibility_cost_weight,
            method_key=method_key,
        )
        return report

    def save_report(self, report: Dict[str, Any], output_dir: Path) -> Path:
        """Write report JSON to output_dir and return the path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = report.get("config", {})
        model_slug = str(cfg.get("model_slug", "unknown"))
        method_key = str(cfg.get("attribution_method", "gradient")).strip().lower()
        randomized = bool(cfg.get("randomize_weights", False))
        random_suffix = "_randomized" if randomized else ""
        prefix = {
            "integrated_gradients": "action_explainer_ig_",
            "deeplift": "action_explainer_deeplift_",
        }.get(method_key, "action_explainer_")
        output_path = output_dir / f"{prefix}{model_slug}{random_suffix}_{int(time.time())}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        label = {
            "integrated_gradients": "IG XAI",
            "deeplift": "DeepLIFT XAI",
        }.get(method_key, "XAI")
        print(f"Saved {label} report to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(
        self,
        args: Any,
        summary: Dict[str, Any],
        step_records: List[Dict[str, Any]],
        instance_traces: List[Dict[str, Any]],
        top_k_effective: int,
        num_store: int,
        selected_features: List[str],
        topk_list: List[int],
        importance_mode: str,
        feasibility_weight: float,
        feasibility_top_m: int,
        feasibility_cost_weight: float,
        method_key: str,
    ) -> Dict[str, Any]:
        config = self.config
        config_id = self.config_id
        checkpoint_path = self.checkpoint_path
        data_seed = getattr(args, "data_seed", None)
        if data_seed is None:
            data_seed = getattr(args, "seed", None)

        run_name = repr(config)
        run_group = f"{config.problem}/{config.graph_size}"
        model_label_base = f"{run_group}/{run_name}"

        if self.attribution.uses_reference_baseline():
            baseline_tag = str(
                getattr(
                    args,
                    "deeplift_baseline",
                    getattr(args, "ig_baseline", "mean-fill"),
                )
            )
            method_tag = {
                "integrated_gradients": "IG",
                "deeplift": "DeepLift",
            }.get(method_key, method_key)
            model_label = f"{model_label_base} [{method_tag}:{baseline_tag}]"
            model_slug = slugify(f"{model_label_base}-{method_key}-{baseline_tag}")
        else:
            model_label = f"{run_group}/{run_name}"
            model_slug = slugify(model_label)
            baseline_tag = None

        config_dict: Dict[str, Any] = {
            "seed": int(args.seed) if getattr(args, "seed", None) is not None else None,
            "data_seed": int(data_seed) if data_seed is not None else None,
            "device": str(config.device),
            "num_instances": int(args.num_instances),
            "max_steps": int(args.max_steps),
            "topk_nodes": [int(k) for k in topk_list],
            "attr_features": list(selected_features),
            "constraint_groups": sorted(
                {_feature_to_constraint_group(name) for name in selected_features}
            ),
            "instance_variant_encoding": (
                "['o' if open_route] + vrp + ['m' if mixed_backhaul]"
                " + ['b' if backhaul] + ['l' if distance_limit] + ['tw' if time_windows]"
            ),
            "model_target": "mavrp.env.models.TransformerModel",
            "encoder_target": f"{config.encoder.__module__}.{config.encoder.__name__}",
            "decoder_target": f"{config.decoder.__module__}.{config.decoder.__name__}",
            "run_name": run_name,
            "run_group": run_group,
            "model_slug": model_slug,
            "config_id": config_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_path_resolved": str(checkpoint_path.resolve()),
            "checkpoint_kind": checkpoint_path.name,
            "randomize_weights": bool(getattr(args, "randomize_weights", False)),
            "model_state": (
                "randomized" if getattr(args, "randomize_weights", False) else "trained"
            ),
            "problem": config.problem,
            "graph_size": int(config.graph_size),
            "config_repr": run_name,
        }

        config_dict["model_label"] = model_label
        config_dict["attribution_method"] = method_key
        config_dict["node_importance_mode"] = importance_mode
        config_dict["feasibility_weight"] = float(feasibility_weight)
        config_dict["feasibility_top_m"] = int(feasibility_top_m)
        config_dict["feasibility_cost_weight"] = float(feasibility_cost_weight)
        if baseline_tag is not None:
            config_dict["model_label_base"] = model_label_base
            config_dict["reference_baseline"] = baseline_tag
            if method_key == "integrated_gradients":
                config_dict["ig_steps"] = int(getattr(args, "ig_steps", 50))
                config_dict["ig_baseline"] = baseline_tag
            elif method_key == "deeplift":
                config_dict["deeplift_baseline"] = baseline_tag

        config_dict.update(self.attribution.summary())

        report: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "config": config_dict,
            "summary": summary,
        }

        if args.save_step_records:
            report["steps"] = step_records
        if num_store > 0:
            for trace in instance_traces:
                done_step = None
                for idx, flag in enumerate(trace["done_after"]):
                    if flag:
                        done_step = idx
                        break
                trace["done_step"] = done_step
                trace["top_k_effective"] = int(top_k_effective)
            report["instances"] = instance_traces

        return report


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the class interface)
# ---------------------------------------------------------------------------


def _build_step_record(
    step: int,
    action: torch.Tensor,
    selected_logit: torch.Tensor,
    selected_logprob: torch.Tensor,
    node_scores: torch.Tensor,
    decision_node_scores: torch.Tensor,
    feasibility_node_scores: torch.Tensor,
    use_feasibility_importance: bool,
    topk_metrics: Dict[int, Dict[str, float]],
    top1_nodes: torch.Tensor,
    done_ratio: float,
    top_features: List[Dict[str, float]],
    top_constraints: List[Dict[str, float]],
    top_constraint_states: List[Dict[str, float]],
    decoder_state_constraints: List[Dict[str, float]],
    decoder_dynamic_states: List[Dict[str, float]],
    decoder_state_constraint_states: List[Dict[str, float]],
    feature_attr_mean: Dict[str, float],
    constraint_attr_mean: Dict[str, float],
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
        "done_ratio": float(done_ratio),
        "actions": action.detach().cpu().tolist(),
        "selected_logit_mean": float(selected_logit.mean().item()),
        "selected_logprob_mean": float(selected_logprob.mean().item()),
        "mean_node_attr": float(node_scores.mean().item()),
        "mean_node_attr_decision": float(decision_node_scores.mean().item()),
        "mean_node_attr_feasibility": float(feasibility_node_scores.mean().item()),
        "use_feasibility_importance": bool(use_feasibility_importance),
        "top1_nodes": top1_nodes.detach().cpu().tolist(),
        "top_features": top_features,
        "top_constraints": top_constraints,
        "top_constraint_states": top_constraint_states,
        "decoder_state_constraints": decoder_state_constraints,
        "decoder_dynamic_states": decoder_dynamic_states,
        "decoder_state_constraint_states": decoder_state_constraint_states,
        "feature_attr_mean": feature_attr_mean,
        "constraint_attr_mean": constraint_attr_mean,
        "contrastive": {
            "alt_available_rate": float(contrastive_alt_available_rate),
            "mean_logit_gap": float(contrastive_logit_gap_mean),
            "mean_logprob_gap": float(contrastive_logprob_gap_mean),
            "top_constraints": contrastive_top_constraints,
        },
        "recourse_rate": float(recourse_flags.float().mean().item()),
        "recourse_flags": recourse_flags.detach().cpu().tolist(),
        "recourse_cost_est_mean": float(recourse_cost_est.mean().item()),
        "chosen_action_feasible_rate": float(action_feasible.float().mean().item()),
        "done_before": done_before.detach().cpu().tolist(),
        "done_after": done_after.detach().cpu().tolist(),
        "deletion": {str(k): m for k, m in topk_metrics.items()},
    }


def _compute_step_solution_features(
    common: Dict[str, torch.Tensor],
    state: DecodeState,
    action: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    batch_indices = torch.arange(action.size(0), device=action.device)
    current_nodes = state.current_node

    leave_time = state.leave_time.squeeze(-1)
    route_distance = state.distance.squeeze(-1)
    deliveries = state.deliveries.squeeze(-1)
    pickups = state.pickups.squeeze(-1)
    capacities = common["capacities"].squeeze(-1)
    distance_limits = common["distance_limits"].squeeze(-1)
    time_limits = common["time_limits"].squeeze(-1)

    travel_distance = common["deltas"][batch_indices, current_nodes, action]
    arrival_time = leave_time + travel_distance
    earliest_start = common["earliest_start_time"][batch_indices, action]
    latest_start = common["latest_start_time"][batch_indices, action]
    start_time = torch.maximum(arrival_time, earliest_start)
    wait_time = (start_time - arrival_time).clamp_min(0.0)
    tw_slack = latest_start - start_time
    tw_width = latest_start - earliest_start
    selected_service_time = common["services"][batch_indices, action]

    is_customer = action > 0

    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(num, float("nan"))
        valid = (
            torch.isfinite(num)
            & torch.isfinite(den)
            & (den.abs() > 1e-8)
        )
        out[valid] = num[valid] / den[valid]
        return out

    distance_budget_slack = distance_limits - route_distance
    depot_time_budget_slack = time_limits - leave_time

    tw_slack = torch.where(
        is_customer,
        tw_slack,
        torch.full_like(tw_slack, float("nan")),
    )
    tw_width = torch.where(
        is_customer,
        tw_width,
        torch.full_like(tw_width, float("nan")),
    )
    selected_service_time = torch.where(
        is_customer,
        selected_service_time,
        torch.full_like(selected_service_time, float("nan")),
    )
    wait_time = torch.where(
        is_customer,
        wait_time,
        torch.full_like(wait_time, float("nan")),
    )

    return {
        "current_time": leave_time,
        "current_route_length": route_distance,
        "current_route_length_norm": _safe_div(route_distance, distance_limits),
        "used_capacity_linehaul_share": _safe_div(deliveries, capacities),
        "used_capacity_backhaul_share": _safe_div(pickups, capacities),
        "distance_budget_slack": distance_budget_slack,
        "distance_budget_slack_norm": _safe_div(distance_budget_slack, distance_limits),
        "depot_time_budget_slack": depot_time_budget_slack,
        "depot_time_budget_slack_norm": _safe_div(depot_time_budget_slack, time_limits),
        "selected_travel_distance": travel_distance,
        "selected_service_time": selected_service_time,
        "selected_wait_time": wait_time,
        "selected_tw_slack": tw_slack,
        "selected_tw_slack_norm": _safe_div(tw_slack, tw_width),
        "selected_is_customer": is_customer.to(route_distance.dtype),
    }


def _accumulate_counterfactual(
    payload: Optional[Dict[str, Any]],
    available_history: List[float],
    switch_history: List[float],
    make_feasible_history: List[float],
    approximate_history: List[float],
    relative_delta_history: List[float],
    by_feature: Dict[str, List[float]],
) -> None:
    if payload is None:
        available_history.append(0.0)
        switch_history.append(0.0)
        make_feasible_history.append(0.0)
        approximate_history.append(0.0)
        return
    status = str(payload.get("status", "approximate"))
    feature_name = str(payload.get("feature", "unknown"))
    available_history.append(1.0)
    switch_history.append(1.0 if status == "switch" else 0.0)
    make_feasible_history.append(1.0 if status == "make_feasible" else 0.0)
    approximate_history.append(1.0 if status == "approximate" else 0.0)
    rel_delta = payload.get("relative_delta", None)
    if rel_delta is not None:
        try:
            rel_delta_val = float(rel_delta)
        except (TypeError, ValueError):
            rel_delta_val = float("nan")
        if math.isfinite(rel_delta_val):
            relative_delta_history.append(rel_delta_val)
    by_feature[feature_name].append(1.0)


def _update_instance_trace(
    trace: Dict[str, Any],
    idx: int,
    step: int,
    action: torch.Tensor,
    done_before: torch.Tensor,
    done_after: torch.Tensor,
    action_feasible_store: List,
    recourse_store: List,
    recourse_cost_store: List,
    step_top_nodes: List,
    step_top_scores: List,
    step_top_valid: List,
    step_top_nodes_decision: List,
    step_top_scores_decision: List,
    step_top_valid_decision: List,
    step_top_nodes_feasibility: List,
    step_top_scores_feasibility: List,
    step_top_valid_feasibility: List,
    use_feasibility_importance: bool,
    inst_top_features: List,
    inst_top_constraints: List,
    inst_top_constraint_states: List,
    inst_decoder_state_constraints: List,
    inst_decoder_dynamic_states: List,
    inst_decoder_state_constraint_states: List,
    has_alt_policy_store: List,
    alt_action_policy_store: List,
    alt_action_policy_feasible_store: List,
    alt_recourse_policy_store: List,
    contrastive_logit_gap_policy_store: List,
    contrastive_logprob_gap_policy_store: List,
    has_alt_full_store: List,
    alt_action_full_store: List,
    alt_action_full_feasible_store: List,
    alt_recourse_full_store: List,
    contrastive_logit_gap_full_store: List,
    contrastive_logprob_gap_full_store: List,
    has_alt_store: List,
    alt_action_store: List,
    alt_source_store: List,
    alt_feasible_store: List,
    alt_recourse_store: List,
    contrastive_logit_gap_store: List,
    contrastive_logprob_gap_store: List,
    inst_top_contrastive_constraints: List,
    counterfactual_payload: Optional[Dict[str, Any]],
    step_solution_feature_store: Dict[str, List],
) -> None:
    i = idx
    trace["done_before"].append(bool(done_before[i].item()))
    trace["done_after"].append(bool(done_after[i].item()))
    trace["actions"].append(int(action[i].item()))
    trace["top_nodes"].append(
        [
            int(node)
            for node, valid in zip(step_top_nodes[i], step_top_valid[i])
            if bool(valid)
        ]
    )
    trace["top_scores"].append(
        [
            float(score)
            for score, valid in zip(step_top_scores[i], step_top_valid[i])
            if bool(valid)
        ]
    )
    trace["top_nodes_decision"].append(
        [
            int(node)
            for node, valid in zip(step_top_nodes_decision[i], step_top_valid_decision[i])
            if bool(valid)
        ]
    )
    trace["top_scores_decision"].append(
        [
            float(score)
            for score, valid in zip(step_top_scores_decision[i], step_top_valid_decision[i])
            if bool(valid)
        ]
    )
    if use_feasibility_importance and step_top_nodes_feasibility:
        trace["top_nodes_feasibility"].append(
            [
                int(node)
                for node, valid in zip(
                    step_top_nodes_feasibility[i], step_top_valid_feasibility[i]
                )
                if bool(valid)
            ]
        )
        trace["top_scores_feasibility"].append(
            [
                float(score)
                for score, valid in zip(
                    step_top_scores_feasibility[i], step_top_valid_feasibility[i]
                )
                if bool(valid)
            ]
        )
    else:
        trace["top_nodes_feasibility"].append([])
        trace["top_scores_feasibility"].append([])

    trace["top_features"].append(inst_top_features)
    trace["top_constraints"].append(inst_top_constraints)
    trace["top_constraint_states"].append(inst_top_constraint_states)
    trace["decoder_state_constraints"].append(inst_decoder_state_constraints)
    trace["decoder_dynamic_states"].append(inst_decoder_dynamic_states)
    trace["decoder_state_constraint_states"].append(inst_decoder_state_constraint_states)

    if has_alt_policy_store[i]:
        trace["contrastive_policy_alt_action"].append(int(alt_action_policy_store[i]))
        trace["contrastive_policy_alt_feasible"].append(
            bool(alt_action_policy_feasible_store[i])
        )
        trace["contrastive_policy_alt_recourse"].append(bool(alt_recourse_policy_store[i]))
        trace["contrastive_policy_logit_gap"].append(
            float(contrastive_logit_gap_policy_store[i])
        )
        trace["contrastive_policy_logprob_gap"].append(
            float(contrastive_logprob_gap_policy_store[i])
        )
    else:
        trace["contrastive_policy_alt_action"].append(-1)
        trace["contrastive_policy_alt_feasible"].append(False)
        trace["contrastive_policy_alt_recourse"].append(False)
        trace["contrastive_policy_logit_gap"].append(float("nan"))
        trace["contrastive_policy_logprob_gap"].append(float("nan"))

    if has_alt_full_store[i]:
        trace["contrastive_feasible_alt_action"].append(int(alt_action_full_store[i]))
        trace["contrastive_feasible_alt_feasible"].append(
            bool(alt_action_full_feasible_store[i])
        )
        trace["contrastive_feasible_alt_recourse"].append(bool(alt_recourse_full_store[i]))
        trace["contrastive_feasible_logit_gap"].append(
            float(contrastive_logit_gap_full_store[i])
        )
        trace["contrastive_feasible_logprob_gap"].append(
            float(contrastive_logprob_gap_full_store[i])
        )
    else:
        trace["contrastive_feasible_alt_action"].append(-1)
        trace["contrastive_feasible_alt_feasible"].append(False)
        trace["contrastive_feasible_alt_recourse"].append(False)
        trace["contrastive_feasible_logit_gap"].append(float("nan"))
        trace["contrastive_feasible_logprob_gap"].append(float("nan"))

    if has_alt_store[i]:
        trace["contrastive_alt_action"].append(int(alt_action_store[i]))
        if alt_source_store[i] == 2:
            contrastive_source = "full_feasible"
        elif alt_source_store[i] == 1:
            contrastive_source = "policy_masked"
        else:
            contrastive_source = "none"
        trace["contrastive_alt_source"].append(contrastive_source)
        trace["contrastive_alt_feasible"].append(bool(alt_feasible_store[i]))
        trace["contrastive_alt_recourse"].append(bool(alt_recourse_store[i]))
        trace["contrastive_logit_gap"].append(float(contrastive_logit_gap_store[i]))
        trace["contrastive_logprob_gap"].append(float(contrastive_logprob_gap_store[i]))
        trace["contrastive_top_constraints"].append(inst_top_contrastive_constraints)
    else:
        trace["contrastive_alt_action"].append(-1)
        trace["contrastive_alt_source"].append("none")
        trace["contrastive_alt_feasible"].append(False)
        trace["contrastive_alt_recourse"].append(False)
        trace["contrastive_logit_gap"].append(float("nan"))
        trace["contrastive_logprob_gap"].append(float("nan"))
        trace["contrastive_top_constraints"].append([])

    trace["chosen_feasible"].append(bool(action_feasible_store[i]))
    trace["recourse_triggered"].append(bool(recourse_store[i]))
    trace["recourse_cost_est"].append(float(recourse_cost_store[i]))
    trace["counterfactuals"].append(counterfactual_payload)
    if isinstance(trace.get("solution_features", None), dict):
        for key, values in step_solution_feature_store.items():
            if not isinstance(values, list) or i >= len(values):
                continue
            series = trace["solution_features"].setdefault(key, [])
            try:
                series.append(float(values[i]))
            except (TypeError, ValueError):
                series.append(float("nan"))


def _build_summary(
    args: Any,
    executed_steps: int,
    done_all: bool,
    final_costs: torch.Tensor,
    chosen_feasible_rate_history: List[float],
    recourse_rate_history: List[float],
    recourse_cost_est_history: List[float],
    per_k_logit_drop: Dict[int, List[float]],
    per_k_logprob_drop: Dict[int, List[float]],
    per_k_flip_rate: Dict[int, List[float]],
    per_feature_attr: Dict[str, List[float]],
    per_constraint_attr: Dict[str, List[float]],
    per_decoder_state_constraint_attr: Dict[str, List[float]],
    contrastive_alt_available_history: List[float],
    contrastive_logit_gap_history: List[float],
    contrastive_logprob_gap_history: List[float],
    per_feature_contrastive_attr: Dict[str, List[float]],
    per_constraint_contrastive_attr: Dict[str, List[float]],
    importance_mode: str,
    feasibility_weight: float,
    feasibility_top_m: int,
    feasibility_cost_weight: float,
    decision_node_attr_history: List[float],
    feasibility_node_attr_history: List[float],
    counterfactual_available_history: List[float],
    counterfactual_switch_history: List[float],
    counterfactual_make_feasible_history: List[float],
    counterfactual_approximate_history: List[float],
    counterfactual_relative_delta_history: List[float],
    counterfactual_by_feature: Dict[str, List[float]],
    trajectory_summary: Dict[str, Any],
    variant_meta: List[Dict[str, Any]],
    topk_list: List[int],
    method_key: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_instances": int(args.num_instances),
        "num_steps": int(executed_steps),
        "done_all": bool(done_all),
        "mean_final_reward": (
            float((-final_costs).mean().item()) if final_costs.numel() > 0 else None
        ),
        "mean_final_cost": (
            float(final_costs.mean().item()) if final_costs.numel() > 0 else None
        ),
        "chosen_action_feasible_rate": safe_mean(chosen_feasible_rate_history),
        "recourse_event_rate": safe_mean(recourse_rate_history),
        "recourse_cost_est_mean": safe_mean(recourse_cost_est_history),
        "deletion_faithfulness": {
            str(k): {
                "mean_logit_drop": safe_mean(per_k_logit_drop[k]),
                "mean_logprob_drop": safe_mean(per_k_logprob_drop[k]),
                "mean_action_flip_rate": safe_mean(per_k_flip_rate[k]),
            }
            for k in topk_list
        },
        "feature_importance_mean": {
            key: safe_mean(per_feature_attr[key]) for key in sorted(per_feature_attr.keys())
        },
        "constraint_importance_mean": {
            key: safe_mean(per_constraint_attr[key])
            for key in sorted(per_constraint_attr.keys())
        },
        "decoder_state_constraint_importance_mean": {
            key: safe_mean(per_decoder_state_constraint_attr[key])
            for key in sorted(per_decoder_state_constraint_attr.keys())
        },
        "contrastive": {
            "alt_available_rate": safe_mean(contrastive_alt_available_history),
            "mean_logit_gap": safe_mean(contrastive_logit_gap_history),
            "mean_logprob_gap": safe_mean(contrastive_logprob_gap_history),
            "feature_importance_mean": {
                key: safe_mean(per_feature_contrastive_attr[key])
                for key in sorted(per_feature_contrastive_attr.keys())
            },
            "constraint_importance_mean": {
                key: safe_mean(per_constraint_contrastive_attr[key])
                for key in sorted(per_constraint_contrastive_attr.keys())
            },
        },
        "node_importance": {
            "mode": importance_mode,
            "feasibility_weight": float(feasibility_weight),
            "feasibility_top_m": int(feasibility_top_m),
            "feasibility_cost_weight": float(feasibility_cost_weight),
            "decision_mean": safe_mean(decision_node_attr_history),
            "feasibility_mean": safe_mean(feasibility_node_attr_history),
        },
        "counterfactuals": {
            "available_rate": safe_mean(counterfactual_available_history),
            "switch_rate": safe_mean(counterfactual_switch_history),
            "make_feasible_rate": safe_mean(counterfactual_make_feasible_history),
            "approximate_rate": safe_mean(counterfactual_approximate_history),
            "mean_relative_delta": safe_mean(counterfactual_relative_delta_history),
            "feature_frequency": {
                key: safe_mean(counterfactual_by_feature[key])
                for key in sorted(counterfactual_by_feature.keys())
            },
        },
        "trajectory": trajectory_summary,
        "instance_variant_counts": {
            key: sum(1 for meta in variant_meta if meta["code"] == key)
            for key in sorted({meta["code"] for meta in variant_meta})
        },
    }

    summary["method"] = method_key

    # Normalise shares
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

    decoder_state_constraint_total = sum(
        summary["decoder_state_constraint_importance_mean"].values()
    )
    summary["decoder_state_constraint_importance_share"] = {
        key: (
            value / decoder_state_constraint_total
            if decoder_state_constraint_total > 0
            else 0.0
        )
        for key, value in summary["decoder_state_constraint_importance_mean"].items()
    }

    contrastive_feature_total = sum(
        summary["contrastive"]["feature_importance_mean"].values()
    )
    summary["contrastive"]["feature_importance_share"] = {
        key: (value / contrastive_feature_total if contrastive_feature_total > 0 else 0.0)
        for key, value in summary["contrastive"]["feature_importance_mean"].items()
    }

    contrastive_constraint_total = sum(
        summary["contrastive"]["constraint_importance_mean"].values()
    )
    summary["contrastive"]["constraint_importance_share"] = {
        key: (
            value / contrastive_constraint_total if contrastive_constraint_total > 0 else 0.0
        )
        for key, value in summary["contrastive"]["constraint_importance_mean"].items()
    }

    cf_total = sum(summary["counterfactuals"]["feature_frequency"].values())
    summary["counterfactuals"]["feature_share"] = {
        key: (value / cf_total if cf_total > 0 else 0.0)
        for key, value in summary["counterfactuals"]["feature_frequency"].items()
    }

    return summary
