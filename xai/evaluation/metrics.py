"""Metric functions for XAI report evaluation (moved from evaluate_explanations.py)."""
from __future__ import annotations

import itertools
import math

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from domain.score_ops import entropy_concentration
from utils.math_utils import safe_mean, safe_std


# ---------------------------------------------------------------------------
# Report-level helpers
# ---------------------------------------------------------------------------


def model_name(report: Dict[str, Any]) -> str:
    cfg = report["data"].get("config", {})
    label = str(cfg.get("model_label", "")).strip()
    if bool(cfg.get("randomize_weights", False)) and label:
        label = f"{label} [randomized]"
    if label:
        return label
    ckpt = str(cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or "")
    if ckpt:
        return ckpt.split("/runs/")[-1].replace("/checkpoints/last.ckpt", "")
    return Path(report["file"]).name


def report_method(report: Dict[str, Any]) -> str:
    cfg = report.get("data", {}).get("config", {}) or {}
    m = str(cfg.get("attribution_method", "")).strip().lower()
    if m in {"integrated_gradients", "gradient"}:
        return m
    label = str(cfg.get("model_label", "")).strip().lower()
    if "[ig:" in label:
        return "integrated_gradients"
    return "gradient"


def report_seed(report: Dict[str, Any]) -> Optional[int]:
    cfg = report["data"].get("config", {})
    raw_seed = cfg.get("seed", None)
    if raw_seed in (None, ""):
        return None
    return int(raw_seed)


def shared_group_key(report: Dict[str, Any]) -> Tuple[Any, ...]:
    cfg = report["data"].get("config", {})
    ckpt = str(cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or "")
    return (
        ckpt,
        int(cfg.get("num_instances", 0)),
        int(cfg.get("max_steps", 0)),
        tuple(int(v) for v in cfg.get("topk_nodes", [])),
        bool(cfg.get("randomize_weights", False)),
        str(cfg.get("node_importance_mode", "")),
        str(cfg.get("ig_baseline", "")),
        float(cfg.get("feasibility_weight", 0.0)),
        int(cfg.get("feasibility_top_m", 0)),
        float(cfg.get("feasibility_cost_weight", 0.0)),
    )


def reproducibility_group_key(report: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    seed = report_seed(report)
    if seed is None:
        return None
    return shared_group_key(report) + (seed,)


def variant_mix(summary: Dict[str, Any]) -> str:
    counts = summary.get("instance_variant_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "-"
    parts = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ", ".join(f"{k}:{v}" for k, v in parts[:3])


def variant_mix_across_reports(reports: List[Dict[str, Any]]) -> str:
    if not reports:
        return "-"
    aggregate_counts: Dict[str, float] = defaultdict(float)
    for report in reports:
        counts = report.get("data", {}).get("summary", {}).get("instance_variant_counts", {})
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            try:
                aggregate_counts[str(key)] += float(value)
            except (TypeError, ValueError):
                continue
    if not aggregate_counts:
        return "-"
    runs = max(len(reports), 1)
    parts = sorted(aggregate_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    labels = []
    for key, total in parts:
        avg = total / runs
        if abs(avg - round(avg)) < 1e-6:
            labels.append(f"{key}:{int(round(avg))}")
        else:
            labels.append(f"{key}:{avg:.1f}")
    return ", ".join(labels)


# ---------------------------------------------------------------------------
# Constraint share helper
# ---------------------------------------------------------------------------


def constraint_share(payload: List[Dict[str, Any]], constraint_name: str) -> float:
    aliases = [constraint_name]
    if constraint_name == "route_structure":
        aliases.append("route_recourse")
    elif constraint_name == "route_recourse":
        aliases.append("route_structure")
    for item in payload or []:
        if str(item.get("constraint", "")) in aliases:
            try:
                return max(float(item.get("share", 0.0)), 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Deletion faithfulness
# ---------------------------------------------------------------------------


def extract_deletion_metrics(report: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    data = report["data"]
    summary = data.get("summary", {}) or {}
    deletion = summary.get("deletion_faithfulness", {}) or {}
    out: Dict[int, Dict[str, float]] = {}

    if isinstance(deletion, dict) and deletion:
        for raw_k, payload in deletion.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            out[k] = {
                "logit_drop": float(payload.get("mean_logit_drop", float("nan"))),
                "logprob_drop": float(payload.get("mean_logprob_drop", float("nan"))),
                "flip_rate": float(payload.get("mean_action_flip_rate", float("nan"))),
            }
        if out:
            return out

    steps = data.get("steps", []) or []
    accum: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"logit_drop": [], "logprob_drop": [], "flip_rate": []}
    )
    for step in steps:
        payload = step.get("deletion", {}) or {}
        if not isinstance(payload, dict):
            continue
        for raw_k, metric in payload.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            if not isinstance(metric, dict):
                continue
            for src_key, dst_key in [
                ("mean_logit_drop", "logit_drop"),
                ("mean_logprob_drop", "logprob_drop"),
                ("mean_action_flip_rate", "flip_rate"),
            ]:
                try:
                    value = float(metric.get(src_key, float("nan")))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    accum[k][dst_key].append(value)

    for k, vals in accum.items():
        out[k] = {
            "logit_drop": safe_mean(vals["logit_drop"]),
            "logprob_drop": safe_mean(vals["logprob_drop"]),
            "flip_rate": safe_mean(vals["flip_rate"]),
        }
    return out


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------


def _loc_distance(locs: List[List[float]], src: int, dst: int) -> float:
    if not (0 <= src < len(locs) and 0 <= dst < len(locs)):
        return float("nan")
    src_xy = locs[src]
    dst_xy = locs[dst]
    if len(src_xy) < 2 or len(dst_xy) < 2:
        return float("nan")
    return math.hypot(
        float(src_xy[0]) - float(dst_xy[0]),
        float(src_xy[1]) - float(dst_xy[1]),
    )


def _fallback_trajectory_metrics(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    depot_returns: List[float] = []
    depot_shares: List[float] = []
    customer_hops: List[float] = []
    recourse_bursts: List[float] = []
    late_capacity_terms: List[float] = []
    tw_slack_norm_terms: List[float] = []
    late_tw_tight_terms: List[float] = []
    recourse_tw_tight_terms: List[float] = []

    for trace in traces:
        actions = [int(v) for v in (trace.get("actions", []) or [])]
        if not actions:
            continue
        locs = trace.get("locs", []) or []
        recourse_flags = [bool(v) for v in (trace.get("recourse_triggered", []) or [])]
        top_constraints = trace.get("top_constraints", []) or []

        depot_count = sum(1 for action in actions if action == 0)
        depot_returns.append(float(depot_count))
        depot_shares.append(float(depot_count / max(len(actions), 1)))

        per_customer_hops: List[float] = []
        current = 0
        for action in actions:
            hop = _loc_distance(locs, current, action)
            if math.isfinite(hop) and current > 0 and action > 0:
                per_customer_hops.append(hop)
            current = action
        customer_hops.append(safe_mean(per_customer_hops))

        bursts = 0
        prev = False
        for flag in recourse_flags:
            curr = bool(flag)
            if curr and not prev:
                bursts += 1
            prev = curr
        recourse_bursts.append(float(bursts))

        split_idx = max(1, len(actions) // 2)
        late_constraints = top_constraints[split_idx:] or top_constraints[:split_idx]
        for payload in late_constraints:
            late_capacity_terms.append(constraint_share(payload or [], "capacity_demands"))

        solution = trace.get("solution_features", {}) or {}
        is_customer = [bool(v) for v in (solution.get("selected_is_customer", []) or [])]
        tw_slack_norm_raw = solution.get("selected_tw_slack_norm", []) or []
        tw_slack_norm_customer: List[float] = []
        for raw_value, customer_flag in zip(tw_slack_norm_raw, is_customer):
            if not bool(customer_flag):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                tw_slack_norm_customer.append(value)
        if tw_slack_norm_customer:
            tw_slack_norm_terms.append(safe_mean(tw_slack_norm_customer))

        late_tw_values = tw_slack_norm_customer[len(tw_slack_norm_customer) // 2 :]
        if late_tw_values:
            late_tw_tight_terms.append(
                sum(1.0 for value in late_tw_values if value <= 0.10)
                / len(late_tw_values)
            )

        recourse_tw_values: List[float] = []
        for raw_value, customer_flag, recourse_flag in zip(
            tw_slack_norm_raw,
            is_customer,
            recourse_flags,
        ):
            if not bool(customer_flag) or not bool(recourse_flag):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                recourse_tw_values.append(value)
        if recourse_tw_values:
            recourse_tw_tight_terms.append(
                sum(1.0 for value in recourse_tw_values if value <= 0.10)
                / len(recourse_tw_values)
            )

    return {
        "trajectory_depot_returns": safe_mean(depot_returns),
        "trajectory_depot_share": safe_mean(depot_shares),
        "trajectory_customer_hop_distance": safe_mean(customer_hops),
        "trajectory_recourse_burst_count": safe_mean(recourse_bursts),
        "trajectory_late_capacity_share": safe_mean(late_capacity_terms),
        "trajectory_mean_selected_tw_slack_norm": safe_mean(tw_slack_norm_terms),
        "trajectory_late_tw_tight_share": safe_mean(late_tw_tight_terms),
        "trajectory_recourse_under_tw_tight_share": safe_mean(recourse_tw_tight_terms),
    }


def trajectory_metrics(
    summary: Dict[str, Any], traces: List[Dict[str, Any]]
) -> Dict[str, float]:
    trajectory = summary.get("trajectory", {}) or {}
    if trajectory:
        late_share = trajectory.get("late_constraint_share", {}) or {}
        solution = trajectory.get("solution_features", {}) or {}
        return {
            "trajectory_depot_returns": float(
                trajectory.get("mean_depot_returns_per_instance", float("nan"))
            ),
            "trajectory_depot_share": float(
                trajectory.get("mean_depot_return_share", float("nan"))
            ),
            "trajectory_customer_hop_distance": float(
                trajectory.get("mean_customer_hop_distance", float("nan"))
            ),
            "trajectory_recourse_burst_count": float(
                trajectory.get("mean_recourse_burst_count", float("nan"))
            ),
            "trajectory_late_capacity_share": float(
                late_share.get("capacity_demands", float("nan"))
            ),
            "trajectory_mean_selected_tw_slack_norm": float(
                solution.get(
                    "mean_selected_tw_slack_norm",
                    trajectory.get("mean_selected_tw_slack_norm", float("nan")),
                )
            ),
            "trajectory_late_tw_tight_share": float(
                solution.get(
                    "late_tw_tight_step_share",
                    trajectory.get("late_tw_tight_step_share", float("nan")),
                )
            ),
            "trajectory_recourse_under_tw_tight_share": float(
                solution.get(
                    "recourse_under_tw_tight_share",
                    trajectory.get("recourse_under_tw_tight_share", float("nan")),
                )
            ),
        }
    return _fallback_trajectory_metrics(traces)


# ---------------------------------------------------------------------------
# Single-report evaluation
# ---------------------------------------------------------------------------


def evaluate_single_report(report: Dict[str, Any]) -> Dict[str, Any]:
    data = report["data"]
    cfg = data.get("config", {}) or {}
    summary = data.get("summary", {})
    traces = data.get("instances", [])
    counterfactual_summary = summary.get("counterfactuals", {}) or {}
    traj_metrics = trajectory_metrics(summary, traces)

    top1_shares: List[float] = []
    top3_shares: List[float] = []
    clarity_scores: List[float] = []
    tw_active_terms: List[float] = []
    tw_inactive_terms: List[float] = []
    route_active_terms: List[float] = []
    route_inactive_terms: List[float] = []
    optional_consistency_terms: List[float] = []
    stored_step_count = 0

    saw_open_route = False
    saw_route_signal = False

    for trace in traces:
        flags = trace.get("instance_variant_flags", {}) or {}
        has_tw = bool(flags.get("time_windows", False))
        has_open = bool(flags.get("open_route", False))
        saw_open_route = saw_open_route or has_open

        top_scores_all = trace.get("top_scores", [])
        top_constraints_all = trace.get("top_constraints", [])
        step_count = min(len(top_scores_all), len(top_constraints_all))

        for step in range(step_count):
            stored_step_count += 1
            scores = [max(float(v), 0.0) for v in top_scores_all[step] if float(v) > 0]
            if scores:
                total = sum(scores)
                if total > 0:
                    top1_shares.append(scores[0] / total)
                    top3_shares.append(sum(scores[:3]) / total)
                    clarity_scores.append(entropy_concentration(scores))

            cp = top_constraints_all[step] or []
            tw_share = constraint_share(cp, "time_windows_service")
            route_share_val = constraint_share(cp, "route_structure")
            if route_share_val > 0:
                saw_route_signal = True

            if has_tw:
                tw_active_terms.append(tw_share)
                optional_consistency_terms.append(tw_share)
            else:
                tw_inactive_terms.append(tw_share)
                optional_consistency_terms.append(1.0 - tw_share)

            if has_open:
                route_active_terms.append(route_share_val)
            else:
                route_inactive_terms.append(route_share_val)

    include_route_consistency = saw_open_route or saw_route_signal
    if include_route_consistency:
        optional_consistency_terms.extend(route_active_terms)
        optional_consistency_terms.extend(1.0 - share for share in route_inactive_terms)

    contrastive_summary = summary.get("contrastive", {}) or {}

    if not counterfactual_summary and traces:
        cf_available_terms: List[float] = []
        cf_switch_terms: List[float] = []
        cf_make_feasible_terms: List[float] = []
        cf_delta_terms: List[float] = []
        for trace in traces:
            for payload in trace.get("counterfactuals", []) or []:
                if not isinstance(payload, dict) or not payload:
                    cf_available_terms.append(0.0)
                    cf_switch_terms.append(0.0)
                    cf_make_feasible_terms.append(0.0)
                    continue
                status = str(payload.get("status", "approximate"))
                cf_available_terms.append(1.0)
                cf_switch_terms.append(1.0 if status == "switch" else 0.0)
                cf_make_feasible_terms.append(1.0 if status == "make_feasible" else 0.0)
                rel = payload.get("relative_delta", None)
                if rel is not None:
                    try:
                        rel_value = float(rel)
                    except (TypeError, ValueError):
                        rel_value = float("nan")
                    if math.isfinite(rel_value):
                        cf_delta_terms.append(rel_value)
        counterfactual_summary = {
            "available_rate": safe_mean(cf_available_terms),
            "switch_rate": safe_mean(cf_switch_terms),
            "make_feasible_rate": safe_mean(cf_make_feasible_terms),
            "mean_relative_delta": safe_mean(cf_delta_terms),
        }

    return {
        "file": report["file"],
        "model": model_name(report),
        "seed": cfg.get("seed", None),
        "variant_mix": variant_mix(summary),
        "stored_steps": stored_step_count,
        "focus_top1": safe_mean(top1_shares),
        "focus_top3": safe_mean(top3_shares),
        "clarity": safe_mean(clarity_scores),
        "contrast_gap": contrastive_summary.get("mean_logit_gap"),
        "contrast_alt_rate": contrastive_summary.get("alt_available_rate"),
        "optional_consistency": safe_mean(optional_consistency_terms),
        "tw_active_mass": safe_mean(tw_active_terms),
        "tw_inactive_mass": safe_mean(tw_inactive_terms),
        "counterfactual_available_rate": counterfactual_summary.get("available_rate", float("nan")),
        "counterfactual_switch_rate": counterfactual_summary.get("switch_rate", float("nan")),
        "counterfactual_make_feasible_rate": counterfactual_summary.get("make_feasible_rate", float("nan")),
        "counterfactual_mean_relative_delta": counterfactual_summary.get("mean_relative_delta", float("nan")),
        "trajectory_depot_returns": traj_metrics["trajectory_depot_returns"],
        "trajectory_depot_share": traj_metrics["trajectory_depot_share"],
        "trajectory_customer_hop_distance": traj_metrics["trajectory_customer_hop_distance"],
        "trajectory_recourse_burst_count": traj_metrics["trajectory_recourse_burst_count"],
        "trajectory_late_capacity_share": traj_metrics["trajectory_late_capacity_share"],
        "trajectory_mean_selected_tw_slack_norm": traj_metrics[
            "trajectory_mean_selected_tw_slack_norm"
        ],
        "trajectory_late_tw_tight_share": traj_metrics["trajectory_late_tw_tight_share"],
        "trajectory_recourse_under_tw_tight_share": traj_metrics[
            "trajectory_recourse_under_tw_tight_share"
        ],
        "recourse_rate": summary.get("recourse_event_rate"),
        "chosen_feasible_rate": summary.get("chosen_action_feasible_rate"),
    }


# ---------------------------------------------------------------------------
# Aggregate / stability / robustness
# ---------------------------------------------------------------------------


def aggregate_report_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        metric_rows = [evaluate_single_report(report) for report in group_reports]
        seeds = sorted(
            {
                seed
                for seed in (report_seed(report) for report in group_reports)
                if seed is not None
            }
        )
        rows.append(
            {
                "model": model_name(group_reports[0]),
                "variant_mix": variant_mix_across_reports(group_reports),
                "runs": len(metric_rows),
                "seeds": len(seeds),
                "seed_range": f"{min(seeds)}..{max(seeds)}" if seeds else "-",
                "stored_steps": safe_mean(row["stored_steps"] for row in metric_rows),
                "focus_top1": safe_mean(row["focus_top1"] for row in metric_rows),
                "focus_top3": safe_mean(row["focus_top3"] for row in metric_rows),
                "clarity": safe_mean(row["clarity"] for row in metric_rows),
                "contrast_gap": safe_mean(row["contrast_gap"] for row in metric_rows),
                "contrast_alt_rate": safe_mean(row["contrast_alt_rate"] for row in metric_rows),
                "optional_consistency": safe_mean(row["optional_consistency"] for row in metric_rows),
                "counterfactual_available_rate": safe_mean(row["counterfactual_available_rate"] for row in metric_rows),
                "counterfactual_switch_rate": safe_mean(row["counterfactual_switch_rate"] for row in metric_rows),
                "counterfactual_make_feasible_rate": safe_mean(row["counterfactual_make_feasible_rate"] for row in metric_rows),
                "counterfactual_mean_relative_delta": safe_mean(row["counterfactual_mean_relative_delta"] for row in metric_rows),
                "trajectory_depot_returns": safe_mean(row["trajectory_depot_returns"] for row in metric_rows),
                "trajectory_depot_share": safe_mean(row["trajectory_depot_share"] for row in metric_rows),
                "trajectory_customer_hop_distance": safe_mean(row["trajectory_customer_hop_distance"] for row in metric_rows),
                "trajectory_recourse_burst_count": safe_mean(row["trajectory_recourse_burst_count"] for row in metric_rows),
                "trajectory_late_capacity_share": safe_mean(row["trajectory_late_capacity_share"] for row in metric_rows),
                "trajectory_mean_selected_tw_slack_norm": safe_mean(
                    row["trajectory_mean_selected_tw_slack_norm"] for row in metric_rows
                ),
                "trajectory_late_tw_tight_share": safe_mean(
                    row["trajectory_late_tw_tight_share"] for row in metric_rows
                ),
                "trajectory_recourse_under_tw_tight_share": safe_mean(
                    row["trajectory_recourse_under_tw_tight_share"] for row in metric_rows
                ),
                "tw_active_mass": safe_mean(row["tw_active_mass"] for row in metric_rows),
                "tw_inactive_mass": safe_mean(row["tw_inactive_mass"] for row in metric_rows),
                "recourse_rate": safe_mean(row["recourse_rate"] for row in metric_rows),
                "chosen_feasible_rate": safe_mean(row["chosen_feasible_rate"] for row in metric_rows),
            }
        )
    return rows


def pairwise_stability(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    traces_a = a["data"].get("instances", [])
    traces_b = b["data"].get("instances", [])

    overlap_at_1: List[float] = []
    overlap_at_3: List[float] = []
    constraint_top1_match: List[float] = []
    alt_action_match: List[float] = []
    compared_steps = 0

    for idx in range(min(len(traces_a), len(traces_b))):
        ta = traces_a[idx]
        tb = traces_b[idx]
        step_count = min(
            len(ta.get("top_nodes", [])),
            len(tb.get("top_nodes", [])),
            len(ta.get("top_constraints", [])),
            len(tb.get("top_constraints", [])),
        )
        for step in range(step_count):
            compared_steps += 1
            nodes_a = [int(v) for v in ta.get("top_nodes", [])[step]]
            nodes_b = [int(v) for v in tb.get("top_nodes", [])[step]]
            if nodes_a and nodes_b:
                overlap_at_1.append(1.0 if nodes_a[0] == nodes_b[0] else 0.0)
                set_a = set(nodes_a[:3])
                set_b = set(nodes_b[:3])
                union = set_a | set_b
                overlap_at_3.append((len(set_a & set_b) / len(union)) if union else float("nan"))

            constraints_a = ta.get("top_constraints", [])[step] or []
            constraints_b = tb.get("top_constraints", [])[step] or []
            if constraints_a and constraints_b:
                c1 = str(constraints_a[0].get("constraint", ""))
                c2 = str(constraints_b[0].get("constraint", ""))
                if c1 and c2:
                    constraint_top1_match.append(1.0 if c1 == c2 else 0.0)

            alt_a = ta.get("contrastive_alt_action", [])
            alt_b = tb.get("contrastive_alt_action", [])
            if step < len(alt_a) and step < len(alt_b):
                a_val = int(alt_a[step])
                b_val = int(alt_b[step])
                if a_val >= 0 and b_val >= 0:
                    alt_action_match.append(1.0 if a_val == b_val else 0.0)

    return {
        "compared_steps": float(compared_steps),
        "overlap_at_1": safe_mean(overlap_at_1),
        "overlap_at_3": safe_mean(overlap_at_3),
        "constraint_top1_match": safe_mean(constraint_top1_match),
        "alt_action_match": safe_mean(alt_action_match),
    }


def evaluate_reproducibility(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        key = reproducibility_group_key(report)
        if key is None:
            continue
        groups[key].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        if len(group_reports) < 2:
            continue
        pair_metrics = [
            pairwise_stability(a, b) for a, b in itertools.combinations(group_reports, 2)
        ]
        rows.append(
            {
                "model": model_name(group_reports[0]),
                "runs": len(group_reports),
                "pairs": len(pair_metrics),
                "compared_steps": safe_mean(m["compared_steps"] for m in pair_metrics),
                "overlap_at_1": safe_mean(m["overlap_at_1"] for m in pair_metrics),
                "overlap_at_3": safe_mean(m["overlap_at_3"] for m in pair_metrics),
                "constraint_top1_match": safe_mean(m["constraint_top1_match"] for m in pair_metrics),
                "alt_action_match": safe_mean(m["alt_action_match"] for m in pair_metrics),
            }
        )
    return rows


def evaluate_robustness(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_reports: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        if report_seed(report) is None:
            continue
        grouped_reports[shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in grouped_reports.items():
        by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for report in group_reports:
            seed = report_seed(report)
            if seed is None:
                continue
            by_seed[seed].append(evaluate_single_report(report))

        if len(by_seed) < 2:
            continue

        seed_rows: List[Dict[str, Any]] = []
        for seed, metrics in sorted(by_seed.items()):
            seed_rows.append(
                {
                    "seed": seed,
                    "focus_top1": safe_mean(m["focus_top1"] for m in metrics),
                    "focus_top3": safe_mean(m["focus_top3"] for m in metrics),
                    "clarity": safe_mean(m["clarity"] for m in metrics),
                    "contrast_gap": safe_mean(m["contrast_gap"] for m in metrics),
                    "counterfactual_available_rate": safe_mean(m["counterfactual_available_rate"] for m in metrics),
                    "counterfactual_switch_rate": safe_mean(m["counterfactual_switch_rate"] for m in metrics),
                    "counterfactual_make_feasible_rate": safe_mean(m["counterfactual_make_feasible_rate"] for m in metrics),
                    "counterfactual_mean_relative_delta": safe_mean(m["counterfactual_mean_relative_delta"] for m in metrics),
                    "trajectory_depot_returns": safe_mean(m["trajectory_depot_returns"] for m in metrics),
                    "trajectory_customer_hop_distance": safe_mean(m["trajectory_customer_hop_distance"] for m in metrics),
                    "trajectory_late_capacity_share": safe_mean(m["trajectory_late_capacity_share"] for m in metrics),
                    "trajectory_mean_selected_tw_slack_norm": safe_mean(
                        m["trajectory_mean_selected_tw_slack_norm"] for m in metrics
                    ),
                    "trajectory_late_tw_tight_share": safe_mean(
                        m["trajectory_late_tw_tight_share"] for m in metrics
                    ),
                    "trajectory_recourse_under_tw_tight_share": safe_mean(
                        m["trajectory_recourse_under_tw_tight_share"] for m in metrics
                    ),
                    "optional_consistency": safe_mean(m["optional_consistency"] for m in metrics),
                    "tw_inactive_mass": safe_mean(m["tw_inactive_mass"] for m in metrics),
                    "recourse_rate": safe_mean(m["recourse_rate"] for m in metrics),
                    "chosen_feasible_rate": safe_mean(m["chosen_feasible_rate"] for m in metrics),
                }
            )

        rows.append(
            {
                "model": model_name(group_reports[0]),
                "runs": len(group_reports),
                "seeds": len(seed_rows),
                "seed_min": min(row["seed"] for row in seed_rows),
                "seed_max": max(row["seed"] for row in seed_rows),
                "focus_top1_mean": safe_mean(r["focus_top1"] for r in seed_rows),
                "focus_top1_std": safe_std(r["focus_top1"] for r in seed_rows),
                "clarity_mean": safe_mean(r["clarity"] for r in seed_rows),
                "clarity_std": safe_std(r["clarity"] for r in seed_rows),
                "contrast_gap_mean": safe_mean(r["contrast_gap"] for r in seed_rows),
                "contrast_gap_std": safe_std(r["contrast_gap"] for r in seed_rows),
                "counterfactual_available_rate_mean": safe_mean(r["counterfactual_available_rate"] for r in seed_rows),
                "counterfactual_available_rate_std": safe_std(r["counterfactual_available_rate"] for r in seed_rows),
                "counterfactual_switch_rate_mean": safe_mean(r["counterfactual_switch_rate"] for r in seed_rows),
                "counterfactual_switch_rate_std": safe_std(r["counterfactual_switch_rate"] for r in seed_rows),
                "counterfactual_make_feasible_rate_mean": safe_mean(r["counterfactual_make_feasible_rate"] for r in seed_rows),
                "counterfactual_make_feasible_rate_std": safe_std(r["counterfactual_make_feasible_rate"] for r in seed_rows),
                "counterfactual_mean_relative_delta_mean": safe_mean(r["counterfactual_mean_relative_delta"] for r in seed_rows),
                "counterfactual_mean_relative_delta_std": safe_std(r["counterfactual_mean_relative_delta"] for r in seed_rows),
                "trajectory_depot_returns_mean": safe_mean(r["trajectory_depot_returns"] for r in seed_rows),
                "trajectory_depot_returns_std": safe_std(r["trajectory_depot_returns"] for r in seed_rows),
                "trajectory_customer_hop_distance_mean": safe_mean(r["trajectory_customer_hop_distance"] for r in seed_rows),
                "trajectory_customer_hop_distance_std": safe_std(r["trajectory_customer_hop_distance"] for r in seed_rows),
                "trajectory_late_capacity_share_mean": safe_mean(r["trajectory_late_capacity_share"] for r in seed_rows),
                "trajectory_late_capacity_share_std": safe_std(r["trajectory_late_capacity_share"] for r in seed_rows),
                "trajectory_mean_selected_tw_slack_norm_mean": safe_mean(
                    r["trajectory_mean_selected_tw_slack_norm"] for r in seed_rows
                ),
                "trajectory_mean_selected_tw_slack_norm_std": safe_std(
                    r["trajectory_mean_selected_tw_slack_norm"] for r in seed_rows
                ),
                "trajectory_late_tw_tight_share_mean": safe_mean(
                    r["trajectory_late_tw_tight_share"] for r in seed_rows
                ),
                "trajectory_late_tw_tight_share_std": safe_std(
                    r["trajectory_late_tw_tight_share"] for r in seed_rows
                ),
                "trajectory_recourse_under_tw_tight_share_mean": safe_mean(
                    r["trajectory_recourse_under_tw_tight_share"] for r in seed_rows
                ),
                "trajectory_recourse_under_tw_tight_share_std": safe_std(
                    r["trajectory_recourse_under_tw_tight_share"] for r in seed_rows
                ),
                "optional_consistency_mean": safe_mean(r["optional_consistency"] for r in seed_rows),
                "optional_consistency_std": safe_std(r["optional_consistency"] for r in seed_rows),
                "tw_inactive_mass_mean": safe_mean(r["tw_inactive_mass"] for r in seed_rows),
                "tw_inactive_mass_std": safe_std(r["tw_inactive_mass"] for r in seed_rows),
                "recourse_rate_mean": safe_mean(r["recourse_rate"] for r in seed_rows),
                "recourse_rate_std": safe_std(r["recourse_rate"] for r in seed_rows),
                "chosen_feasible_rate_mean": safe_mean(r["chosen_feasible_rate"] for r in seed_rows),
                "chosen_feasible_rate_std": safe_std(r["chosen_feasible_rate"] for r in seed_rows),
            }
        )
    return rows


def deletion_rows(
    reports: List[Dict[str, Any]], aggregate: bool
) -> List[Dict[str, Any]]:
    if not reports:
        return []
    if not aggregate:
        return [
            {
                "model": model_name(report),
                "variant_mix": variant_mix(report["data"].get("summary", {}) or {}),
                "deletion_metrics": extract_deletion_metrics(report),
            }
            for report in reports
        ]

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        metrics_by_report = [extract_deletion_metrics(report) for report in group_reports]
        all_k = sorted({k for metrics in metrics_by_report for k in metrics.keys()})
        agg_metrics: Dict[int, Dict[str, float]] = {}
        for k in all_k:
            agg_metrics[k] = {
                "flip_rate": safe_mean(
                    metrics.get(k, {}).get("flip_rate", float("nan"))
                    for metrics in metrics_by_report
                ),
                "logprob_drop": safe_mean(
                    metrics.get(k, {}).get("logprob_drop", float("nan"))
                    for metrics in metrics_by_report
                ),
                "logit_drop": safe_mean(
                    metrics.get(k, {}).get("logit_drop", float("nan"))
                    for metrics in metrics_by_report
                ),
            }
        seeds = sorted(
            {
                seed
                for seed in (report_seed(report) for report in group_reports)
                if seed is not None
            }
        )
        rows.append(
            {
                "model": model_name(group_reports[0]),
                "variant_mix": variant_mix_across_reports(group_reports),
                "runs": len(group_reports),
                "seeds": len(seeds),
                "seed_range": f"{min(seeds)}..{max(seeds)}" if seeds else "-",
                "deletion_metrics": agg_metrics,
            }
        )
    return rows
