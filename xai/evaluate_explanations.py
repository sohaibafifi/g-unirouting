import argparse
import glob
import itertools
import json
import math
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.table import Table


def _table_console(layout: str, compact_threshold: int) -> Tuple[Console, bool]:
    console = Console()
    width = getattr(console.size, "width", console.width)
    compact = layout == "compact" or (layout == "auto" and width < compact_threshold)
    return console, compact


def _compact_metric_lines(pairs: List[Tuple[str, Any]], ndigits: int = 4) -> str:
    return "\n".join(f"{label}={_fmt_float(value, ndigits=ndigits)}" for label, value in pairs)


def _cell_text(value: Any) -> str:
    text = str(value)
    return text if text else "-"


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _safe_std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


def _fmt_float(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "nan"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{ndigits}f}"


def _entropy_concentration(scores: List[float]) -> float:
    positive = [max(float(s), 0.0) for s in scores if float(s) > 0]
    if not positive:
        return float("nan")
    total = sum(positive)
    if total <= 0:
        return float("nan")
    probs = [s / total for s in positive]
    if len(probs) == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 1.0
    return 1.0 - (entropy / max_entropy)


def _constraint_share(payload: List[Dict[str, Any]], constraint_name: str) -> float:
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


def _load_reports(pattern: str, latest: int | None) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if latest is not None:
        files = files[:latest]

    reports: List[Dict[str, Any]] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        reports.append({"file": path, "data": data})
    return reports


def _model_name(report: Dict[str, Any]) -> str:
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


def _variant_mix(summary: Dict[str, Any]) -> str:
    counts = summary.get("instance_variant_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "-"
    parts = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ", ".join(f"{k}:{v}" for k, v in parts[:3])


def _variant_mix_across_reports(reports: List[Dict[str, Any]]) -> str:
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


def _loc_distance(locs: List[List[float]], src: int, dst: int) -> float:
    if not (0 <= src < len(locs) and 0 <= dst < len(locs)):
        return float("nan")
    src_xy = locs[src]
    dst_xy = locs[dst]
    if len(src_xy) < 2 or len(dst_xy) < 2:
        return float("nan")
    return math.hypot(float(src_xy[0]) - float(dst_xy[0]), float(src_xy[1]) - float(dst_xy[1]))


def _fallback_trajectory_metrics(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    depot_returns: List[float] = []
    depot_shares: List[float] = []
    customer_hops: List[float] = []
    recourse_bursts: List[float] = []
    late_capacity_terms: List[float] = []

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
        customer_hops.append(_safe_mean(per_customer_hops))

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
            late_capacity_terms.append(_constraint_share(payload or [], "capacity_demands"))

    return {
        "trajectory_depot_returns": _safe_mean(depot_returns),
        "trajectory_depot_share": _safe_mean(depot_shares),
        "trajectory_customer_hop_distance": _safe_mean(customer_hops),
        "trajectory_recourse_burst_count": _safe_mean(recourse_bursts),
        "trajectory_late_capacity_share": _safe_mean(late_capacity_terms),
    }


def _trajectory_metrics(
    summary: Dict[str, Any], traces: List[Dict[str, Any]]
) -> Dict[str, float]:
    trajectory = summary.get("trajectory", {}) or {}
    if trajectory:
        late_share = trajectory.get("late_constraint_share", {}) or {}
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
        }
    return _fallback_trajectory_metrics(traces)


def _evaluate_single_report(report: Dict[str, Any]) -> Dict[str, Any]:
    data = report["data"]
    cfg = data.get("config", {}) or {}
    summary = data.get("summary", {})
    traces = data.get("instances", [])
    counterfactual_summary = summary.get("counterfactuals", {}) or {}
    trajectory_metrics = _trajectory_metrics(summary, traces)

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
                    clarity_scores.append(_entropy_concentration(scores))

            constraint_payload = top_constraints_all[step] or []
            tw_share = _constraint_share(constraint_payload, "time_windows_service")
            route_share = _constraint_share(constraint_payload, "route_structure")
            if route_share > 0:
                saw_route_signal = True

            if has_tw:
                tw_active_terms.append(tw_share)
                optional_consistency_terms.append(tw_share)
            else:
                tw_inactive_terms.append(tw_share)
                optional_consistency_terms.append(1.0 - tw_share)

            if has_open:
                route_active_terms.append(route_share)
            else:
                route_inactive_terms.append(route_share)

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
                cf_make_feasible_terms.append(
                    1.0 if status == "make_feasible" else 0.0
                )
                rel = payload.get("relative_delta", None)
                if rel is not None:
                    try:
                        rel_value = float(rel)
                    except (TypeError, ValueError):
                        rel_value = float("nan")
                    if math.isfinite(rel_value):
                        cf_delta_terms.append(rel_value)
        counterfactual_summary = {
            "available_rate": _safe_mean(cf_available_terms),
            "switch_rate": _safe_mean(cf_switch_terms),
            "make_feasible_rate": _safe_mean(cf_make_feasible_terms),
            "mean_relative_delta": _safe_mean(cf_delta_terms),
        }

    return {
        "file": report["file"],
        "model": _model_name(report),
        "seed": cfg.get("seed", None),
        "variant_mix": _variant_mix(summary),
        "stored_steps": stored_step_count,
        "focus_top1": _safe_mean(top1_shares),
        "focus_top3": _safe_mean(top3_shares),
        "clarity": _safe_mean(clarity_scores),
        "contrast_gap": contrastive_summary.get("mean_logit_gap"),
        "contrast_alt_rate": contrastive_summary.get("alt_available_rate"),
        "optional_consistency": _safe_mean(optional_consistency_terms),
        "tw_active_mass": _safe_mean(tw_active_terms),
        "tw_inactive_mass": _safe_mean(tw_inactive_terms),
        "counterfactual_available_rate": counterfactual_summary.get(
            "available_rate", float("nan")
        ),
        "counterfactual_switch_rate": counterfactual_summary.get(
            "switch_rate", float("nan")
        ),
        "counterfactual_make_feasible_rate": counterfactual_summary.get(
            "make_feasible_rate", float("nan")
        ),
        "counterfactual_mean_relative_delta": counterfactual_summary.get(
            "mean_relative_delta", float("nan")
        ),
        "trajectory_depot_returns": trajectory_metrics["trajectory_depot_returns"],
        "trajectory_depot_share": trajectory_metrics["trajectory_depot_share"],
        "trajectory_customer_hop_distance": trajectory_metrics[
            "trajectory_customer_hop_distance"
        ],
        "trajectory_recourse_burst_count": trajectory_metrics[
            "trajectory_recourse_burst_count"
        ],
        "trajectory_late_capacity_share": trajectory_metrics[
            "trajectory_late_capacity_share"
        ],
        "recourse_rate": summary.get("recourse_event_rate"),
        "chosen_feasible_rate": summary.get("chosen_action_feasible_rate"),
    }


def _aggregate_report_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[_shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        metric_rows = [_evaluate_single_report(report) for report in group_reports]
        seeds = sorted(
            {
                seed
                for seed in (_report_seed(report) for report in group_reports)
                if seed is not None
            }
        )
        rows.append(
            {
                "model": _model_name(group_reports[0]),
                "variant_mix": _variant_mix_across_reports(group_reports),
                "runs": len(metric_rows),
                "seeds": len(seeds),
                "seed_range": (
                    f"{min(seeds)}..{max(seeds)}" if seeds else "-"
                ),
                "stored_steps": _safe_mean(row["stored_steps"] for row in metric_rows),
                "focus_top1": _safe_mean(row["focus_top1"] for row in metric_rows),
                "focus_top3": _safe_mean(row["focus_top3"] for row in metric_rows),
                "clarity": _safe_mean(row["clarity"] for row in metric_rows),
                "contrast_gap": _safe_mean(
                    row["contrast_gap"] for row in metric_rows
                ),
                "contrast_alt_rate": _safe_mean(
                    row["contrast_alt_rate"] for row in metric_rows
                ),
                "optional_consistency": _safe_mean(
                    row["optional_consistency"] for row in metric_rows
                ),
                "counterfactual_available_rate": _safe_mean(
                    row["counterfactual_available_rate"] for row in metric_rows
                ),
                "counterfactual_switch_rate": _safe_mean(
                    row["counterfactual_switch_rate"] for row in metric_rows
                ),
                "counterfactual_make_feasible_rate": _safe_mean(
                    row["counterfactual_make_feasible_rate"] for row in metric_rows
                ),
                "counterfactual_mean_relative_delta": _safe_mean(
                    row["counterfactual_mean_relative_delta"] for row in metric_rows
                ),
                "trajectory_depot_returns": _safe_mean(
                    row["trajectory_depot_returns"] for row in metric_rows
                ),
                "trajectory_depot_share": _safe_mean(
                    row["trajectory_depot_share"] for row in metric_rows
                ),
                "trajectory_customer_hop_distance": _safe_mean(
                    row["trajectory_customer_hop_distance"] for row in metric_rows
                ),
                "trajectory_recourse_burst_count": _safe_mean(
                    row["trajectory_recourse_burst_count"] for row in metric_rows
                ),
                "trajectory_late_capacity_share": _safe_mean(
                    row["trajectory_late_capacity_share"] for row in metric_rows
                ),
                "tw_active_mass": _safe_mean(
                    row["tw_active_mass"] for row in metric_rows
                ),
                "tw_inactive_mass": _safe_mean(
                    row["tw_inactive_mass"] for row in metric_rows
                ),
                "recourse_rate": _safe_mean(
                    row["recourse_rate"] for row in metric_rows
                ),
                "chosen_feasible_rate": _safe_mean(
                    row["chosen_feasible_rate"] for row in metric_rows
                ),
            }
        )
    return rows


def _report_seed(report: Dict[str, Any]) -> Optional[int]:
    cfg = report["data"].get("config", {})
    raw_seed = cfg.get("seed", None)
    if raw_seed in (None, ""):
        return None
    return int(raw_seed)


def _shared_group_key(report: Dict[str, Any]) -> Tuple[Any, ...]:
    cfg = report["data"].get("config", {})
    ckpt = str(cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or "")
    return (
        ckpt,
        int(cfg.get("num_instances", 0)),
        int(cfg.get("max_steps", 0)),
        tuple(int(v) for v in cfg.get("topk_nodes", [])),
        bool(cfg.get("randomize_weights", False)),
        str(cfg.get("node_importance_mode", "")),
        float(cfg.get("feasibility_weight", 0.0)),
        int(cfg.get("feasibility_top_m", 0)),
        float(cfg.get("feasibility_cost_weight", 0.0)),
    )


def _reproducibility_group_key(report: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    seed = _report_seed(report)
    if seed is None:
        return None

    return _shared_group_key(report) + (seed,)


def _pairwise_stability(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
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
                overlap_at_3.append(
                    (len(set_a & set_b) / len(union)) if union else float("nan")
                )

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
        "overlap_at_1": _safe_mean(overlap_at_1),
        "overlap_at_3": _safe_mean(overlap_at_3),
        "constraint_top1_match": _safe_mean(constraint_top1_match),
        "alt_action_match": _safe_mean(alt_action_match),
    }


def _evaluate_reproducibility(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        key = _reproducibility_group_key(report)
        if key is None:
            continue
        groups[key].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in groups.items():
        if len(group_reports) < 2:
            continue

        pair_metrics = [
            _pairwise_stability(a, b) for a, b in itertools.combinations(group_reports, 2)
        ]
        rows.append(
            {
                "model": _model_name(group_reports[0]),
                "runs": len(group_reports),
                "pairs": len(pair_metrics),
                "compared_steps": _safe_mean(m["compared_steps"] for m in pair_metrics),
                "overlap_at_1": _safe_mean(m["overlap_at_1"] for m in pair_metrics),
                "overlap_at_3": _safe_mean(m["overlap_at_3"] for m in pair_metrics),
                "constraint_top1_match": _safe_mean(
                    m["constraint_top1_match"] for m in pair_metrics
                ),
                "alt_action_match": _safe_mean(
                    m["alt_action_match"] for m in pair_metrics
                ),
            }
        )
    return rows


def _evaluate_robustness(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_reports: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for report in reports:
        if _report_seed(report) is None:
            continue
        grouped_reports[_shared_group_key(report)].append(report)

    rows: List[Dict[str, Any]] = []
    for _, group_reports in grouped_reports.items():
        by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for report in group_reports:
            seed = _report_seed(report)
            if seed is None:
                continue
            by_seed[seed].append(_evaluate_single_report(report))

        if len(by_seed) < 2:
            continue

        seed_rows: List[Dict[str, Any]] = []
        for seed, metrics in sorted(by_seed.items()):
            seed_rows.append(
                {
                    "seed": seed,
                    "focus_top1": _safe_mean(m["focus_top1"] for m in metrics),
                    "focus_top3": _safe_mean(m["focus_top3"] for m in metrics),
                    "clarity": _safe_mean(m["clarity"] for m in metrics),
                    "contrast_gap": _safe_mean(m["contrast_gap"] for m in metrics),
                    "counterfactual_available_rate": _safe_mean(
                        m["counterfactual_available_rate"] for m in metrics
                    ),
                    "counterfactual_switch_rate": _safe_mean(
                        m["counterfactual_switch_rate"] for m in metrics
                    ),
                    "counterfactual_make_feasible_rate": _safe_mean(
                        m["counterfactual_make_feasible_rate"] for m in metrics
                    ),
                    "counterfactual_mean_relative_delta": _safe_mean(
                        m["counterfactual_mean_relative_delta"] for m in metrics
                    ),
                    "trajectory_depot_returns": _safe_mean(
                        m["trajectory_depot_returns"] for m in metrics
                    ),
                    "trajectory_customer_hop_distance": _safe_mean(
                        m["trajectory_customer_hop_distance"] for m in metrics
                    ),
                    "trajectory_late_capacity_share": _safe_mean(
                        m["trajectory_late_capacity_share"] for m in metrics
                    ),
                    "optional_consistency": _safe_mean(
                        m["optional_consistency"] for m in metrics
                    ),
                    "tw_inactive_mass": _safe_mean(
                        m["tw_inactive_mass"] for m in metrics
                    ),
                    "recourse_rate": _safe_mean(m["recourse_rate"] for m in metrics),
                    "chosen_feasible_rate": _safe_mean(
                        m["chosen_feasible_rate"] for m in metrics
                    ),
                }
            )

        rows.append(
            {
                "model": _model_name(group_reports[0]),
                "runs": len(group_reports),
                "seeds": len(seed_rows),
                "seed_min": min(row["seed"] for row in seed_rows),
                "seed_max": max(row["seed"] for row in seed_rows),
                "focus_top1_mean": _safe_mean(row["focus_top1"] for row in seed_rows),
                "focus_top1_std": _safe_std(row["focus_top1"] for row in seed_rows),
                "clarity_mean": _safe_mean(row["clarity"] for row in seed_rows),
                "clarity_std": _safe_std(row["clarity"] for row in seed_rows),
                "contrast_gap_mean": _safe_mean(row["contrast_gap"] for row in seed_rows),
                "contrast_gap_std": _safe_std(row["contrast_gap"] for row in seed_rows),
                "counterfactual_available_rate_mean": _safe_mean(
                    row["counterfactual_available_rate"] for row in seed_rows
                ),
                "counterfactual_available_rate_std": _safe_std(
                    row["counterfactual_available_rate"] for row in seed_rows
                ),
                "counterfactual_switch_rate_mean": _safe_mean(
                    row["counterfactual_switch_rate"] for row in seed_rows
                ),
                "counterfactual_switch_rate_std": _safe_std(
                    row["counterfactual_switch_rate"] for row in seed_rows
                ),
                "counterfactual_make_feasible_rate_mean": _safe_mean(
                    row["counterfactual_make_feasible_rate"] for row in seed_rows
                ),
                "counterfactual_make_feasible_rate_std": _safe_std(
                    row["counterfactual_make_feasible_rate"] for row in seed_rows
                ),
                "counterfactual_mean_relative_delta_mean": _safe_mean(
                    row["counterfactual_mean_relative_delta"] for row in seed_rows
                ),
                "counterfactual_mean_relative_delta_std": _safe_std(
                    row["counterfactual_mean_relative_delta"] for row in seed_rows
                ),
                "trajectory_depot_returns_mean": _safe_mean(
                    row["trajectory_depot_returns"] for row in seed_rows
                ),
                "trajectory_depot_returns_std": _safe_std(
                    row["trajectory_depot_returns"] for row in seed_rows
                ),
                "trajectory_customer_hop_distance_mean": _safe_mean(
                    row["trajectory_customer_hop_distance"] for row in seed_rows
                ),
                "trajectory_customer_hop_distance_std": _safe_std(
                    row["trajectory_customer_hop_distance"] for row in seed_rows
                ),
                "trajectory_late_capacity_share_mean": _safe_mean(
                    row["trajectory_late_capacity_share"] for row in seed_rows
                ),
                "trajectory_late_capacity_share_std": _safe_std(
                    row["trajectory_late_capacity_share"] for row in seed_rows
                ),
                "optional_consistency_mean": _safe_mean(
                    row["optional_consistency"] for row in seed_rows
                ),
                "optional_consistency_std": _safe_std(
                    row["optional_consistency"] for row in seed_rows
                ),
                "tw_inactive_mass_mean": _safe_mean(
                    row["tw_inactive_mass"] for row in seed_rows
                ),
                "tw_inactive_mass_std": _safe_std(
                    row["tw_inactive_mass"] for row in seed_rows
                ),
                "recourse_rate_mean": _safe_mean(
                    row["recourse_rate"] for row in seed_rows
                ),
                "recourse_rate_std": _safe_std(row["recourse_rate"] for row in seed_rows),
                "chosen_feasible_rate_mean": _safe_mean(
                    row["chosen_feasible_rate"] for row in seed_rows
                ),
                "chosen_feasible_rate_std": _safe_std(
                    row["chosen_feasible_rate"] for row in seed_rows
                ),
            }
        )
    return rows


def _print_report_table(rows: List[Dict[str, Any]], layout: str = "auto") -> None:
    console, compact = _table_console(layout, compact_threshold=150)
    if not rows:
        console.print("[yellow]No reports matched[/yellow]")
        return

    aggregated = "runs" in rows[0]

    if compact:
        table = Table(
            title="Explanation Evaluation",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", ratio=3)
        table.add_column("setup", overflow="fold", ratio=2)
        table.add_column("metrics", overflow="fold", ratio=3)

        for row in rows:
            setup_lines = [f"variants: {_cell_text(row['variant_mix'])}"]
            if aggregated:
                setup_lines.append(
                    f"runs={int(row['runs'])} seeds={int(row['seeds'])} range={row['seed_range']}"
                )

            metrics_lines = []
            metrics_lines.append(
                "steps="
                + (
                    _fmt_float(row["stored_steps"], ndigits=1)
                    if aggregated
                    else str(int(row["stored_steps"]))
                )
            )
            metrics_lines.append(
                " ".join(
                    [
                        f"f1={_fmt_float(row['focus_top1'])}",
                        f"f3={_fmt_float(row['focus_top3'])}",
                        f"clr={_fmt_float(row['clarity'])}",
                    ]
                )
            )
            metrics_lines.append(
                " ".join(
                    [
                        f"ctr={_fmt_float(row['contrast_gap'])}",
                        f"alt={_fmt_float(row['contrast_alt_rate'])}",
                        f"opt={_fmt_float(row['optional_consistency'])}",
                    ]
                )
            )
            metrics_lines.append(
                " ".join(
                    [
                        f"cf={_fmt_float(row['counterfactual_available_rate'])}",
                        f"sw={_fmt_float(row['counterfactual_switch_rate'])}",
                        f"cfd={_fmt_float(row['counterfactual_mean_relative_delta'])}",
                    ]
                )
            )
            metrics_lines.append(
                " ".join(
                    [
                        f"dep={_fmt_float(row['trajectory_depot_returns'])}",
                        f"hop={_fmt_float(row['trajectory_customer_hop_distance'])}",
                        f"late_cap={_fmt_float(row['trajectory_late_capacity_share'])}",
                    ]
                )
            )
            metrics_lines.append(
                " ".join(
                    [
                        f"tw={_fmt_float(row['tw_inactive_mass'])}",
                        f"rec={_fmt_float(row['recourse_rate'])}",
                        f"feas={_fmt_float(row['chosen_feasible_rate'])}",
                    ]
                )
            )

            table.add_row(
                _cell_text(row["model"]),
                "\n".join(setup_lines),
                "\n".join(metrics_lines),
            )
        console.print(table)
        return

    table = Table(
        title="Explanation Evaluation",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("model", style="bold", overflow="fold", max_width=42)
    table.add_column("variants", overflow="fold", max_width=32)
    if aggregated:
        table.add_column("runs", justify="right")
        table.add_column("seeds", justify="right")
        table.add_column("seed_range", no_wrap=True, max_width=12)
    table.add_column("steps", justify="right")
    table.add_column("focus@1", justify="right")
    table.add_column("focus@3", justify="right")
    table.add_column("clarity", justify="right")
    table.add_column("contrast", justify="right")
    table.add_column("alt_rate", justify="right")
    table.add_column("opt_cons", justify="right")
    table.add_column("cf_avail", justify="right")
    table.add_column("cf_switch", justify="right")
    table.add_column("cf_feas", justify="right")
    table.add_column("cf_delta", justify="right")
    table.add_column("depots", justify="right")
    table.add_column("hop", justify="right")
    table.add_column("late_cap", justify="right")
    table.add_column("tw_off", justify="right")
    table.add_column("recourse", justify="right")
    table.add_column("feasible", justify="right")

    for row in rows:
        cells = [
            str(row["model"]),
            str(row["variant_mix"]),
        ]
        if aggregated:
            cells.extend(
                [
                    str(int(row["runs"])),
                    str(int(row["seeds"])),
                    str(row["seed_range"]),
                ]
            )
        cells.extend(
            [
                _fmt_float(row["stored_steps"], ndigits=1)
                if aggregated
                else str(int(row["stored_steps"])),
                _fmt_float(row["focus_top1"]),
                _fmt_float(row["focus_top3"]),
                _fmt_float(row["clarity"]),
                _fmt_float(row["contrast_gap"]),
                _fmt_float(row["contrast_alt_rate"]),
                _fmt_float(row["optional_consistency"]),
                _fmt_float(row["counterfactual_available_rate"]),
                _fmt_float(row["counterfactual_switch_rate"]),
                _fmt_float(row["counterfactual_make_feasible_rate"]),
                _fmt_float(row["counterfactual_mean_relative_delta"]),
                _fmt_float(row["trajectory_depot_returns"]),
                _fmt_float(row["trajectory_customer_hop_distance"]),
                _fmt_float(row["trajectory_late_capacity_share"]),
                _fmt_float(row["tw_inactive_mass"]),
                _fmt_float(row["recourse_rate"]),
                _fmt_float(row["chosen_feasible_rate"]),
            ]
        )
        table.add_row(*cells)
    console.print(table)


def _print_stability_table(rows: List[Dict[str, Any]], layout: str = "auto") -> None:
    console, compact = _table_console(layout, compact_threshold=130)
    if not rows:
        console.print(
            "[yellow]No comparable repeated runs for stability analysis[/yellow]"
        )
        return

    if compact:
        table = Table(
            title="Stability Across Repeated Reports",
            box=box.SIMPLE_HEAVY,
            header_style="bold green",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", ratio=3)
        table.add_column("counts", overflow="fold", ratio=2)
        table.add_column("stats", overflow="fold", ratio=2)

        for row in rows:
            table.add_row(
                _cell_text(row["model"]),
                "\n".join(
                    [
                        f"runs={int(row['runs'])}",
                        f"pairs={int(row['pairs'])}",
                        f"steps={_fmt_float(row['compared_steps'], ndigits=1)}",
                    ]
                ),
                "\n".join(
                    [
                        f"ov1={_fmt_float(row['overlap_at_1'])}",
                        f"ov3={_fmt_float(row['overlap_at_3'])}",
                        f"con={_fmt_float(row['constraint_top1_match'])}",
                        f"alt={_fmt_float(row['alt_action_match'])}",
                    ]
                ),
            )
        console.print(table)
        return

    table = Table(
        title="Stability Across Repeated Reports",
        box=box.SIMPLE_HEAVY,
        header_style="bold green",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("model", style="bold", overflow="fold", max_width=52)
    table.add_column("runs", justify="right")
    table.add_column("pairs", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("ov@1", justify="right")
    table.add_column("ov@3", justify="right")
    table.add_column("constraint@1", justify="right")
    table.add_column("alt_match", justify="right")

    for row in rows:
        table.add_row(
            str(row["model"]),
            str(int(row["runs"])),
            str(int(row["pairs"])),
            _fmt_float(row["compared_steps"], ndigits=1),
            _fmt_float(row["overlap_at_1"]),
            _fmt_float(row["overlap_at_3"]),
            _fmt_float(row["constraint_top1_match"]),
            _fmt_float(row["alt_action_match"]),
        )
    console.print(table)


def _print_robustness_table(rows: List[Dict[str, Any]], layout: str = "auto") -> None:
    console, compact = _table_console(layout, compact_threshold=155)
    if not rows:
        console.print("[yellow]No multi-seed groups for robustness analysis[/yellow]")
        return

    if compact:
        table = Table(
            title="Robustness Across Different Seeds",
            box=box.SIMPLE_HEAVY,
            header_style="bold magenta",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", ratio=3)
        table.add_column("seed info", overflow="fold", ratio=2)
        table.add_column("means", overflow="fold", ratio=2)
        table.add_column("std", overflow="fold", ratio=2)

        for row in rows:
            table.add_row(
                _cell_text(row["model"]),
                "\n".join(
                    [
                        f"runs={int(row['runs'])}",
                        f"seeds={int(row['seeds'])}",
                        f"range={int(row['seed_min'])}..{int(row['seed_max'])}",
                    ]
                ),
                _compact_metric_lines(
                    [
                        ("f1", row["focus_top1_mean"]),
                        ("clr", row["clarity_mean"]),
                        ("ctr", row["contrast_gap_mean"]),
                        ("cf", row["counterfactual_available_rate_mean"]),
                        ("sw", row["counterfactual_switch_rate_mean"]),
                        ("cfd", row["counterfactual_mean_relative_delta_mean"]),
                        ("dep", row["trajectory_depot_returns_mean"]),
                        ("hop", row["trajectory_customer_hop_distance_mean"]),
                        ("late_cap", row["trajectory_late_capacity_share_mean"]),
                        ("opt", row["optional_consistency_mean"]),
                        ("tw", row["tw_inactive_mass_mean"]),
                        ("rec", row["recourse_rate_mean"]),
                        ("feas", row["chosen_feasible_rate_mean"]),
                    ]
                ),
                _compact_metric_lines(
                    [
                        ("f1_sd", row["focus_top1_std"]),
                        ("clr_sd", row["clarity_std"]),
                        ("ctr_sd", row["contrast_gap_std"]),
                        ("cf_sd", row["counterfactual_available_rate_std"]),
                        ("sw_sd", row["counterfactual_switch_rate_std"]),
                        ("cfd_sd", row["counterfactual_mean_relative_delta_std"]),
                        ("dep_sd", row["trajectory_depot_returns_std"]),
                        ("hop_sd", row["trajectory_customer_hop_distance_std"]),
                        ("late_sd", row["trajectory_late_capacity_share_std"]),
                        ("opt_sd", row["optional_consistency_std"]),
                        ("tw_sd", row["tw_inactive_mass_std"]),
                        ("rec_sd", row["recourse_rate_std"]),
                        ("feas_sd", row["chosen_feasible_rate_std"]),
                    ]
                ),
            )
        console.print(table)
        return

    table = Table(
        title="Robustness Across Different Seeds",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        expand=True,
        collapse_padding=True,
    )
    table.add_column("model", style="bold", overflow="fold", max_width=42)
    table.add_column("runs", justify="right")
    table.add_column("seeds", justify="right")
    table.add_column("seed_range", no_wrap=True, max_width=12)
    table.add_column("focus@1", justify="right")
    table.add_column("focus_sd", justify="right")
    table.add_column("clarity", justify="right")
    table.add_column("clarity_sd", justify="right")
    table.add_column("contrast", justify="right")
    table.add_column("contrast_sd", justify="right")
    table.add_column("cf_avail", justify="right")
    table.add_column("cf_sd", justify="right")
    table.add_column("cf_switch", justify="right")
    table.add_column("sw_sd", justify="right")
    table.add_column("cf_delta", justify="right")
    table.add_column("cfd_sd", justify="right")
    table.add_column("depots", justify="right")
    table.add_column("dep_sd", justify="right")
    table.add_column("hop", justify="right")
    table.add_column("hop_sd", justify="right")
    table.add_column("late_cap", justify="right")
    table.add_column("late_sd", justify="right")
    table.add_column("opt_cons", justify="right")
    table.add_column("opt_sd", justify="right")
    table.add_column("tw_off", justify="right")
    table.add_column("tw_sd", justify="right")
    table.add_column("recourse", justify="right")
    table.add_column("recourse_sd", justify="right")
    table.add_column("feasible", justify="right")
    table.add_column("feas_sd", justify="right")

    for row in rows:
        table.add_row(
            str(row["model"]),
            str(int(row["runs"])),
            str(int(row["seeds"])),
            f"{int(row['seed_min'])}..{int(row['seed_max'])}",
            _fmt_float(row["focus_top1_mean"]),
            _fmt_float(row["focus_top1_std"]),
            _fmt_float(row["clarity_mean"]),
            _fmt_float(row["clarity_std"]),
            _fmt_float(row["contrast_gap_mean"]),
            _fmt_float(row["contrast_gap_std"]),
            _fmt_float(row["counterfactual_available_rate_mean"]),
            _fmt_float(row["counterfactual_available_rate_std"]),
            _fmt_float(row["counterfactual_switch_rate_mean"]),
            _fmt_float(row["counterfactual_switch_rate_std"]),
            _fmt_float(row["counterfactual_mean_relative_delta_mean"]),
            _fmt_float(row["counterfactual_mean_relative_delta_std"]),
            _fmt_float(row["trajectory_depot_returns_mean"]),
            _fmt_float(row["trajectory_depot_returns_std"]),
            _fmt_float(row["trajectory_customer_hop_distance_mean"]),
            _fmt_float(row["trajectory_customer_hop_distance_std"]),
            _fmt_float(row["trajectory_late_capacity_share_mean"]),
            _fmt_float(row["trajectory_late_capacity_share_std"]),
            _fmt_float(row["optional_consistency_mean"]),
            _fmt_float(row["optional_consistency_std"]),
            _fmt_float(row["tw_inactive_mass_mean"]),
            _fmt_float(row["tw_inactive_mass_std"]),
            _fmt_float(row["recourse_rate_mean"]),
            _fmt_float(row["recourse_rate_std"]),
            _fmt_float(row["chosen_feasible_rate_mean"]),
            _fmt_float(row["chosen_feasible_rate_std"]),
        )
    console.print(table)


def _write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    import csv

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate explanation quality and stability across action_explainer reports."
    )
    parser.add_argument(
        "--pattern",
        default="logs/xai/action_explainer_*.json",
        help="Glob pattern for reports.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Only inspect latest N reports by mtime.",
    )
    parser.add_argument(
        "--sort-by",
        default="clarity",
        choices=[
            "focus_top1",
            "focus_top3",
            "clarity",
            "contrast_gap",
            "counterfactual_available_rate",
            "counterfactual_switch_rate",
            "counterfactual_make_feasible_rate",
            "counterfactual_mean_relative_delta",
            "trajectory_depot_returns",
            "trajectory_depot_share",
            "trajectory_customer_hop_distance",
            "trajectory_recourse_burst_count",
            "trajectory_late_capacity_share",
            "optional_consistency",
            "recourse_rate",
        ],
        help="Sort key for per-report table.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for per-report evaluation.",
    )
    parser.add_argument(
        "--stability-csv",
        default=None,
        help="Optional CSV path for stability table.",
    )
    parser.add_argument(
        "--stability-mode",
        choices=["reproducibility", "robustness"],
        default="reproducibility",
        help=(
            "reproducibility compares same-seed repeated runs step-by-step; "
            "robustness compares different seeds via cross-seed dispersion of aggregate metrics."
        ),
    )
    parser.add_argument(
        "--aggregate-by-model",
        action="store_true",
        help=(
            "Collapse the top evaluation table to one row per comparable model setting. "
            "This is enabled automatically in robustness mode."
        ),
    )
    parser.add_argument(
        "--table-layout",
        choices=["auto", "wide", "compact"],
        default="auto",
        help=(
            "Table rendering layout. 'auto' switches to a compact multiline table "
            "on narrow terminals."
        ),
    )
    args = parser.parse_args()

    reports = _load_reports(args.pattern, args.latest)
    aggregate_report_rows = args.aggregate_by_model or args.stability_mode == "robustness"
    if aggregate_report_rows:
        report_rows = _aggregate_report_rows(reports)
    else:
        report_rows = [_evaluate_single_report(report) for report in reports]
    report_rows = sorted(
        report_rows,
        key=lambda row: (
            float("-inf")
            if not math.isfinite(float(row.get(args.sort_by, float("nan"))))
            else float(row[args.sort_by])
        ),
        reverse=not args.ascending,
    )

    render_layout = "compact" if args.table_layout == "compact" else "auto"
    if args.table_layout == "wide":
        render_layout = "wide"

    _print_report_table(report_rows, layout=render_layout)
    if args.stability_mode == "robustness":
        stability_rows = _evaluate_robustness(reports)
        _print_robustness_table(stability_rows, layout=render_layout)
    else:
        stability_rows = _evaluate_reproducibility(reports)
        _print_stability_table(stability_rows, layout=render_layout)

    if args.output_csv:
        _write_csv(report_rows, args.output_csv)
        print(f"\nWrote evaluation CSV: {args.output_csv}")
    if args.stability_csv:
        _write_csv(stability_rows, args.stability_csv)
        print(f"Wrote stability CSV: {args.stability_csv}")


if __name__ == "__main__":
    main()
