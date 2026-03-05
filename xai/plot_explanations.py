import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from domain.constants import CONSTRAINT_COLORS, CONSTRAINT_LABELS_EN as CONSTRAINT_LABELS


def _parse_instance_ids(
    raw: str | None, max_available: int, default_n: int
) -> Tuple[List[int], bool]:
    if raw is None:
        return list(range(min(default_n, max_available))), False
    if raw.strip().lower() == "all":
        return list(range(max_available)), True
    ids = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if 0 <= idx < max_available:
            ids.append(idx)
    if not ids:
        return list(range(min(default_n, max_available))), False
    return sorted(set(ids)), False


from utils.text_utils import slugify as _slugify


def _model_label_from_report(report: dict) -> str:
    cfg = report.get("config", {})
    label = str(cfg.get("model_label", "")).strip()
    if label:
        return label

    ckpt = str(cfg.get("checkpoint_path", "")).strip()
    if ckpt:
        ckpt_path = Path(ckpt)
        run_name = (
            ckpt_path.parent.parent.name
            if len(ckpt_path.parents) >= 2
            else ckpt_path.stem
        )
        run_group = (
            ckpt_path.parent.parent.parent.name if len(ckpt_path.parents) >= 3 else ""
        )
        return f"{run_group}/{run_name}" if run_group else run_name

    return "unknown-model"


def _report_id_from_path(report_path: Path, model_slug: str) -> str:
    stem = report_path.stem
    prefix = "action_explainer_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    model_prefix = f"{model_slug}_"
    if stem.startswith(model_prefix):
        stem = stem[len(model_prefix) :]
    # Keep compact, filesystem-safe identifier.
    return _slugify(stem) or "report"


def _figure_name(prefix: str, body: str) -> str:
    if prefix:
        return f"{prefix}_{body}.png"
    return f"{body}.png"


def _top_share_items(payload: object, limit: int = 5) -> list[tuple[str, float]]:
    if not isinstance(payload, dict):
        return []
    items: list[tuple[str, float]] = []
    for key, raw_value in payload.items():
        try:
            value = max(float(raw_value), 0.0)
        except (TypeError, ValueError):
            continue
        items.append((str(key), value))
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:limit]


def _aggregate_step_payload_shares(
    step_payloads: object, field_name: str
) -> dict[str, float]:
    if not isinstance(step_payloads, list) or not step_payloads:
        return {}
    totals: dict[str, float] = {}
    valid_steps = 0
    for payload in step_payloads:
        if not isinstance(payload, list):
            continue
        seen_any = False
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get(field_name, "")).strip()
            if not name:
                continue
            try:
                share = max(float(item.get("share", 0.0)), 0.0)
            except (TypeError, ValueError):
                continue
            totals[name] = totals.get(name, 0.0) + share
            seen_any = True
        if seen_any:
            valid_steps += 1
    if valid_steps <= 0:
        return {}
    return {name: value / valid_steps for name, value in totals.items()}


def _node_before_step(actions: List[int], step: int) -> int:
    node = 0
    for t in range(max(0, step)):
        node = int(actions[t])
    return node


def _clip_step(step: int, actions: List[int], done_step: int | None) -> int:
    if not actions:
        return 0
    if step < 0:
        step = len(actions) - 1
    step = min(step, len(actions) - 1)
    if done_step is not None:
        step = min(step, done_step)
    return step


def _select_steps(raw_step: str, actions: List[int], done_step: int | None) -> List[int]:
    if not actions:
        return []

    max_step = len(actions) - 1
    if done_step is not None:
        max_step = min(max_step, int(done_step))

    raw = raw_step.strip().lower()
    if raw == "all":
        return list(range(max_step + 1))

    step_tokens = [raw] if "," not in raw else [t.strip() for t in raw.split(",")]
    steps: List[int] = []
    for token in step_tokens:
        if not token:
            continue
        step = int(token)
        step = _clip_step(step, actions, done_step)
        steps.append(step)
    if not steps:
        return [0]
    return sorted(set(steps))


def _plot_route(ax, locs: np.ndarray, actions: List[int], done_step: int | None) -> None:
    ax.scatter(locs[1:, 0], locs[1:, 1], s=16, c="#4c78a8", alpha=0.8, label="customers")
    ax.scatter(
        locs[0, 0],
        locs[0, 1],
        s=100,
        marker="*",
        c="#d62728",
        edgecolors="black",
        linewidths=0.7,
        label="depot",
        zorder=4,
    )

    num_edges = len(actions)
    if done_step is not None:
        num_edges = min(num_edges, done_step + 1)

    current = 0
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(num_edges, 1)))
    for t in range(num_edges):
        nxt = int(actions[t])
        ax.plot(
            [locs[current, 0], locs[nxt, 0]],
            [locs[current, 1], locs[nxt, 1]],
            color=colors[t],
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )
        current = nxt

    ax.set_title("Route Trajectory")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)


def _plot_step_explanation(
    ax,
    locs: np.ndarray,
    actions: List[int],
    top_nodes: List[int],
    top_scores: List[float],
    step: int,
    done_before: bool,
    chosen_feasible: bool,
    recourse_triggered: bool,
    recourse_cost_est: float,
) -> None:
    current = _node_before_step(actions, step)
    chosen = int(actions[step])

    ax.scatter(locs[1:, 0], locs[1:, 1], s=14, c="#9ecae1", alpha=0.75)
    ax.scatter(locs[0, 0], locs[0, 1], s=95, marker="*", c="#d62728", edgecolors="black")

    if top_nodes:
        top_nodes_arr = np.array(top_nodes, dtype=int)
        top_scores_arr = np.array(top_scores, dtype=float)
        top_scores_arr = np.maximum(top_scores_arr, 1e-8)
        top_scores_arr = top_scores_arr / top_scores_arr.max()
        sizes = 80 + 220 * top_scores_arr
        ax.scatter(
            locs[top_nodes_arr, 0],
            locs[top_nodes_arr, 1],
            s=sizes,
            c="#ff7f0e",
            alpha=0.55,
            edgecolors="black",
            linewidths=0.6,
            label="top attributed nodes",
            zorder=4,
        )

    ax.scatter(
        locs[current, 0],
        locs[current, 1],
        s=110,
        c="#9467bd",
        marker="s",
        edgecolors="black",
        linewidths=0.8,
        label="current node",
        zorder=5,
    )
    chosen_color = "#d62728" if recourse_triggered else "#2ca02c"
    ax.scatter(
        locs[chosen, 0],
        locs[chosen, 1],
        s=130,
        c=chosen_color,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
        label="chosen action (recourse)" if recourse_triggered else "chosen action",
        zorder=6,
    )

    step_state = "done-state" if done_before else "active-state"
    ax.set_title(f"Step {step} Explanation ({step_state})")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if recourse_triggered:
        ax.text(
            0.02,
            0.98,
            f"RECOURSE\\ninfeasible customer\\ncost~{recourse_cost_est:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="#fee0d2", edgecolor="#de2d26", alpha=0.85),
        )
    elif not chosen_feasible:
        ax.text(
            0.02,
            0.98,
            "Chosen action infeasible\\nunder full constraints",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="#fff5f0", edgecolor="#fb6a4a", alpha=0.85),
        )
    ax.legend(loc="upper right", fontsize=8)


def _dominant_constraint(step_payload: object) -> tuple[str, float]:
    if not isinstance(step_payload, list) or not step_payload:
        return "other", 0.0
    best_name = "other"
    best_share = 0.0
    for item in step_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("constraint", "")).strip() or "other"
        try:
            share = max(float(item.get("share", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        if share > best_share:
            best_name = name
            best_share = share
    return best_name, best_share


def _trajectory_plot_summary(
    locs: np.ndarray,
    actions: List[int],
    top_constraints: List[object],
    recourse_flags: List[bool],
    done_step: int | None,
) -> List[str]:
    if done_step is not None:
        max_step = min(len(actions), int(done_step) + 1)
    else:
        max_step = len(actions)
    actions = actions[:max_step]
    top_constraints = top_constraints[:max_step]
    recourse_flags = recourse_flags[:max_step]
    if not actions:
        return []

    depot_returns = sum(1 for action in actions if int(action) == 0)
    depot_share = depot_returns / max(len(actions), 1)

    customer_hops: List[float] = []
    current = 0
    for action in actions:
        nxt = int(action)
        if (
            current > 0
            and nxt > 0
            and 0 <= current < len(locs)
            and 0 <= nxt < len(locs)
        ):
            customer_hops.append(
                float(np.linalg.norm(locs[current] - locs[nxt]))
            )
        current = nxt
    mean_customer_hop = float(np.mean(customer_hops)) if customer_hops else float("nan")

    split_idx = max(1, len(top_constraints) // 2)
    early = _dominant_constraint(top_constraints[:split_idx])[0]
    late = _dominant_constraint(top_constraints[split_idx:] or top_constraints[:split_idx])[0]
    recourse_count = sum(1 for flag in recourse_flags if bool(flag))

    lines: List[str] = []
    if depot_share >= 0.35:
        lines.append(f"Fragmentation: élevée ({depot_share * 100.0:.1f}% retours dépôt)")
    elif depot_share >= 0.15:
        lines.append(f"Fragmentation: modérée ({depot_share * 100.0:.1f}% retours dépôt)")
    else:
        lines.append(f"Fragmentation: faible ({depot_share * 100.0:.1f}% retours dépôt)")

    if np.isfinite(mean_customer_hop):
        if mean_customer_hop <= 0.14:
            lines.append(f"Compacité: élevée (hop={mean_customer_hop:.3f})")
        elif mean_customer_hop >= 0.20:
            lines.append(f"Compacité: faible (hop={mean_customer_hop:.3f})")
        else:
            lines.append(f"Compacité: intermédiaire (hop={mean_customer_hop:.3f})")

    if recourse_count >= 3:
        lines.append(f"Recours: récurrent ({recourse_count} événements)")
    elif recourse_count >= 1:
        lines.append(f"Recours: ponctuel ({recourse_count} événement{'s' if recourse_count > 1 else ''})")
    else:
        lines.append("Recours: absent")

    early_label = CONSTRAINT_LABELS.get(early, early)
    late_label = CONSTRAINT_LABELS.get(late, late)
    if early_label and late_label and early_label != late_label:
        lines.append(f"Régime: {early_label} -> {late_label}")
    elif early_label:
        lines.append(f"Régime: stable ({early_label})")
    return lines


def _plot_global_trajectory_summary(
    fig,
    axes,
    locs: np.ndarray,
    actions: List[int],
    top_constraints: List[object],
    recourse_flags: List[bool],
    chosen_feasible: List[bool],
    done_step: int | None,
) -> None:
    ax_actions, ax_constraints, ax_events = axes
    if done_step is not None:
        max_step = min(len(actions), int(done_step) + 1)
    else:
        max_step = len(actions)
    actions = actions[:max_step]
    top_constraints = top_constraints[:max_step]
    recourse_flags = recourse_flags[:max_step]
    chosen_feasible = chosen_feasible[:max_step]
    steps = np.arange(len(actions), dtype=int)

    if len(actions) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No stored trajectory", ha="center", va="center")
            ax.set_axis_off()
        return

    summary_lines = _trajectory_plot_summary(
        locs=locs,
        actions=actions,
        top_constraints=top_constraints,
        recourse_flags=recourse_flags,
        done_step=done_step,
    )

    depot_mask = np.array([action == 0 for action in actions], dtype=bool)
    recourse_mask = np.array([bool(flag) for flag in recourse_flags], dtype=bool)
    infeasible_mask = np.array(
        [not bool(flag) for flag in chosen_feasible], dtype=bool
    )

    customer_steps = steps[~depot_mask]
    customer_actions = np.array(actions, dtype=int)[~depot_mask]
    if customer_steps.size > 0:
        ax_actions.scatter(
            customer_steps,
            customer_actions,
            s=18,
            c="#4c78a8",
            alpha=0.75,
            label="client actions",
        )
    if depot_mask.any():
        ax_actions.scatter(
            steps[depot_mask],
            np.zeros(int(depot_mask.sum()), dtype=float),
            s=48,
            marker="*",
            c="#d62728",
            edgecolors="black",
            linewidths=0.5,
            label="depot returns",
            zorder=4,
        )
    if recourse_mask.any():
        recourse_steps = steps[recourse_mask]
        recourse_actions = np.array(actions, dtype=int)[recourse_mask]
        ax_actions.scatter(
            recourse_steps,
            recourse_actions,
            s=64,
            marker="x",
            c="#e45756",
            linewidths=1.4,
            label="recourse events",
            zorder=5,
        )
    if infeasible_mask.any():
        infeasible_steps = steps[infeasible_mask]
        infeasible_actions = np.array(actions, dtype=int)[infeasible_mask]
        ax_actions.scatter(
            infeasible_steps,
            infeasible_actions,
            s=42,
            marker="s",
            facecolors="none",
            edgecolors="#f58518",
            linewidths=0.9,
            label="infeasible before recourse",
            zorder=4,
        )
    ax_actions.set_ylabel("chosen node")
    ax_actions.set_xlabel("step")
    ax_actions.set_title("Action Timeline")
    ax_actions.grid(alpha=0.2)
    ax_actions.legend(loc="upper right", fontsize=7, ncols=2)
    if summary_lines:
        ax_actions.text(
            0.01,
            0.98,
            "\n".join(summary_lines[:4]),
            transform=ax_actions.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="#9e9e9e", alpha=0.9),
        )

    dom_names: List[str] = []
    dom_shares: List[float] = []
    dom_colors: List[str] = []
    for payload in top_constraints:
        name, share = _dominant_constraint(payload)
        dom_names.append(name)
        dom_shares.append(share)
        dom_colors.append(CONSTRAINT_COLORS.get(name, CONSTRAINT_COLORS["other"]))
    ax_constraints.bar(
        steps,
        dom_shares,
        width=0.9,
        color=dom_colors,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.2,
    )
    ax_constraints.set_ylim(0.0, 1.0)
    ax_constraints.set_ylabel("dominant attribution share")
    ax_constraints.set_xlabel("step")
    ax_constraints.set_title("Dominant Constraint Family Per Step (Attribution)")
    ax_constraints.grid(axis="y", alpha=0.2)
    legend_handles = []
    legend_labels = []
    for name, color in CONSTRAINT_COLORS.items():
        if name in dom_names:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=6))
            legend_labels.append(CONSTRAINT_LABELS.get(name, name))
    if legend_handles:
        ax_constraints.legend(
            legend_handles, legend_labels, loc="upper right", fontsize=7, ncols=2
        )

    depot_cum = np.cumsum(depot_mask.astype(int))
    recourse_cum = np.cumsum(recourse_mask.astype(int))
    ax_events.plot(
        steps,
        depot_cum,
        color="#d62728",
        linewidth=1.8,
        label="cumulative depot returns",
    )
    ax_events.plot(
        steps,
        recourse_cum,
        color="#e45756",
        linewidth=1.8,
        linestyle="--",
        label="cumulative recourse",
    )
    ax_events.set_ylabel("count")
    ax_events.set_xlabel("step")
    ax_events.set_title("Event Accumulation")
    ax_events.grid(alpha=0.2)
    ax_events.legend(loc="upper left", fontsize=7)
    ax_events.text(
        0.99,
        0.02,
        "Guide de lecture:\n"
        "- Fragmentation = retours dépôt\n"
        "- Compacité = distance client-client\n"
        "- Recours = secours cumulés",
        transform=ax_events.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="#9e9e9e", alpha=0.9),
    )

    for ax in axes:
        ax.set_xlim(-0.5, max(len(actions) - 0.5, 0.5))
    fig.align_ylabels(axes)


def _plot_method_summary(
    report: dict,
    report_path: Path,
    output_dir: Path,
    filename_prefix: str,
    method_label: str | None = None,
) -> str | None:
    summary = report.get("summary", {}) or {}
    if not isinstance(summary, dict) or not summary:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = _model_label_from_report(report)
    node_importance_mode = report.get("config", {}).get(
        "node_importance_mode", "decision-only"
    )
    method_text = method_label or str(
        report.get("config", {}).get("attribution_method", "")
    ).strip()
    method_text = method_text or "gradient"

    contrastive = summary.get("contrastive", {}) or {}
    counterfactuals = summary.get("counterfactuals", {}) or {}
    trajectory = summary.get("trajectory", {}) or {}
    deletion = summary.get("deletion_faithfulness", {}) or {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_rates, ax_constraints, ax_features, ax_text = axes.flatten()

    rate_metrics = [
        ("feasible", float(summary.get("chosen_action_feasible_rate", 0.0) or 0.0)),
        ("recourse", float(summary.get("recourse_event_rate", 0.0) or 0.0)),
        ("alt", float(contrastive.get("alt_available_rate", 0.0) or 0.0)),
        ("cf_switch", float(counterfactuals.get("switch_rate", 0.0) or 0.0)),
    ]
    deletion_top1 = deletion.get("1", {}) if isinstance(deletion.get("1", {}), dict) else {}
    if deletion_top1:
        rate_metrics.append(
            ("flip@1", float(deletion_top1.get("mean_action_flip_rate", 0.0) or 0.0))
        )
    labels = [item[0] for item in rate_metrics]
    values = [min(max(item[1], 0.0), 1.0) for item in rate_metrics]
    y_pos = np.arange(len(labels), dtype=float)
    ax_rates.barh(y_pos, values, color="#4c78a8", alpha=0.9)
    ax_rates.set_yticks(y_pos, labels)
    ax_rates.set_xlim(0.0, 1.0)
    ax_rates.set_xlabel("rate")
    ax_rates.set_title("Key Rates")
    ax_rates.grid(axis="x", alpha=0.2)
    for y, value in zip(y_pos, values):
        ax_rates.text(min(value + 0.015, 0.98), y, f"{value:.3f}", va="center", fontsize=8)

    constraint_items = _top_share_items(summary.get("constraint_importance_share", {}))
    if constraint_items:
        c_labels = [CONSTRAINT_LABELS.get(name, name) for name, _ in constraint_items]
        c_values = [value for _, value in constraint_items]
        c_pos = np.arange(len(c_labels), dtype=float)
        c_colors = [
            CONSTRAINT_COLORS.get(name, CONSTRAINT_COLORS["other"])
            for name, _ in constraint_items
        ]
        ax_constraints.barh(c_pos, c_values, color=c_colors, alpha=0.9)
        ax_constraints.set_yticks(c_pos, c_labels)
        ax_constraints.set_xlim(0.0, max(max(c_values) * 1.15, 0.05))
        ax_constraints.set_xlabel("share")
        ax_constraints.set_title("Constraint Families")
        ax_constraints.grid(axis="x", alpha=0.2)
    else:
        ax_constraints.text(0.5, 0.5, "No constraint summary", ha="center", va="center")
        ax_constraints.set_axis_off()

    feature_items = _top_share_items(summary.get("feature_importance_share", {}))
    if feature_items:
        f_labels = [name for name, _ in feature_items]
        f_values = [value for _, value in feature_items]
        f_pos = np.arange(len(f_labels), dtype=float)
        ax_features.barh(f_pos, f_values, color="#f58518", alpha=0.9)
        ax_features.set_yticks(f_pos, f_labels)
        ax_features.set_xlim(0.0, max(max(f_values) * 1.15, 0.05))
        ax_features.set_xlabel("share")
        ax_features.set_title("Top Input Features")
        ax_features.grid(axis="x", alpha=0.2)
    else:
        ax_features.text(0.5, 0.5, "No feature summary", ha="center", va="center")
        ax_features.set_axis_off()

    early_name = str(trajectory.get("early_top_constraint", "")).strip()
    late_name = str(trajectory.get("late_top_constraint", "")).strip()
    summary_lines = [
        f"model: {model_label}",
        f"method: {method_text}",
        f"importance: {node_importance_mode}",
        f"instances: {int(summary.get('num_instances', 0) or 0)}",
        f"steps: {int(summary.get('num_steps', 0) or 0)}",
        f"logit gap: {float(contrastive.get('mean_logit_gap', 0.0) or 0.0):.3f}",
        f"logprob gap: {float(contrastive.get('mean_logprob_gap', 0.0) or 0.0):.3f}",
        f"depot returns: {float(trajectory.get('mean_depot_returns_per_instance', 0.0) or 0.0):.2f}",
        f"customer hop: {float(trajectory.get('mean_customer_hop_distance', 0.0) or 0.0):.3f}",
        f"recourse events: {float(trajectory.get('mean_recourse_events_per_instance', 0.0) or 0.0):.2f}",
    ]
    if early_name or late_name:
        early_label = CONSTRAINT_LABELS.get(early_name, early_name or "n/a")
        late_label = CONSTRAINT_LABELS.get(late_name, late_name or "n/a")
        if early_label == late_label and early_label != "n/a":
            summary_lines.append(f"regime: stable ({early_label})")
        else:
            summary_lines.append(f"regime: {early_label} -> {late_label}")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#9e9e9e", alpha=0.92),
    )
    ax_text.set_title("Trajectory / Global Summary")
    ax_text.set_axis_off()

    fig.suptitle(
        f"Model={model_label} | method={method_text} | importance={node_importance_mode} | summary",
        fontsize=11,
    )
    out_path = output_dir / _figure_name(filename_prefix, "summary")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _plot_instance_summary(
    trace: dict,
    output_dir: Path,
    model_label: str,
    node_importance_mode: str,
    method_label: str,
    instance_index: int,
) -> str | None:
    actions = [int(a) for a in trace.get("actions", [])]
    if not actions:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    locs = np.array(trace.get("locs", []), dtype=float)
    top_constraints_all = trace.get("top_constraints", []) or []
    top_features_all = trace.get("top_features", []) or []
    recourse_flags = [bool(v) for v in trace.get("recourse_triggered", [])]
    chosen_feasible = [bool(v) for v in trace.get("chosen_feasible", [])]
    counterfactuals = trace.get("counterfactuals", []) or []

    depot_returns = sum(1 for action in actions if action == 0)
    depot_share = depot_returns / max(len(actions), 1)
    feasible_rate = (
        float(np.mean(np.array(chosen_feasible, dtype=float)))
        if chosen_feasible
        else 1.0
    )
    recourse_rate = (
        float(np.mean(np.array(recourse_flags, dtype=float)))
        if recourse_flags
        else 0.0
    )
    cf_switch = 0
    cf_available = 0
    rel_deltas: list[float] = []
    for item in counterfactuals:
        if not isinstance(item, dict):
            continue
        cf_available += 1
        if str(item.get("status", "")).strip() == "switch":
            cf_switch += 1
        try:
            rel_delta = float(
                item.get("relative_delta", item.get("estimated_delta", item.get("delta", 0.0)))
            )
        except (TypeError, ValueError):
            continue
        if math.isfinite(rel_delta):
            rel_deltas.append(abs(rel_delta))
    cf_switch_rate = cf_switch / cf_available if cf_available > 0 else 0.0

    customer_hops: list[float] = []
    current = 0
    for action in actions:
        nxt = int(action)
        if (
            current > 0
            and nxt > 0
            and len(locs) > max(current, nxt)
        ):
            customer_hops.append(float(np.linalg.norm(locs[current] - locs[nxt])))
        current = nxt
    hop_mean = float(np.mean(customer_hops)) if customer_hops else float("nan")

    constraint_share = _aggregate_step_payload_shares(top_constraints_all, "constraint")
    feature_share = _aggregate_step_payload_shares(top_features_all, "feature")
    constraint_items = _top_share_items(constraint_share)
    feature_items = _top_share_items(feature_share)

    split_idx = max(1, len(top_constraints_all) // 2)
    early_name = _dominant_constraint(top_constraints_all[:split_idx])[0]
    late_name = _dominant_constraint(
        top_constraints_all[split_idx:] or top_constraints_all[:split_idx]
    )[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax_rates, ax_constraints, ax_features, ax_text = axes.flatten()

    rate_metrics = [
        ("feasible", feasible_rate),
        ("recourse", recourse_rate),
        ("depot", depot_share),
        ("cf_switch", cf_switch_rate),
    ]
    labels = [item[0] for item in rate_metrics]
    values = [min(max(item[1], 0.0), 1.0) for item in rate_metrics]
    y_pos = np.arange(len(labels), dtype=float)
    ax_rates.barh(y_pos, values, color="#4c78a8", alpha=0.9)
    ax_rates.set_yticks(y_pos, labels)
    ax_rates.set_xlim(0.0, 1.0)
    ax_rates.set_xlabel("rate")
    ax_rates.set_title("Instance Rates")
    ax_rates.grid(axis="x", alpha=0.2)
    for y, value in zip(y_pos, values):
        ax_rates.text(min(value + 0.015, 0.98), y, f"{value:.3f}", va="center", fontsize=8)

    if constraint_items:
        c_labels = [CONSTRAINT_LABELS.get(name, name) for name, _ in constraint_items]
        c_values = [value for _, value in constraint_items]
        c_pos = np.arange(len(c_labels), dtype=float)
        c_colors = [
            CONSTRAINT_COLORS.get(name, CONSTRAINT_COLORS["other"])
            for name, _ in constraint_items
        ]
        ax_constraints.barh(c_pos, c_values, color=c_colors, alpha=0.9)
        ax_constraints.set_yticks(c_pos, c_labels)
        ax_constraints.set_xlim(0.0, max(max(c_values) * 1.15, 0.05))
        ax_constraints.set_xlabel("avg share")
        ax_constraints.set_title("Constraint Families")
        ax_constraints.grid(axis="x", alpha=0.2)
    else:
        ax_constraints.text(0.5, 0.5, "No constraint trace", ha="center", va="center")
        ax_constraints.set_axis_off()

    if feature_items:
        f_labels = [name for name, _ in feature_items]
        f_values = [value for _, value in feature_items]
        f_pos = np.arange(len(f_labels), dtype=float)
        ax_features.barh(f_pos, f_values, color="#f58518", alpha=0.9)
        ax_features.set_yticks(f_pos, f_labels)
        ax_features.set_xlim(0.0, max(max(f_values) * 1.15, 0.05))
        ax_features.set_xlabel("avg share")
        ax_features.set_title("Top Input Features")
        ax_features.grid(axis="x", alpha=0.2)
    else:
        ax_features.text(0.5, 0.5, "No feature trace", ha="center", va="center")
        ax_features.set_axis_off()

    text_lines = [
        f"instance: {instance_index}",
        f"variant: {str(trace.get('instance_variant_code', 'n/a'))}",
        f"steps: {len(actions)}",
        f"depot returns: {depot_returns}",
        f"customer hop: {hop_mean:.3f}" if math.isfinite(hop_mean) else "customer hop: n/a",
        f"recourse events: {sum(1 for flag in recourse_flags if flag)}",
        f"cf available: {cf_available}",
        (
            f"mean rel delta: {float(np.mean(rel_deltas)):.3f}"
            if rel_deltas
            else "mean rel delta: n/a"
        ),
    ]
    early_label = CONSTRAINT_LABELS.get(early_name, early_name or "n/a")
    late_label = CONSTRAINT_LABELS.get(late_name, late_name or "n/a")
    if early_label == late_label and early_label != "n/a":
        text_lines.append(f"regime: stable ({early_label})")
    else:
        text_lines.append(f"regime: {early_label} -> {late_label}")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#9e9e9e", alpha=0.92),
    )
    ax_text.set_title("Trace Summary")
    ax_text.set_axis_off()

    fig.suptitle(
        f"Model={model_label} | method={method_label} | importance={node_importance_mode} | instance={instance_index} | summary",
        fontsize=11,
    )
    out_path = output_dir / "summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_single_report(
    report: dict,
    report_path: Path,
    output_dir: Path,
    instances_arg: str | None,
    num_instances: int,
    selected_steps_arg: str,
    filename_prefix: str | None = None,
) -> List[str]:
    instances = report.get("instances", None)
    if not instances:
        raise ValueError(
            f"Report does not contain 'instances': {report_path}. "
            "Re-run action_explainer with saved instance traces enabled."
        )

    instance_ids, all_instances_mode = _parse_instance_ids(
        instances_arg, len(instances), num_instances
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = _model_label_from_report(report)
    model_slug = _slugify(model_label)
    report_id = _report_id_from_path(report_path, model_slug)
    stem_prefix = report_id if filename_prefix is None else filename_prefix
    node_importance_mode = report.get("config", {}).get(
        "node_importance_mode", "decision-only"
    )
    method_text = str(report.get("config", {}).get("attribution_method", "")).strip()
    method_text = method_text or "gradient"
    written_files: List[str] = []

    for idx in instance_ids:
        trace = instances[idx]
        instance_output_dir = (
            output_dir / f"inst{idx:03d}" if all_instances_mode else output_dir
        )
        instance_output_dir.mkdir(parents=True, exist_ok=True)
        locs = np.array(trace["locs"], dtype=float)
        actions = [int(a) for a in trace.get("actions", [])]
        done_before = [bool(v) for v in trace.get("done_before", [])]
        done_step = trace.get("done_step", None)
        top_constraints_all = trace.get("top_constraints", [])
        recourse_flags = [bool(v) for v in trace.get("recourse_triggered", [])]
        chosen_feasible_all = [bool(v) for v in trace.get("chosen_feasible", [])]

        if not actions:
            continue

        if all_instances_mode:
            instance_summary = _plot_instance_summary(
                trace=trace,
                output_dir=instance_output_dir,
                model_label=model_label,
                node_importance_mode=node_importance_mode,
                method_label=method_text,
                instance_index=idx,
            )
            if instance_summary:
                written_files.append(instance_summary)

        global_fig, global_axes = plt.subplots(
            3, 1, figsize=(12, 7.2), constrained_layout=True
        )
        _plot_global_trajectory_summary(
            global_fig,
            list(global_axes),
            locs,
            actions,
            top_constraints_all,
            recourse_flags,
            chosen_feasible_all,
            done_step,
        )
        global_fig.suptitle(
            f"Model={model_label} | importance={node_importance_mode} | instance={idx} | trajectory",
            fontsize=11,
        )
        trajectory_body = (
            "trajectory" if all_instances_mode else f"inst{idx:03d}_trajectory"
        )
        global_out = instance_output_dir / _figure_name(stem_prefix, trajectory_body)
        global_fig.savefig(global_out, dpi=180)
        plt.close(global_fig)
        written_files.append(str(global_out))

        selected_steps = _select_steps(selected_steps_arg, actions, done_step)
        for step in selected_steps:
            top_nodes_all = trace.get("top_nodes", [])
            top_scores_all = trace.get("top_scores", [])
            top_nodes = (
                [int(v) for v in top_nodes_all[step]] if step < len(top_nodes_all) else []
            )
            top_scores = (
                [float(v) for v in top_scores_all[step]]
                if step < len(top_scores_all)
                else []
            )
            done_before_step = done_before[step] if step < len(done_before) else False
            recourse_costs = trace.get("recourse_cost_est", [])
            recourse_step = (
                bool(recourse_flags[step]) if step < len(recourse_flags) else False
            )
            recourse_cost_step = (
                float(recourse_costs[step]) if step < len(recourse_costs) else 0.0
            )
            chosen_feasible_step = (
                bool(chosen_feasible_all[step])
                if step < len(chosen_feasible_all)
                else True
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
            _plot_route(axes[0], locs, actions, done_step)
            _plot_step_explanation(
                axes[1],
                locs,
                actions,
                top_nodes,
                top_scores,
                step,
                done_before_step,
                chosen_feasible_step,
                recourse_step,
                recourse_cost_step,
            )

            recourse_txt = (
                f" | RECOURSE cost~{recourse_cost_step:.3f}" if recourse_step else ""
            )
            fig.suptitle(
                f"Model={model_label} | importance={node_importance_mode} | instance={idx} | step={step}{recourse_txt}",
                fontsize=11,
            )

            step_body = "step{step:03d}".format(step=step)
            if not all_instances_mode:
                step_body = f"inst{idx:03d}_{step_body}"
            out_path = instance_output_dir / _figure_name(stem_prefix, step_body)
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            written_files.append(str(out_path))

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot route and node-attribution explanations"
    )
    parser.add_argument(
        "--report", required=True, help="Path to action_explainer JSON report"
    )
    parser.add_argument(
        "--output-dir",
        default="logs/xai/figures",
        help="Directory to save generated PNG figures",
    )
    parser.add_argument(
        "--output",
        dest="output_legacy",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="Comma-separated instance IDs to plot (default: first N)",
    )
    parser.add_argument(
        "--instance",
        dest="instance_legacy",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=4,
        help="How many instances to plot by default",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help='Step selection: single index (e.g. "5"), comma list ("0,5,12"), or "all"',
    )
    parser.add_argument(
        "--step",
        dest="step_legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = _load_json(report_path)

    output_dir = Path(args.output_legacy or args.output_dir)
    selected_steps_arg = (
        args.steps if args.steps is not None else (args.step_legacy or "0")
    )
    instances_arg = args.instances if args.instances is not None else args.instance_legacy
    written_files: List[str] = []

    if str(report.get("kind", "")).strip() == "xai_dual_bundle":
        report_refs = report.get("reports", {}) or {}
        for method_key in ["gradient", "integrated_gradients"]:
            ref = report_refs.get(method_key)
            if not isinstance(ref, dict):
                continue
            candidate = Path(str(ref.get("path_resolved") or ref.get("path") or "").strip())
            if not candidate.exists():
                continue
            method_report = _load_json(candidate)
            summary_path = _plot_method_summary(
                report=method_report,
                report_path=candidate,
                output_dir=output_dir,
                filename_prefix=method_key,
                method_label=method_key,
            )
            if summary_path:
                written_files.append(summary_path)
            method_output_dir = output_dir / method_key
            written_files.extend(
                _plot_single_report(
                    report=method_report,
                    report_path=candidate,
                    output_dir=method_output_dir,
                    instances_arg=instances_arg,
                    num_instances=args.num_instances,
                    selected_steps_arg=selected_steps_arg,
                    filename_prefix="",
                )
            )
    else:
        summary_path = _plot_method_summary(
            report=report,
            report_path=report_path,
            output_dir=output_dir,
            filename_prefix=_report_id_from_path(
                report_path, _slugify(_model_label_from_report(report))
            ),
            method_label=str(report.get("config", {}).get("attribution_method", "")).strip(),
        )
        if summary_path:
            written_files.append(summary_path)
        written_files.extend(
            _plot_single_report(
                report=report,
                report_path=report_path,
                output_dir=output_dir,
                instances_arg=instances_arg,
                num_instances=args.num_instances,
                selected_steps_arg=selected_steps_arg,
            )
        )

    print(f"Generated {len(written_files)} figure(s)")
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
