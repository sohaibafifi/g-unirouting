import argparse
import json
import re

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

CONSTRAINT_COLORS = {
    "space_distance": "#4c78a8",
    "time_windows_service": "#f58518",
    "capacity_demands": "#54a24b",
    "route_structure": "#e45756",
    "route_recourse": "#e45756",
    "other": "#b279a2",
}

CONSTRAINT_LABELS = {
    "space_distance": "space / distance",
    "time_windows_service": "time windows / service",
    "capacity_demands": "capacity / demands",
    "route_structure": "route structure",
    "route_recourse": "route structure",
    "other": "other",
}


def _parse_instance_ids(raw: str | None, max_available: int, default_n: int) -> List[int]:
    if raw is None:
        return list(range(min(default_n, max_available)))
    ids = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if 0 <= idx < max_available:
            ids.append(idx)
    if not ids:
        return list(range(min(default_n, max_available)))
    return sorted(set(ids))


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-_.")
    return text.lower() or "unknown"


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


def _plot_global_trajectory_summary(
    fig,
    axes,
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

    for ax in axes:
        ax.set_xlim(-0.5, max(len(actions) - 0.5, 0.5))
    fig.align_ylabels(axes)


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
        "--instances",
        default=None,
        help="Comma-separated instance IDs to plot (default: first N)",
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

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    instances = report.get("instances", None)
    if not instances:
        raise ValueError(
            "Report does not contain 'instances'. Re-run action_explainer with +xai.save_instance_traces=True"
        )

    instance_ids = _parse_instance_ids(args.instances, len(instances), args.num_instances)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = _model_label_from_report(report)
    model_slug = _slugify(model_label)
    report_id = _report_id_from_path(report_path, model_slug)
    node_importance_mode = report.get("config", {}).get(
        "node_importance_mode", "decision-only"
    )
    selected_steps_arg = (
        args.steps if args.steps is not None else (args.step_legacy or "0")
    )
    written_files = []

    for idx in instance_ids:
        trace = instances[idx]
        locs = np.array(trace["locs"], dtype=float)
        actions = [int(a) for a in trace.get("actions", [])]
        done_before = [bool(v) for v in trace.get("done_before", [])]
        done_step = trace.get("done_step", None)
        top_constraints_all = trace.get("top_constraints", [])
        recourse_flags = [bool(v) for v in trace.get("recourse_triggered", [])]
        chosen_feasible_all = [bool(v) for v in trace.get("chosen_feasible", [])]

        if not actions:
            continue

        global_fig, global_axes = plt.subplots(
            3, 1, figsize=(12, 7.2), constrained_layout=True
        )
        _plot_global_trajectory_summary(
            global_fig,
            list(global_axes),
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
        global_out = output_dir / (f"{report_id}_inst{idx:03d}_trajectory.png")
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

            out_path = output_dir / (f"{report_id}_inst{idx:03d}_step{step:03d}.png")
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            written_files.append(str(out_path))

    print(f"Generated {len(written_files)} figure(s)")
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
