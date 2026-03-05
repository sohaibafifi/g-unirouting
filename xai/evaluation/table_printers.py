"""Rich table printers for XAI evaluation results."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rich import box
from rich.console import Console
from rich.table import Table

from utils.text_utils import fmt_float, cell_text


def _table_console(layout: str, compact_threshold: int) -> Tuple[Console, bool]:
    console = Console()
    width = getattr(console.size, "width", console.width)
    compact = layout == "compact" or (layout == "auto" and width < compact_threshold)
    return console, compact


def _compact_metric_lines(pairs: List[Tuple[str, Any]], ndigits: int = 4) -> str:
    return "\n".join(f"{label}={fmt_float(value, ndigits=ndigits)}" for label, value in pairs)


def _format_deletion_metric_block(
    deletion_metrics: Dict[int, Dict[str, float]], key: str, label: str
) -> str:
    if not deletion_metrics:
        return "-"
    lines = []
    for k in sorted(deletion_metrics.keys()):
        value = deletion_metrics[k].get(key, float("nan"))
        lines.append(f"{label}@{k}={fmt_float(value)}")
    return "\n".join(lines) if lines else "-"


class EvalTablePrinter:
    """Prints the main evaluation metrics table (replaces _print_report_table)."""

    def __init__(self, layout: str = "auto") -> None:
        self.layout = layout

    def print(self, rows: List[Dict[str, Any]]) -> None:
        console, compact = _table_console(self.layout, compact_threshold=150)
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
                setup_lines = [f"variants: {cell_text(row['variant_mix'])}"]
                if aggregated:
                    setup_lines.append(
                        f"runs={int(row['runs'])} seeds={int(row['seeds'])} range={row['seed_range']}"
                    )

                metrics_lines = []
                metrics_lines.append(
                    "steps="
                    + (
                        fmt_float(row["stored_steps"], ndigits=1)
                        if aggregated
                        else str(int(row["stored_steps"]))
                    )
                )
                metrics_lines.append(
                    " ".join(
                        [
                            f"f1={fmt_float(row['focus_top1'])}",
                            f"f3={fmt_float(row['focus_top3'])}",
                            f"clr={fmt_float(row['clarity'])}",
                        ]
                    )
                )
                metrics_lines.append(
                    " ".join(
                        [
                            f"ctr={fmt_float(row['contrast_gap'])}",
                            f"alt={fmt_float(row['contrast_alt_rate'])}",
                            f"opt={fmt_float(row['optional_consistency'])}",
                        ]
                    )
                )
                metrics_lines.append(
                    " ".join(
                        [
                            f"cf={fmt_float(row['counterfactual_available_rate'])}",
                            f"sw={fmt_float(row['counterfactual_switch_rate'])}",
                            f"cfd={fmt_float(row['counterfactual_mean_relative_delta'])}",
                        ]
                    )
                )
                metrics_lines.append(
                    " ".join(
                        [
                            f"dep={fmt_float(row['trajectory_depot_returns'])}",
                            f"hop={fmt_float(row['trajectory_customer_hop_distance'])}",
                            f"late_cap={fmt_float(row['trajectory_late_capacity_share'])}",
                        ]
                    )
                )
                metrics_lines.append(
                    " ".join(
                        [
                            f"tw={fmt_float(row['tw_inactive_mass'])}",
                            f"rec={fmt_float(row['recourse_rate'])}",
                            f"feas={fmt_float(row['chosen_feasible_rate'])}",
                        ]
                    )
                )

                table.add_row(
                    cell_text(row["model"]),
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
                    fmt_float(row["stored_steps"], ndigits=1)
                    if aggregated
                    else str(int(row["stored_steps"])),
                    fmt_float(row["focus_top1"]),
                    fmt_float(row["focus_top3"]),
                    fmt_float(row["clarity"]),
                    fmt_float(row["contrast_gap"]),
                    fmt_float(row["contrast_alt_rate"]),
                    fmt_float(row["optional_consistency"]),
                    fmt_float(row["counterfactual_available_rate"]),
                    fmt_float(row["counterfactual_switch_rate"]),
                    fmt_float(row["counterfactual_make_feasible_rate"]),
                    fmt_float(row["counterfactual_mean_relative_delta"]),
                    fmt_float(row["trajectory_depot_returns"]),
                    fmt_float(row["trajectory_customer_hop_distance"]),
                    fmt_float(row["trajectory_late_capacity_share"]),
                    fmt_float(row["tw_inactive_mass"]),
                    fmt_float(row["recourse_rate"]),
                    fmt_float(row["chosen_feasible_rate"]),
                ]
            )
            table.add_row(*cells)
        console.print(table)


class StabilityTablePrinter:
    """Prints the stability (repeated runs) table (replaces _print_stability_table)."""

    def __init__(self, layout: str = "auto") -> None:
        self.layout = layout

    def print(self, rows: List[Dict[str, Any]]) -> None:
        console, compact = _table_console(self.layout, compact_threshold=130)
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
                    cell_text(row["model"]),
                    "\n".join(
                        [
                            f"runs={int(row['runs'])}",
                            f"pairs={int(row['pairs'])}",
                            f"steps={fmt_float(row['compared_steps'], ndigits=1)}",
                        ]
                    ),
                    "\n".join(
                        [
                            f"ov1={fmt_float(row['overlap_at_1'])}",
                            f"ov3={fmt_float(row['overlap_at_3'])}",
                            f"con={fmt_float(row['constraint_top1_match'])}",
                            f"alt={fmt_float(row['alt_action_match'])}",
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
                fmt_float(row["compared_steps"], ndigits=1),
                fmt_float(row["overlap_at_1"]),
                fmt_float(row["overlap_at_3"]),
                fmt_float(row["constraint_top1_match"]),
                fmt_float(row["alt_action_match"]),
            )
        console.print(table)


class RobustnessTablePrinter:
    """Prints the robustness (multi-seed) table (replaces _print_robustness_table)."""

    def __init__(self, layout: str = "auto") -> None:
        self.layout = layout

    def print(self, rows: List[Dict[str, Any]]) -> None:
        console, compact = _table_console(self.layout, compact_threshold=155)
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
                    cell_text(row["model"]),
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
                fmt_float(row["focus_top1_mean"]),
                fmt_float(row["focus_top1_std"]),
                fmt_float(row["clarity_mean"]),
                fmt_float(row["clarity_std"]),
                fmt_float(row["contrast_gap_mean"]),
                fmt_float(row["contrast_gap_std"]),
                fmt_float(row["counterfactual_available_rate_mean"]),
                fmt_float(row["counterfactual_available_rate_std"]),
                fmt_float(row["counterfactual_switch_rate_mean"]),
                fmt_float(row["counterfactual_switch_rate_std"]),
                fmt_float(row["counterfactual_mean_relative_delta_mean"]),
                fmt_float(row["counterfactual_mean_relative_delta_std"]),
                fmt_float(row["trajectory_depot_returns_mean"]),
                fmt_float(row["trajectory_depot_returns_std"]),
                fmt_float(row["trajectory_customer_hop_distance_mean"]),
                fmt_float(row["trajectory_customer_hop_distance_std"]),
                fmt_float(row["trajectory_late_capacity_share_mean"]),
                fmt_float(row["trajectory_late_capacity_share_std"]),
                fmt_float(row["optional_consistency_mean"]),
                fmt_float(row["optional_consistency_std"]),
                fmt_float(row["tw_inactive_mass_mean"]),
                fmt_float(row["tw_inactive_mass_std"]),
                fmt_float(row["recourse_rate_mean"]),
                fmt_float(row["recourse_rate_std"]),
                fmt_float(row["chosen_feasible_rate_mean"]),
                fmt_float(row["chosen_feasible_rate_std"]),
            )
        console.print(table)


class DeletionTablePrinter:
    """Prints the deletion faithfulness table (replaces _print_deletion_table)."""

    def __init__(self, layout: str = "auto") -> None:
        self.layout = layout

    def print(self, rows: List[Dict[str, Any]]) -> None:
        console, compact = _table_console(self.layout, compact_threshold=150)
        if not rows:
            return

        aggregated = "runs" in rows[0]

        if compact:
            table = Table(
                title="IG Deletion Faithfulness",
                box=box.SIMPLE_HEAVY,
                header_style="bold yellow",
                expand=True,
                collapse_padding=True,
            )
            table.add_column("model", style="bold", overflow="fold", ratio=3)
            table.add_column("setup", overflow="fold", ratio=2)
            table.add_column("metrics", overflow="fold", ratio=3)

            for row in rows:
                setup_lines = [f"variants: {cell_text(row['variant_mix'])}"]
                if aggregated:
                    setup_lines.append(
                        f"runs={int(row['runs'])} seeds={int(row['seeds'])} range={row['seed_range']}"
                    )
                metrics_block = [
                    _format_deletion_metric_block(row["deletion_metrics"], "flip_rate", "flip"),
                    _format_deletion_metric_block(
                        row["deletion_metrics"], "logprob_drop", "dlogp"
                    ),
                    _format_deletion_metric_block(
                        row["deletion_metrics"], "logit_drop", "dlogit"
                    ),
                ]
                table.add_row(
                    cell_text(row["model"]),
                    "\n".join(setup_lines),
                    "\n".join(block for block in metrics_block if block and block != "-"),
                )
            console.print(table)
            return

        table = Table(
            title="IG Deletion Faithfulness",
            box=box.SIMPLE_HEAVY,
            header_style="bold yellow",
            expand=True,
            collapse_padding=True,
        )
        table.add_column("model", style="bold", overflow="fold", max_width=42)
        table.add_column("variants", overflow="fold", max_width=32)
        if aggregated:
            table.add_column("runs", justify="right")
            table.add_column("seeds", justify="right")
            table.add_column("seed_range", no_wrap=True, max_width=12)
        table.add_column("flip@k", overflow="fold", max_width=20)
        table.add_column("dlogp@k", overflow="fold", max_width=20)
        table.add_column("dlogit@k", overflow="fold", max_width=20)

        for row in rows:
            cells = [str(row["model"]), str(row["variant_mix"])]
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
                    _format_deletion_metric_block(row["deletion_metrics"], "flip_rate", "flip"),
                    _format_deletion_metric_block(row["deletion_metrics"], "logprob_drop", "dlogp"),
                    _format_deletion_metric_block(row["deletion_metrics"], "logit_drop", "dlogit"),
                ]
            )
            table.add_row(*cells)
        console.print(table)


class SummaryTablePrinter:
    """Prints the XAI report summary table (replaces summarize_reports.print_table)."""

    def __init__(self, width: int = 180) -> None:
        self.width = width

    def print(self, rows: List[Dict[str, Any]]) -> None:
        console = Console(width=self.width)
        if not rows:
            console.print("[yellow]No reports matched[/yellow]")
            return

        table = Table(
            title="XAI Report Summary",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("model", style="bold", no_wrap=True, overflow="ellipsis")
        table.add_column("mode", style="magenta", justify="center", no_wrap=True)
        table.add_column("w_feas", justify="right")
        table.add_column("flip@1", justify="right")
        table.add_column("flip@3", justify="right")
        table.add_column("flip@5", justify="right")
        table.add_column("dlogp@1", justify="right")
        table.add_column("dlogp@3", justify="right")
        table.add_column("dlogp@5", justify="right")
        table.add_column("reward", justify="right")
        table.add_column("steps", justify="right")
        table.add_column("done", justify="center")
        table.add_column("feasible", justify="right")
        table.add_column("recourse", justify="right")

        for row in rows:
            done = bool(row.get("done_all", False))
            done_str = "[green]True[/green]" if done else "[red]False[/red]"
            table.add_row(
                str(row.get("model_label", row.get("checkpoint_path", ""))),
                str(row.get("importance_mode", "decision-only")),
                fmt_float(row.get("feasibility_weight"), ndigits=2),
                fmt_float(row.get("flip_at_1")),
                fmt_float(row.get("flip_at_3")),
                fmt_float(row.get("flip_at_5")),
                fmt_float(row.get("dlogp_at_1")),
                fmt_float(row.get("dlogp_at_3")),
                fmt_float(row.get("dlogp_at_5")),
                fmt_float(row.get("mean_final_reward")),
                str(row.get("num_steps", "nan")),
                done_str,
                fmt_float(row.get("chosen_feasible_rate")),
                fmt_float(row.get("recourse_event_rate")),
            )
        console.print(table)


class IGBaselineTablePrinter:
    """Prints the IG baseline comparison table (replaces compare_ig_baselines._print_table)."""

    def print(self, rows: List[Dict[str, Any]], topk_values: List[int]) -> None:
        def _fmt(value: float) -> str:
            import math
            return "n/a" if not math.isfinite(float(value)) else f"{float(value):.4f}"

        console = Console()
        table = Table(title="IG Baseline Comparison")
        table.add_column("baseline")
        table.add_column("metrics")
        for row in rows:
            flips = " ".join(f"flip@{k}={_fmt(row[f'flip@{k}'])}" for k in topk_values)
            metrics = (
                f"f1={_fmt(row['focus@1'])} f3={_fmt(row['focus@3'])}\n"
                f"clr={_fmt(row['clarity'])} ctr={_fmt(row['contrast'])}\n"
                f"{flips}\n"
                f"rec={_fmt(row['recourse'])} feas={_fmt(row['feasible'])}"
            )
            table.add_row(row["baseline"], metrics)
        console.print(table)
