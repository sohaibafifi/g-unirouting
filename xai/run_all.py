from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def _resolve_path(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _mpl_env() -> Dict[str, str]:
    env = dict(os.environ)
    cache_root = Path(tempfile.gettempdir()) / "codex-mpl-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(cache_root))
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    return env


def _append_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.append(f"{flag}={value}")


def _append_list_arg(cmd: List[str], flag: str, values: Sequence[Any] | None) -> None:
    if not values:
        return
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def _slugify(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(raw)).strip("-").lower()
    return text or "item"


def _parse_decoder_method_csv(raw: str) -> List[str]:
    methods: List[str] = []
    for token in str(raw).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"gradient", "grad", "saliency", "gradient_local"}:
            key = "gradient"
        elif token in {"integrated_gradients", "ig"}:
            key = "integrated_gradients"
        elif token in {"deeplift", "deep_lift", "deep-lift", "dl"}:
            key = "deeplift"
        else:
            raise ValueError(
                f"Unsupported decoder attribution method {token!r}. "
                "Use 'gradient', 'integrated_gradients', and/or 'deeplift'."
            )
        if key not in methods:
            methods.append(key)
    return methods or ["gradient", "integrated_gradients"]


def _format_decoder_method_csv(methods: Sequence[str]) -> str:
    preferred = ["gradient", "integrated_gradients", "deeplift"]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(method for method in methods if method not in ordered)
    return ",".join(ordered)


def _effective_decoder_methods(args: argparse.Namespace) -> str:
    methods = _parse_decoder_method_csv(args.decoder_attribution_methods)
    if args.decoder_compare_ig_deeplift:
        for required in ("integrated_gradients", "deeplift"):
            if required not in methods:
                methods.append(required)
    return _format_decoder_method_csv(methods)


def _parse_topk_values(raw: str) -> List[int]:
    values = sorted({int(tok.strip()) for tok in str(raw).strip("[]").split(",") if tok.strip()})
    if not values:
        raise ValueError("--decoder-topk-nodes must contain at least one integer")
    return values


def _report_attr_features(cfg: Dict[str, Any]) -> tuple[str, ...]:
    raw = cfg.get("attr_features", "auto")
    if isinstance(raw, list):
        return tuple(str(v).strip() for v in raw)
    return tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())


def _decoder_compare_specs(
    project_root: Path,
    pattern: str,
    model_filter: str | None,
    max_runs: int | None,
) -> List[Path]:
    checkpoints: List[Path] = []
    wanted = None
    if model_filter:
        wanted = {tok.strip().lower() for tok in model_filter.split(",") if tok.strip()}
    for raw_path in sorted(glob.glob(str(project_root / pattern))):
        ckpt = Path(raw_path).resolve()
        if not ckpt.exists():
            continue
        if wanted:
            run_name = ckpt.parent.name
            graph_size = ckpt.parent.parent.name if len(ckpt.parents) >= 2 else "unknown"
            problem = ckpt.parent.parent.parent.name if len(ckpt.parents) >= 3 else "unknown"
            model_label = f"{problem}/{graph_size}/{run_name}".lower()
            if not any(term in model_label for term in wanted):
                continue
        checkpoints.append(ckpt)
        if max_runs is not None and len(checkpoints) >= int(max_runs):
            break
    return checkpoints


def _run(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    dry_run: bool,
    executed: List[str],
) -> None:
    rendered = shlex.join(cmd)
    executed.append(rendered)
    print(f"$ {rendered}")
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _discover_decoder_sources(output_dir: Path, source_mode: str) -> List[Path]:
    bundles = sorted(output_dir.glob("xai_bundle_*.json"))
    reports = sorted(output_dir.glob("action_explainer_*.json"))
    if source_mode == "bundles":
        return bundles
    if source_mode == "reports":
        return reports
    return bundles if bundles else reports


def _encoder_constraint_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_encoder_probes.py",
        f"--num-samples={args.encoder_num_samples}",
        f"--pooling={args.encoder_pooling}",
        f"--max-k={args.encoder_max_k}",
        f"--seed={args.encoder_seed}",
        f"--sort-by={args.encoder_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_encoder_probes.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        if args.encoder_projection_methods:
            plot_cmd.append("--projection-methods")
            plot_cmd.extend(args.encoder_projection_methods)
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "projection_methods": list(args.encoder_projection_methods),
    }


def _encoder_graph_concept_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_concept_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_concept_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_concept_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_concept_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_concept_probes.py",
        f"--num-samples={args.encoder_concept_num_samples}",
        f"--pooling={args.encoder_concept_pooling}",
        f"--max-k={args.encoder_concept_max_k}",
        f"--seed={args.encoder_concept_seed}",
        f"--sort-by={args.encoder_concept_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_concept_probes.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        if args.encoder_projection_methods:
            plot_cmd.append("--projection-methods")
            plot_cmd.extend(args.encoder_projection_methods)
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "projection_methods": list(args.encoder_projection_methods),
    }


def _encoder_discovered_direction_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_discovered_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_discovered_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_discovered_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_discovered_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_discovered_directions.py",
        f"--num-samples={args.encoder_discovered_num_samples}",
        f"--pooling={args.encoder_discovered_pooling}",
        f"--seed={args.encoder_discovered_seed}",
        f"--num-components={args.encoder_discovered_num_components}",
        f"--top-components={args.encoder_discovered_top_components}",
        f"--sort-by={args.encoder_discovered_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_discovered_directions.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
            f"--top-components={args.encoder_discovered_top_components}",
        ]
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "num_components": int(args.encoder_discovered_num_components),
        "top_components": int(args.encoder_discovered_top_components),
    }


def _encoder_unexplained_dossiers_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    input_json = _resolve_path(project_root, args.encoder_discovered_output_json)
    output_dir = _resolve_path(project_root, args.encoder_unexplained_output_dir)

    cmd = [
        args.python_bin,
        "xai/inspect_unexplained_directions.py",
        f"--input-json={input_json}",
        f"--output-dir={output_dir}",
        f"--top-components={args.encoder_unexplained_top_components}",
        f"--known-threshold={args.encoder_unexplained_known_threshold}",
        f"--top-instances={args.encoder_unexplained_top_instances}",
    ]
    _run(cmd, project_root, env, args.dry_run, executed)

    return {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "top_components": int(args.encoder_unexplained_top_components),
        "known_threshold": float(args.encoder_unexplained_known_threshold),
        "top_instances": int(args.encoder_unexplained_top_instances),
    }


def _encoder_node_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_node_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_node_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_node_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_node_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_node_probes.py",
        f"--num-samples={args.encoder_node_num_samples}",
        f"--decode-mode={args.encoder_node_decode_mode}",
        f"--inference-batch-size={args.encoder_node_inference_batch_size}",
        f"--max-probe-nodes={args.encoder_node_max_probe_nodes}",
        f"--max-k={args.encoder_node_max_k}",
        f"--seed={args.encoder_node_seed}",
        f"--sort-by={args.encoder_node_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_node_probes.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        if args.encoder_projection_methods:
            plot_cmd.append("--projection-methods")
            plot_cmd.extend(args.encoder_projection_methods)
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "projection_methods": list(args.encoder_projection_methods),
    }


def _encoder_edge_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_edge_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_edge_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_edge_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_edge_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_edge_probes.py",
        f"--num-samples={args.encoder_edge_num_samples}",
        f"--decode-mode={args.encoder_edge_decode_mode}",
        f"--inference-batch-size={args.encoder_edge_inference_batch_size}",
        f"--max-probe-edges={args.encoder_edge_max_probe_edges}",
        f"--max-k={args.encoder_edge_max_k}",
        f"--seed={args.encoder_edge_seed}",
        f"--sort-by={args.encoder_edge_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_edge_probes.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        if args.encoder_projection_methods:
            plot_cmd.append("--projection-methods")
            plot_cmd.extend(args.encoder_projection_methods)
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "projection_methods": list(args.encoder_projection_methods),
    }


def _encoder_discovered_stability_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_discovered_stability_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_discovered_stability_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_discovered_stability_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_discovered_stability_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_discovered_direction_stability.py",
        f"--num-samples={args.encoder_discovered_stability_num_samples}",
        f"--pooling={args.encoder_discovered_stability_pooling}",
        f"--data-seed={args.encoder_discovered_stability_data_seed}",
        f"--seed-start={args.encoder_discovered_stability_seed_start}",
        f"--num-runs={args.encoder_discovered_stability_num_runs}",
        f"--num-components={args.encoder_discovered_stability_num_components}",
        f"--top-components={args.encoder_discovered_stability_top_components}",
        f"--sort-by={args.encoder_discovered_stability_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_discovered_direction_stability.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "num_runs": int(args.encoder_discovered_stability_num_runs),
    }


def _encoder_intervention_validation_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    comparison_json = _resolve_path(project_root, args.encoder_intervention_output_json)
    comparison_md = _resolve_path(project_root, args.encoder_intervention_output_md)
    per_config_dir = _resolve_path(project_root, args.encoder_intervention_per_config_dir)
    plot_dir = _resolve_path(project_root, args.encoder_intervention_plot_dir)

    compare_cmd = [
        args.python_bin,
        "xai/compare_intervention_validation.py",
        f"--num-samples={args.encoder_intervention_num_samples}",
        f"--pooling={args.encoder_intervention_pooling}",
        f"--seed={args.encoder_intervention_seed}",
        f"--num-components={args.encoder_intervention_num_components}",
        f"--top-components={args.encoder_intervention_top_components}",
        f"--sort-by={args.encoder_intervention_sort_by}",
        f"--output-json={comparison_json}",
        f"--output-md={comparison_md}",
        f"--per-config-dir={per_config_dir}",
    ]
    _append_list_arg(compare_cmd, "--config-ids", args.encoder_config_ids)
    _append_arg(compare_cmd, "--device", args.encoder_device)
    _append_arg(compare_cmd, "--graph-size", args.encoder_graph_size)
    _append_arg(compare_cmd, "--problem", args.encoder_problem)
    _run(compare_cmd, project_root, env, args.dry_run, executed)

    if not args.skip_encoder_plots:
        plot_cmd = [
            args.python_bin,
            "xai/plot_intervention_validation.py",
            f"--input-json={comparison_json}",
            f"--output-dir={plot_dir}",
            f"--dpi={args.encoder_plot_dpi}",
        ]
        _run(plot_cmd, project_root, env, args.dry_run, executed)

    return {
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        "per_config_dir": str(per_config_dir),
        "plot_dir": str(plot_dir),
        "num_components": int(args.encoder_intervention_num_components),
        "top_components": int(args.encoder_intervention_top_components),
    }


def _encoder_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    if not args.skip_encoder_constraints:
        manifest["constraints"] = _encoder_constraint_flow(args, project_root, env, executed)
    if not args.skip_encoder_graph_concepts:
        manifest["graph_concepts"] = _encoder_graph_concept_flow(args, project_root, env, executed)
    if not args.skip_encoder_discovered_directions:
        manifest["discovered_directions"] = _encoder_discovered_direction_flow(
            args, project_root, env, executed
        )
    if args.enable_encoder_unexplained_dossiers:
        manifest["unexplained_dossiers"] = _encoder_unexplained_dossiers_flow(
            args, project_root, env, executed
        )
    if args.enable_encoder_discovered_stability:
        manifest["discovered_direction_stability"] = _encoder_discovered_stability_flow(
            args, project_root, env, executed
        )
    if args.enable_encoder_intervention_validation:
        manifest["intervention_validation"] = _encoder_intervention_validation_flow(
            args, project_root, env, executed
        )
    if not args.skip_encoder_node_probes:
        manifest["node_probes"] = _encoder_node_flow(args, project_root, env, executed)
    if not args.skip_encoder_edge_probes:
        manifest["edge_probes"] = _encoder_edge_flow(args, project_root, env, executed)
    return manifest


def _decoder_batch_flow(
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Path:
    output_dir = _resolve_path(project_root, args.decoder_output_dir)
    effective_methods = _effective_decoder_methods(args)
    cmd = [
        args.python_bin,
        "xai/run_all_action_explainer.py",
        f"--project-root={project_root}",
        f"--output-dir={output_dir}",
        f"--checkpoints-glob={args.decoder_checkpoints_glob}",
        f"--num-instances={args.decoder_num_instances}",
        f"--max-steps={args.decoder_max_steps}",
        f"--attribution-methods={effective_methods}",
        f"--repeats={args.decoder_repeats}",
        f"--repeat-mode={args.decoder_repeat_mode}",
        f"--topk-nodes={args.decoder_topk_nodes}",
        f"--feasibility-top-m={args.decoder_feasibility_top_m}",
        f"--feasibility-cost-weight={args.decoder_feasibility_cost_weight}",
        f"--attr-features={args.decoder_attr_features}",
        f"--ig-steps={args.decoder_ig_steps}",
        f"--ig-baseline={args.decoder_ig_baseline}",
        f"--max-instances-to-store={args.decoder_max_instances_to_store}",
    ]
    _append_arg(cmd, "--seed", args.decoder_seed)
    _append_arg(cmd, "--device", args.decoder_device)
    _append_arg(cmd, "--feasibility-weight", args.decoder_feasibility_weight)
    _append_arg(cmd, "--model-filter", args.decoder_model_filter)
    _append_arg(cmd, "--max-runs", args.decoder_max_runs)
    if args.decoder_skip_existing:
        cmd.append("--skip-existing")
    if args.decoder_stop_on_error:
        cmd.append("--stop-on-error")
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, project_root, env, args.dry_run, executed)
    return output_dir


def _decoder_compare_ig_deeplift_flow(
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    from evaluation.metrics import evaluate_single_report, report_method
    from evaluation.table_printers import MethodComparisonTablePrinter

    compare_root = (
        _resolve_path(project_root, args.decoder_compare_output_dir)
        if args.decoder_compare_output_dir is not None
        else (output_dir.parent / "comparisons" / "ig_vs_deeplift").resolve()
    )
    if args.dry_run:
        print(f"Dry-run: would generate IG/DeepLIFT comparisons under {compare_root}")
        return {
            "output_dir": str(compare_root),
            "index_json": str(compare_root / "ig_vs_deeplift_index.json"),
            "comparisons": [],
            "dry_run": True,
        }
    compare_root.mkdir(parents=True, exist_ok=True)

    target_checkpoints = {
        str(path)
        for path in _decoder_compare_specs(
            project_root=project_root,
            pattern=args.decoder_checkpoints_glob,
            model_filter=args.decoder_model_filter,
            max_runs=args.decoder_max_runs,
        )
    }
    wanted_topk = tuple(_parse_topk_values(args.decoder_topk_nodes))
    wanted_baseline = str(args.decoder_ig_baseline).strip()
    report_files = sorted(
        output_dir.glob("action_explainer_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    grouped: Dict[tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for report_path in report_files:
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        cfg = data.get("config", {}) or {}
        ckpt = str(cfg.get("checkpoint_path_resolved") or cfg.get("checkpoint_path") or "").strip()
        if target_checkpoints and ckpt not in target_checkpoints:
            continue
        try:
            report_topk = tuple(int(v) for v in cfg.get("topk_nodes", []))
        except Exception:
            continue
        if report_topk != wanted_topk:
            continue
        if int(cfg.get("num_instances", -1)) != int(args.decoder_num_instances):
            continue
        if int(cfg.get("max_steps", -1)) != int(args.decoder_max_steps):
            continue
        if _report_attr_features(cfg) != _report_attr_features({"attr_features": args.decoder_attr_features}):
            if str(args.decoder_attr_features).lower() != "auto":
                continue
        try:
            if int(cfg.get("feasibility_top_m", -1)) != int(args.decoder_feasibility_top_m):
                continue
            if float(cfg.get("feasibility_cost_weight", -1.0)) != float(
                args.decoder_feasibility_cost_weight
            ):
                continue
        except Exception:
            continue
        wanted_weight = args.decoder_feasibility_weight
        if wanted_weight is not None:
            try:
                if float(cfg.get("feasibility_weight", float("nan"))) != float(wanted_weight):
                    continue
            except Exception:
                continue

        baseline = str(
            cfg.get("reference_baseline") or cfg.get("ig_baseline") or cfg.get("deeplift_baseline") or ""
        ).strip()
        if baseline != wanted_baseline:
            continue

        method = report_method({"file": str(report_path), "data": data})
        if method not in {"integrated_gradients", "deeplift"}:
            continue
        seed = cfg.get("seed", None)
        data_seed = cfg.get("data_seed", None)
        key = (
            ckpt,
            seed,
            data_seed,
            baseline,
            int(cfg.get("num_instances", 0)),
            int(cfg.get("max_steps", 0)),
            report_topk,
        )
        bucket = grouped.setdefault(key, {})
        if method not in bucket:
            bucket[method] = {"file": str(report_path), "data": data}

    comparisons: List[Dict[str, Any]] = []
    printer = MethodComparisonTablePrinter()

    for key in sorted(grouped.keys(), key=lambda item: (item[0], str(item[1]), str(item[2]))):
        bucket = grouped[key]
        if "integrated_gradients" not in bucket or "deeplift" not in bucket:
            continue

        rows: List[Dict[str, Any]] = []
        report_paths: List[str] = []
        model_label = ""
        for method in ("integrated_gradients", "deeplift"):
            report = bucket[method]
            raw = report["data"]
            cfg = raw.get("config", {}) or {}
            summary = raw.get("summary", {}) or {}
            deletion = summary.get("deletion_faithfulness", {}) or {}
            metrics = evaluate_single_report(report)
            model_label = model_label or str(
                cfg.get("model_label_base") or cfg.get("model_label") or Path(key[0]).name
            )
            row: Dict[str, Any] = {
                "method": str(cfg.get("attribution_method", method)),
                "baseline": str(
                    cfg.get("reference_baseline")
                    or cfg.get("ig_baseline")
                    or cfg.get("deeplift_baseline")
                    or ""
                ),
                "model": str(cfg.get("model_label", "")),
                "path": str(report["file"]),
                "steps": int(summary.get("num_steps", 0)),
                "focus@1": float(metrics["focus_top1"]),
                "focus@3": float(metrics["focus_top3"]),
                "clarity": float(metrics["clarity"]),
                "contrast": float(metrics["contrast_gap"]),
                "recourse": float(metrics["recourse_rate"]),
                "feasible": float(metrics["chosen_feasible_rate"]),
            }
            for topk in wanted_topk:
                row[f"flip@{topk}"] = float(
                    (deletion.get(str(topk)) or {}).get("mean_action_flip_rate", 0.0)
                )
            rows.append(row)
            report_paths.append(str(report["file"]))

        title_seed = f", seed={key[1]}" if key[1] not in (None, "") else ""
        printer.print(rows, list(wanted_topk), title=f"IG vs DeepLIFT Comparison ({model_label}{title_seed})")

        summary_name = _slugify(f"{model_label}-{key[1]}-{key[2]}-{key[3]}")
        summary_path = compare_root / f"ig_vs_deeplift_{summary_name}.json"
        payload = {
            "timestamp": int(time.time()),
            "model_label": model_label,
            "checkpoint_path": key[0],
            "seed": key[1],
            "data_seed": key[2],
            "baseline": key[3],
            "reports": report_paths,
            "rows": rows,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        comparisons.append(
            {
                "checkpoint_path": key[0],
                "seed": key[1],
                "data_seed": key[2],
                "baseline": key[3],
                "summary_json": str(summary_path),
                "reports": report_paths,
            }
        )

    index_path = compare_root / "ig_vs_deeplift_index.json"
    index_path.write_text(
        json.dumps(
            {
                "timestamp": int(time.time()),
                "baseline": wanted_baseline,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not comparisons:
        print(f"No IG/DeepLIFT comparison pairs found in {output_dir}")

    return {
        "output_dir": str(compare_root),
        "index_json": str(index_path),
        "comparisons": comparisons,
    }


def _decoder_postprocess_flow(
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
    env: Dict[str, str],
    executed: List[str],
) -> Dict[str, Any]:
    sources = _discover_decoder_sources(output_dir, args.decoder_source_mode)
    text_root = (
        _resolve_path(project_root, args.decoder_text_dir)
        if args.decoder_text_dir is not None
        else (output_dir.parent / "text").resolve()
    )
    plot_root = (
        _resolve_path(project_root, args.decoder_plot_dir)
        if args.decoder_plot_dir is not None
        else (output_dir.parent / "plots").resolve()
    )

    processed: List[Dict[str, str]] = []
    if not sources:
        print(f"No decoder bundle/report found in {output_dir}")
        return {
            "output_dir": str(output_dir),
            "source_mode": args.decoder_source_mode,
            "sources": [],
            "text_root": str(text_root),
            "plot_root": str(plot_root),
        }

    for source in sources:
        source_id = source.stem
        item: Dict[str, str] = {"source": str(source)}

        if not args.skip_decoder_text:
            text_cmd = [
                args.python_bin,
                "xai/text_explanations.py",
                f"--report={source}",
                f"--instance={args.decoder_text_instance}",
            ]
            if args.decoder_text_steps is not None:
                text_cmd.append(f"--steps={args.decoder_text_steps}")
            if args.decoder_text_instance.strip().lower() == "all":
                text_out = text_root / source_id
            else:
                text_out = text_root / f"{source_id}.md"
            text_cmd.append(f"--output={text_out}")
            _run(text_cmd, project_root, env, args.dry_run, executed)
            item["text_output"] = str(text_out)

        if not args.skip_decoder_plots:
            plot_out_dir = plot_root / source_id
            plot_cmd = [
                args.python_bin,
                "xai/plot_explanations.py",
                f"--report={source}",
                f"--output-dir={plot_out_dir}",
                f"--num-instances={args.decoder_plot_num_instances}",
            ]
            if args.decoder_plot_instances is not None:
                plot_cmd.append(f"--instances={args.decoder_plot_instances}")
            if args.decoder_plot_steps is not None:
                plot_cmd.append(f"--steps={args.decoder_plot_steps}")
            _run(plot_cmd, project_root, env, args.dry_run, executed)
            item["plot_output_dir"] = str(plot_out_dir)

        processed.append(item)

    return {
        "output_dir": str(output_dir),
        "source_mode": args.decoder_source_mode,
        "sources": processed,
        "text_root": str(text_root),
        "plot_root": str(plot_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full XAI flow: encoder comparison+plots and decoder explanations+text+plots."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--manifest", default="logs/xai/manifests/run_all_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-encoder-plots", action="store_true")
    parser.add_argument("--skip-encoder-constraints", action="store_true")
    parser.add_argument("--skip-encoder-graph-concepts", action="store_true")
    parser.add_argument("--skip-encoder-discovered-directions", action="store_true")
    parser.add_argument("--enable-encoder-unexplained-dossiers", action="store_true")
    parser.add_argument("--enable-encoder-discovered-stability", action="store_true")
    parser.add_argument("--enable-encoder-intervention-validation", action="store_true")
    parser.add_argument("--skip-encoder-node-probes", action="store_true")
    parser.add_argument("--skip-encoder-edge-probes", action="store_true")
    parser.add_argument("--skip-decoder-run", action="store_true")
    parser.add_argument("--skip-decoder-text", action="store_true")
    parser.add_argument("--skip-decoder-plots", action="store_true")
    parser.add_argument("--decoder-compare-ig-deeplift", action="store_true")

    parser.add_argument("--encoder-config-ids", type=int, nargs="*", default=None)
    parser.add_argument("--encoder-device", default=None)
    parser.add_argument("--encoder-graph-size", type=int, default=None)
    parser.add_argument("--encoder-problem", default=None)
    parser.add_argument("--encoder-num-samples", type=int, default=1024)
    parser.add_argument(
        "--encoder-pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--encoder-max-k", type=int, default=12)
    parser.add_argument("--encoder-seed", type=int, default=1234)
    parser.add_argument(
        "--encoder-sort-by",
        choices=[
            "signature_nmi",
            "signature_ari",
            "signature_macro_f1",
            "route_structure_auc",
            "time_windows_auc",
            "distance_limit_auc",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="signature_nmi",
    )
    parser.add_argument(
        "--encoder-output-json",
        default="logs/xai/encoder/graph/constraints/comparison.json",
    )
    parser.add_argument(
        "--encoder-output-md",
        default="logs/xai/encoder/graph/constraints/comparison.md",
    )
    parser.add_argument(
        "--encoder-per-config-dir",
        default="logs/xai/encoder/graph/constraints/runs",
    )
    parser.add_argument(
        "--encoder-plot-dir",
        default="logs/xai/encoder/graph/constraints/plots",
    )
    parser.add_argument(
        "--encoder-projection-methods",
        nargs="*",
        choices=["pca", "tsne"],
        default=["pca"],
    )
    parser.add_argument("--encoder-plot-dpi", type=int, default=180)

    parser.add_argument("--encoder-concept-num-samples", type=int, default=1024)
    parser.add_argument(
        "--encoder-concept-pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--encoder-concept-max-k", type=int, default=12)
    parser.add_argument("--encoder-concept-seed", type=int, default=1234)
    parser.add_argument(
        "--encoder-concept-sort-by",
        choices=[
            "concept_signature_nmi",
            "concept_signature_ari",
            "concept_signature_macro_f1",
            "concept_macro_f1_mean",
            "compactness_f1",
            "clustering_f1",
            "outlier_f1",
            "outlier_auc",
            "load_concentration_f1",
            "lhbh_balance_f1",
            "capacity_prior_f1",
            "tw_density_f1",
            "tw_width_f1",
            "distance_budget_f1",
            "combined_tension_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="concept_signature_nmi",
    )
    parser.add_argument(
        "--encoder-concept-output-json",
        default="logs/xai/encoder/graph/concepts/comparison.json",
    )
    parser.add_argument(
        "--encoder-concept-output-md",
        default="logs/xai/encoder/graph/concepts/comparison.md",
    )
    parser.add_argument(
        "--encoder-concept-per-config-dir",
        default="logs/xai/encoder/graph/concepts/runs",
    )
    parser.add_argument(
        "--encoder-concept-plot-dir",
        default="logs/xai/encoder/graph/concepts/plots",
    )

    parser.add_argument("--encoder-discovered-num-samples", type=int, default=1024)
    parser.add_argument(
        "--encoder-discovered-pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--encoder-discovered-seed", type=int, default=1234)
    parser.add_argument("--encoder-discovered-num-components", type=int, default=8)
    parser.add_argument("--encoder-discovered-top-components", type=int, default=5)
    parser.add_argument(
        "--encoder-discovered-sort-by",
        choices=[
            "mean_best_abs_correlation_top_components",
            "best_abs_correlation_overall",
            "num_components_abs_correlation_ge_0_5",
            "top3_cumulative_explained_variance_ratio",
            "effective_rank_mean",
            "distance_budget_best_abs_correlation",
            "combined_tension_best_abs_correlation",
        ],
        default="mean_best_abs_correlation_top_components",
    )
    parser.add_argument(
        "--encoder-discovered-output-json",
        default="logs/xai/encoder/graph/discovered_directions/comparison.json",
    )
    parser.add_argument(
        "--encoder-discovered-output-md",
        default="logs/xai/encoder/graph/discovered_directions/comparison.md",
    )
    parser.add_argument(
        "--encoder-discovered-per-config-dir",
        default="logs/xai/encoder/graph/discovered_directions/runs",
    )
    parser.add_argument(
        "--encoder-discovered-plot-dir",
        default="logs/xai/encoder/graph/discovered_directions/plots",
    )
    parser.add_argument(
        "--encoder-unexplained-output-dir",
        default="logs/xai/encoder/graph/discovered_directions/unexplained_dossiers",
    )
    parser.add_argument("--encoder-unexplained-top-components", type=int, default=5)
    parser.add_argument("--encoder-unexplained-known-threshold", type=float, default=0.3)
    parser.add_argument("--encoder-unexplained-top-instances", type=int, default=5)
    parser.add_argument("--encoder-discovered-stability-num-samples", type=int, default=1024)
    parser.add_argument(
        "--encoder-discovered-stability-pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--encoder-discovered-stability-data-seed", type=int, default=1234)
    parser.add_argument("--encoder-discovered-stability-seed-start", type=int, default=1234)
    parser.add_argument("--encoder-discovered-stability-num-runs", type=int, default=4)
    parser.add_argument("--encoder-discovered-stability-num-components", type=int, default=8)
    parser.add_argument("--encoder-discovered-stability-top-components", type=int, default=5)
    parser.add_argument(
        "--encoder-discovered-stability-sort-by",
        choices=[
            "pca_component_alignment_mean",
            "ica_component_alignment_mean",
            "pca_same_top_concept_ratio_mean",
            "ica_same_top_concept_ratio_mean",
            "pca_mean_abs_corr_mean",
            "ica_mean_abs_corr_mean",
        ],
        default="ica_component_alignment_mean",
    )
    parser.add_argument(
        "--encoder-discovered-stability-output-json",
        default="logs/xai/encoder/graph/discovered_directions/stability/comparison.json",
    )
    parser.add_argument(
        "--encoder-discovered-stability-output-md",
        default="logs/xai/encoder/graph/discovered_directions/stability/comparison.md",
    )
    parser.add_argument(
        "--encoder-discovered-stability-per-config-dir",
        default="logs/xai/encoder/graph/discovered_directions/stability/runs",
    )
    parser.add_argument(
        "--encoder-discovered-stability-plot-dir",
        default="logs/xai/encoder/graph/discovered_directions/stability/plots",
    )

    parser.add_argument("--encoder-intervention-num-samples", type=int, default=512)
    parser.add_argument(
        "--encoder-intervention-pooling",
        choices=["mean", "meanstd", "depot_meanstd"],
        default="meanstd",
    )
    parser.add_argument("--encoder-intervention-seed", type=int, default=1234)
    parser.add_argument("--encoder-intervention-num-components", type=int, default=8)
    parser.add_argument("--encoder-intervention-top-components", type=int, default=5)
    parser.add_argument(
        "--encoder-intervention-sort-by",
        choices=[
            "pca_mean_directional_success",
            "ica_mean_directional_success",
            "pca_mean_aligned_delta_correlation",
            "ica_mean_aligned_delta_correlation",
            "concept_success_mean",
        ],
        default="ica_mean_directional_success",
    )
    parser.add_argument(
        "--encoder-intervention-output-json",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/comparison.json",
    )
    parser.add_argument(
        "--encoder-intervention-output-md",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/comparison.md",
    )
    parser.add_argument(
        "--encoder-intervention-per-config-dir",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/runs",
    )
    parser.add_argument(
        "--encoder-intervention-plot-dir",
        default="logs/xai/encoder/graph/discovered_directions/intervention_validation/plots",
    )

    parser.add_argument("--encoder-node-num-samples", type=int, default=256)
    parser.add_argument("--encoder-node-decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--encoder-node-inference-batch-size", type=int, default=64)
    parser.add_argument("--encoder-node-max-probe-nodes", type=int, default=20000)
    parser.add_argument("--encoder-node-max-k", type=int, default=12)
    parser.add_argument("--encoder-node-seed", type=int, default=1234)
    parser.add_argument(
        "--encoder-node-sort-by",
        choices=[
            "node_signature_nmi",
            "node_signature_ari",
            "node_signature_macro_f1",
            "node_macro_f1_mean",
            "service_order_f1",
            "route_role_f1",
            "local_cost_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="node_signature_nmi",
    )
    parser.add_argument(
        "--encoder-node-output-json",
        default="logs/xai/encoder/node/probes/comparison.json",
    )
    parser.add_argument(
        "--encoder-node-output-md",
        default="logs/xai/encoder/node/probes/comparison.md",
    )
    parser.add_argument(
        "--encoder-node-per-config-dir",
        default="logs/xai/encoder/node/probes/runs",
    )
    parser.add_argument(
        "--encoder-node-plot-dir",
        default="logs/xai/encoder/node/probes/plots",
    )

    parser.add_argument("--encoder-edge-num-samples", type=int, default=128)
    parser.add_argument("--encoder-edge-decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--encoder-edge-inference-batch-size", type=int, default=32)
    parser.add_argument("--encoder-edge-max-probe-edges", type=int, default=20000)
    parser.add_argument("--encoder-edge-max-k", type=int, default=12)
    parser.add_argument("--encoder-edge-seed", type=int, default=1234)
    parser.add_argument(
        "--encoder-edge-sort-by",
        choices=[
            "edge_signature_nmi",
            "edge_signature_ari",
            "edge_signature_macro_f1",
            "edge_macro_f1_mean",
            "same_route_f1",
            "same_route_auc",
            "solution_edge_f1",
            "solution_edge_auc",
            "local_edge_cost_f1",
            "effective_rank_mean",
            "best_silhouette",
        ],
        default="edge_signature_nmi",
    )
    parser.add_argument(
        "--encoder-edge-output-json",
        default="logs/xai/encoder/edge/probes/comparison.json",
    )
    parser.add_argument(
        "--encoder-edge-output-md",
        default="logs/xai/encoder/edge/probes/comparison.md",
    )
    parser.add_argument(
        "--encoder-edge-per-config-dir",
        default="logs/xai/encoder/edge/probes/runs",
    )
    parser.add_argument(
        "--encoder-edge-plot-dir",
        default="logs/xai/encoder/edge/probes/plots",
    )

    parser.add_argument("--decoder-output-dir", default="logs/xai/decoder/reports")
    parser.add_argument("--decoder-checkpoints-glob", default="output/*/*/*/baseline.pt")
    parser.add_argument("--decoder-num-instances", type=int, default=128)
    parser.add_argument("--decoder-max-steps", type=int, default=300)
    parser.add_argument("--decoder-attribution-methods", default="gradient,integrated_gradients")
    parser.add_argument("--decoder-seed", type=int, default=1234)
    parser.add_argument("--decoder-repeats", type=int, default=1)
    parser.add_argument(
        "--decoder-repeat-mode",
        choices=["reproducibility", "robustness"],
        default="robustness",
    )
    parser.add_argument("--decoder-topk-nodes", default="[1,3,5]")
    parser.add_argument("--decoder-feasibility-weight", type=float, default=None)
    parser.add_argument("--decoder-feasibility-top-m", type=int, default=8)
    parser.add_argument("--decoder-feasibility-cost-weight", type=float, default=0.25)
    parser.add_argument("--decoder-attr-features", default="auto")
    parser.add_argument("--decoder-ig-steps", type=int, default=50)
    parser.add_argument(
        "--decoder-ig-baseline",
        choices=["mean-fill", "zero-with-customers-at-depot", "zero-all", "zero-with-current-locs"],
        default="mean-fill",
    )
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--decoder-max-instances-to-store", type=int, default=8)
    parser.add_argument("--decoder-model-filter", default=None)
    parser.add_argument("--decoder-max-runs", type=int, default=None)
    parser.add_argument("--decoder-skip-existing", action="store_true")
    parser.add_argument("--decoder-stop-on-error", action="store_true")
    parser.add_argument(
        "--decoder-source-mode",
        choices=["auto", "bundles", "reports"],
        default="auto",
    )
    parser.add_argument("--decoder-text-dir", default=None)
    parser.add_argument("--decoder-text-instance", default="all")
    parser.add_argument("--decoder-text-steps", default=None)
    parser.add_argument("--decoder-plot-dir", default=None)
    parser.add_argument("--decoder-plot-instances", default=None)
    parser.add_argument("--decoder-plot-num-instances", type=int, default=4)
    parser.add_argument("--decoder-plot-steps", default=None)
    parser.add_argument("--decoder-compare-output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    env = _mpl_env()
    executed: List[str] = []
    manifest_path = _resolve_path(project_root, args.manifest)

    manifest: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "project_root": str(project_root),
        "python_bin": args.python_bin,
        "dry_run": bool(args.dry_run),
        "commands": executed,
    }

    if not args.skip_encoder:
        manifest["encoder"] = _encoder_flow(args, project_root, env, executed)

    decoder_output_dir = _resolve_path(project_root, args.decoder_output_dir)
    if not args.skip_decoder_run:
        decoder_output_dir = _decoder_batch_flow(args, project_root, env, executed)

    if args.decoder_compare_ig_deeplift:
        manifest["decoder_compare_ig_deeplift"] = _decoder_compare_ig_deeplift_flow(
            args=args,
            project_root=project_root,
            output_dir=decoder_output_dir,
        )

    if not args.skip_decoder_text or not args.skip_decoder_plots:
        manifest["decoder"] = _decoder_postprocess_flow(
            args=args,
            project_root=project_root,
            output_dir=decoder_output_dir,
            env=env,
            executed=executed,
        )

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
