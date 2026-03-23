"""Batch runner: run action_explainer.py for all discovered checkpoints."""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


@dataclass
class RunSpec:
    checkpoint_path: Path
    model_label: str


def _parse_attribution_methods(raw: str) -> List[str]:
    values = []
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
                f"Unsupported attribution method {token!r}. "
                "Use 'gradient', 'integrated_gradients', and/or 'deeplift'."
            )
        if key not in values:
            values.append(key)
    return values or ["gradient", "integrated_gradients"]


def _discover_specs(project_root: Path, pattern: str) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for raw_path in sorted(glob.glob(str(project_root / pattern))):
        ckpt = Path(raw_path).resolve()
        if not ckpt.exists():
            continue
        run_name = ckpt.parent.name
        graph_size = ckpt.parent.parent.name if len(ckpt.parents) >= 2 else "unknown"
        problem = ckpt.parent.parent.parent.name if len(ckpt.parents) >= 3 else "unknown"
        specs.append(
            RunSpec(
                checkpoint_path=ckpt,
                model_label=f"{problem}/{graph_size}/{run_name}",
            )
        )
    return specs


def _existing_reports(output_dir: Path) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("action_explainer_*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        reports.append({"path": path, "config": data.get("config", {}) or {}})
    return reports


def _seed_for_repeat(args: argparse.Namespace, repeat_idx: int) -> Optional[int]:
    if args.seed is None:
        return None
    if args.repeat_mode == "robustness":
        return int(args.seed) + int(repeat_idx)
    return int(args.seed)


def _report_method(report_cfg: Dict[str, Any]) -> str:
    method = str(report_cfg.get("attribution_method", "")).strip().lower()
    if method in {"gradient", "integrated_gradients", "deeplift"}:
        return method
    label = str(report_cfg.get("model_label", "")).strip().lower()
    if "[ig:" in label:
        return "integrated_gradients"
    if "[deeplift:" in label or "[deep-lift:" in label:
        return "deeplift"
    return "gradient"


def _report_reference_baseline(report_cfg: Dict[str, Any]) -> str:
    return str(
        report_cfg.get("reference_baseline")
        or report_cfg.get("ig_baseline")
        or report_cfg.get("deeplift_baseline")
        or ""
    ).strip()


def _report_matches_spec(
    report_cfg: Dict[str, Any], spec: RunSpec, args: argparse.Namespace
) -> bool:
    report_ckpt = str(
        report_cfg.get("checkpoint_path_resolved") or report_cfg.get("checkpoint_path") or ""
    ).strip()
    if report_ckpt != str(spec.checkpoint_path):
        return False
    if int(report_cfg.get("num_instances", -1)) != int(args.num_instances):
        return False
    if int(report_cfg.get("max_steps", -1)) != int(args.max_steps):
        return False
    try:
        report_topk = tuple(int(v) for v in report_cfg.get("topk_nodes", []))
        wanted_topk = tuple(
            sorted(
                {int(v.strip()) for v in str(args.topk_nodes).strip("[]").split(",") if v.strip()}
            )
        )
    except Exception:
        return False
    if report_topk != wanted_topk:
        return False
    report_weight = report_cfg.get("feasibility_weight", None)
    wanted_weight = args.feasibility_weight
    if wanted_weight is not None:
        try:
            if float(report_weight) != float(wanted_weight):
                return False
        except Exception:
            return False
    try:
        if int(report_cfg.get("feasibility_top_m", -1)) != int(args.feasibility_top_m):
            return False
        if float(report_cfg.get("feasibility_cost_weight", -1.0)) != float(
            args.feasibility_cost_weight
        ):
            return False
    except Exception:
        return False
    report_attr = report_cfg.get("attr_features", "auto")
    if str(args.attr_features).lower() != "auto":
        wanted_attr = tuple(
            tok.strip() for tok in str(args.attr_features).split(",") if tok.strip()
        )
        if isinstance(report_attr, list):
            report_attr_norm = tuple(str(v) for v in report_attr)
        else:
            report_attr_norm = tuple(
                tok.strip() for tok in str(report_attr).split(",") if tok.strip()
            )
        if report_attr_norm != wanted_attr:
            return False
    return True


def _matching_seed_counter(
    project_root: Path,
    spec: RunSpec,
    args: argparse.Namespace,
    requested_methods: List[str],
    cached_reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Optional[int], Counter]:
    reports = cached_reports if cached_reports is not None else _existing_reports(project_root)
    counter: Dict[Optional[int], Counter] = {}
    for report in reports:
        cfg = report.get("config", {})
        if not _report_matches_spec(cfg, spec, args):
            continue
        method = _report_method(cfg)
        if method not in requested_methods:
            continue
        if method == "integrated_gradients":
            try:
                if int(cfg.get("ig_steps", -1)) != int(args.ig_steps):
                    continue
            except Exception:
                continue
            if _report_reference_baseline(cfg) != str(args.ig_baseline).strip():
                continue
        elif method == "deeplift":
            if _report_reference_baseline(cfg) != str(args.ig_baseline).strip():
                continue
        raw_seed = cfg.get("seed", None)
        seed_value = None if raw_seed in (None, "") else int(raw_seed)
        counter.setdefault(seed_value, Counter())
        counter[seed_value][method] += 1
    return counter


def run_batch(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.repeats > 1 and args.seed is None:
        raise ValueError("--repeats > 1 requires --seed")

    specs = _discover_specs(project_root, args.checkpoints_glob)
    requested_methods = _parse_attribution_methods(args.attribution_methods)
    if args.model_filter:
        wanted = {tok.strip().lower() for tok in args.model_filter.split(",") if tok.strip()}
        specs = [
            spec for spec in specs if any(term in spec.model_label.lower() for term in wanted)
        ]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    if not specs:
        print("No runnable checkpoints found.")
        return 0

    existing_reports = _existing_reports(output_dir) if args.skip_existing else []
    queued: List[tuple] = []
    for spec in specs:
        seed_counter = (
            _matching_seed_counter(
                project_root, spec, args, requested_methods, cached_reports=existing_reports
            )
            if args.skip_existing
            else {}
        )
        for repeat_idx in range(int(args.repeats)):
            run_seed = _seed_for_repeat(args, repeat_idx)
            if args.skip_existing:
                method_counter = seed_counter.get(run_seed, Counter())
                has_full_run = all(
                    method_counter.get(method, 0) > 0 for method in requested_methods
                )
                if args.repeat_mode == "robustness":
                    if has_full_run:
                        continue
                else:
                    if has_full_run:
                        for method in requested_methods:
                            method_counter[method] -= 1
                        continue
            queued.append((spec, repeat_idx + 1, run_seed))

    if not queued:
        print("No runnable checkpoints found.")
        return 0

    print(f"Discovered {len(specs)} checkpoint(s), queued {len(queued)} execution(s):")
    for i, (spec, repeat_no, run_seed) in enumerate(queued, start=1):
        seed_label = f" | seed={run_seed}" if run_seed is not None else ""
        print(
            f"{i}. model={spec.model_label} | repeat={repeat_no}/{args.repeats}"
            f" | mode={args.repeat_mode}{seed_label} | ckpt={spec.checkpoint_path}"
        )
    print("")

    failures = []
    successes = []
    dry_runs = 0

    for i, (spec, repeat_no, run_seed) in enumerate(queued, start=1):
        cmd = [
            args.python_bin,
            "xai/action_explainer.py",
            f"--checkpoint={spec.checkpoint_path}",
            f"--num-instances={args.num_instances}",
            f"--max-steps={args.max_steps}",
            f"--topk-nodes={args.topk_nodes}",
            f"--attribution-methods={args.attribution_methods}",
            f"--feasibility-top-m={args.feasibility_top_m}",
            f"--feasibility-cost-weight={args.feasibility_cost_weight}",
            f"--attr-features={args.attr_features}",
            f"--ig-steps={args.ig_steps}",
            f"--ig-baseline={args.ig_baseline}",
            f"--max-instances-to-store={args.max_instances_to_store}",
            f"--output-dir={output_dir}",
        ]
        if args.feasibility_weight is not None:
            cmd.append(f"--feasibility-weight={args.feasibility_weight}")
        if args.device:
            cmd.append(f"--device={args.device}")
        if run_seed is not None:
            cmd.append(f"--seed={run_seed}")
        seed_label = f" seed {run_seed}" if run_seed is not None else ""
        print(
            f"[{i}/{len(queued)} {args.repeat_mode} repeat {repeat_no}/{args.repeats}"
            f"{seed_label}] " + " ".join(cmd)
        )
        if args.dry_run:
            dry_runs += 1
            continue
        proc = subprocess.run(cmd, cwd=project_root)
        if proc.returncode == 0:
            successes.append(spec)
        else:
            failures.append((spec, proc.returncode))
            if args.stop_on_error:
                break
        print("")

    print("Batch summary:")
    print(f"- discovered checkpoints: {len(specs)}")
    print(f"- queued executions: {len(queued)}")
    if args.dry_run:
        print(f"- dry-run commands: {dry_runs}")
    print(f"- succeeded: {len(successes)}")
    print(f"- failed: {len(failures)}")

    if failures:
        print("\nFailures:")
        for spec, code in failures:
            print(f"- model={spec.model_label} code={code} ckpt={spec.checkpoint_path}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run xai/action_explainer.py for all g-unirouting checkpoints."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="logs/xai")
    parser.add_argument("--checkpoints-glob", default="output/*/*/*/baseline.pt")
    parser.add_argument("--python-bin", default=".venv/bin/python" if sys.platform != "win32" else ".venv\\Scripts\\python.exe")
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--attribution-methods", default="gradient,integrated_gradients"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-mode", choices=["reproducibility", "robustness"], default="robustness"
    )
    parser.add_argument("--topk-nodes", default="[1,3,5]")
    parser.add_argument("--feasibility-weight", type=float, default=None)
    parser.add_argument("--feasibility-top-m", type=int, default=8)
    parser.add_argument("--feasibility-cost-weight", type=float, default=0.25)
    parser.add_argument("--attr-features", default="auto")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument(
        "--ig-baseline",
        choices=["mean-fill", "zero-with-customers-at-depot", "zero-all", "zero-with-current-locs"],
        default="mean-fill",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-instances-to-store", type=int, default=8)
    parser.add_argument("--model-filter", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
