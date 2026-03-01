from __future__ import annotations

import argparse
import csv
import math

from pathlib import Path
from typing import Iterable


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc


def _union_headers(csv_paths: Iterable[Path]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for csv_path in csv_paths:
        for column in _read_header(csv_path):
            if column not in seen:
                seen.add(column)
                ordered.append(column)
    return ordered


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _resolve_x_column(header: Iterable[str], requested: str) -> str | None:
    columns = list(header)
    if requested != "auto":
        return requested
    for candidate in ("step", "epoch"):
        if candidate in columns:
            return candidate
    return None


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values

    out: list[float] = []
    running_sum = 0.0
    start = 0
    for end, value in enumerate(values):
        running_sum += value
        if end - start + 1 > window:
            running_sum -= values[start]
            start += 1
        out.append(running_sum / (end - start + 1))
    return out


def _default_label(csv_path: Path) -> str:
    if (
        csv_path.name == "metrics.csv"
        and csv_path.parent.name.startswith("version_")
        and csv_path.parent.parent != csv_path.parent
    ):
        return f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
    return csv_path.stem


def _experiment_dir(csv_path: Path) -> Path:
    if csv_path.parent.name.startswith("version_") and csv_path.parent.parent.name == "lightning_logs":
        return csv_path.parent.parent.parent
    return csv_path.parent


def _group_label(csv_paths: list[Path], root: Path | None) -> str:
    if not csv_paths:
        return "-"
    exp_dir = _experiment_dir(csv_paths[0])
    if root is not None:
        try:
            return str(exp_dir.relative_to(root))
        except ValueError:
            pass
    return exp_dir.name


def _default_output(
    csv_paths: list[Path],
    metric: str,
    *,
    discovered: bool,
    root: Path | None,
) -> Path:
    if len(csv_paths) == 1 and not discovered:
        return csv_paths[0].with_name(f"{metric}.png")
    if discovered and root is not None:
        return root / "plots" / f"{metric}_all_runs.png"
    return Path.cwd() / f"{metric}_compare.png"


def _discover_csv_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(path.resolve() for path in root.glob(pattern) if path.is_file())


def _load_series(csv_path: Path, metric: str, x_column: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if metric not in reader.fieldnames:
            raise KeyError(f"Metric '{metric}' not found in {csv_path}")
        if x_column != "row" and x_column not in reader.fieldnames:
            raise KeyError(f"X column '{x_column}' not found in {csv_path}")

        for row_index, row in enumerate(reader):
            y = _parse_float(row.get(metric))
            if y is None:
                continue
            if x_column == "row":
                x = float(row_index)
            else:
                x = _parse_float(row.get(x_column))
                if x is None:
                    continue
            xs.append(x)
            ys.append(y)

    if not ys:
        raise ValueError(f"No numeric values found for '{metric}' in {csv_path}")

    paired = sorted(zip(xs, ys), key=lambda pair: pair[0])
    return [x for x, _ in paired], [y for _, y in paired]


def _aggregate_series(
    series_list: list[tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]]:
    buckets: dict[float, list[float]] = {}
    for xs, ys in series_list:
        for x, y in zip(xs, ys):
            buckets.setdefault(float(x), []).append(float(y))
    if not buckets:
        raise ValueError("No numeric points available to aggregate.")

    xs_sorted = sorted(buckets.keys())
    ys_mean = [sum(buckets[x]) / len(buckets[x]) for x in xs_sorted]
    return xs_sorted, ys_mean


def _group_csv_paths(csv_paths: list[Path]) -> list[list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for csv_path in csv_paths:
        key = _experiment_dir(csv_path)
        groups.setdefault(key, []).append(csv_path)
    return [sorted(group) for _, group in sorted(groups.items(), key=lambda item: str(item[0]))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one metric column from Lightning metrics.csv files. By default, "
            "the script discovers all runs under output/ and aggregates versions "
            "of the same experiment."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="Optional metrics.csv files. If omitted, all matching CSVs under --root are used.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric column to plot, for example test_avg_cost.",
    )
    parser.add_argument(
        "--x-column",
        default="auto",
        help="X-axis column. Use 'auto' (default), a column name, or 'row'.",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Optional trailing rolling mean window applied to Y values.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--xlabel", default=None, help="Optional X-axis label override.")
    parser.add_argument("--ylabel", default=None, help="Optional Y-axis label override.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    parser.add_argument(
        "--root",
        default="output",
        help="Root directory used when csv_paths are omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="**/lightning_logs/version_*/metrics.csv",
        help="Glob pattern under --root used for auto-discovery.",
    )
    parser.add_argument(
        "--aggregate-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate multiple versioned runs of the same experiment into one curve.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the available columns from the first CSV and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    discovered = False
    root = Path(args.root).expanduser().resolve()
    if args.csv_paths:
        csv_paths = [Path(path).expanduser().resolve() for path in args.csv_paths]
        root_for_labels: Path | None = None
    else:
        discovered = True
        csv_paths = _discover_csv_paths(root, args.pattern)
        root_for_labels = root
        if not csv_paths:
            raise FileNotFoundError(
                f"No metrics.csv files found under {root} matching {args.pattern}"
            )
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    first_header = _read_header(csv_paths[0])
    if args.list_metrics:
        print("\n".join(_union_headers(csv_paths)))
        return
    if not args.metric:
        raise SystemExit("Pass --metric (or use --list-metrics to inspect the CSV header).")

    x_column = _resolve_x_column(first_header, args.x_column)
    if x_column is None:
        x_column = "row"

    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the environment first."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = _group_csv_paths(csv_paths) if args.aggregate_runs else [[path] for path in csv_paths]
    plotted = 0
    for group in groups:
        series_list: list[tuple[list[float], list[float]]] = []
        for csv_path in group:
            try:
                series_list.append(_load_series(csv_path, metric=args.metric, x_column=x_column))
            except (ValueError, KeyError):
                continue
        if not series_list:
            continue
        xs, ys = _aggregate_series(series_list) if len(series_list) > 1 else series_list[0]
        ys = _rolling_mean(ys, max(int(args.rolling), 1))
        label = (
            f"{_group_label(group, root_for_labels)} (n={len(series_list)})"
            if args.aggregate_runs and len(group) > 1
            else (
                _group_label(group, root_for_labels)
                if args.aggregate_runs
                else _default_label(group[0])
            )
        )
        marker = "o" if len(xs) <= 50 else None
        ax.plot(xs, ys, label=label, linewidth=2.0, marker=marker, markersize=3)
        plotted += 1

    if plotted == 0:
        raise SystemExit(
            f"No numeric series found for metric '{args.metric}' in the selected CSV files."
        )

    ax.set_title(args.title or f"{args.metric} over {x_column}")
    ax.set_xlabel(args.xlabel or x_column)
    ax.set_ylabel(args.ylabel or args.metric)
    ax.grid(True, alpha=0.3)
    if plotted > 1:
        ax.legend()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output(csv_paths, args.metric, discovered=discovered, root=root_for_labels)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(args.dpi))
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
