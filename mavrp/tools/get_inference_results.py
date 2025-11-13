import wandb
from rich.console import Console
from rich.table import Table

from mavrp.env.problems import MTVRP

if __name__ == "__main__":
    results = dict()
    for size in [50, 100]:
        results[size] = dict()
        for variant in MTVRP.get_variants():
            results[size][variant] = dict()

    wandb.init()
    api = wandb.Api()
    runs = api.runs(path="inference")
    for run in runs:

        if run.config["description"] == "pyvrp":
            if "Pyvrp Cost" not in run.summaryMetrics:
                continue
            cost = run.summaryMetrics["Pyvrp Cost"]
            cpu = run.summaryMetrics["Pyvrp CPU"]
            # convert to float
            if isinstance(cost, str):
                cost = float(cost)
            if isinstance(cpu, str):
                cpu = float(cpu)
            results[run.summaryMetrics["graph_size"]][run.summaryMetrics["variant"]]["pyvrp"] = (cost, cpu)
        else:
            if "Cost" not in run.summaryMetrics:
                continue
            method = run.summaryMetrics["encoder"]
            cost = run.summaryMetrics["Cost"]
            cpu = run.summaryMetrics["CPU Time"]
            # convert to float
            if isinstance(cost, str):
                cost = float(cost)
            if isinstance(cpu, str):
                cpu = float(cpu)

            results[run.summaryMetrics["graph_size"]][run.summaryMetrics["variant"]][method] = (cost, cpu)
    # collect all the methods
    methods = set()
    for size in results:
        for variant in results[size]:
            for method in results[size][variant]:
                methods.add(method)
    # collect all the variants
    variants = set()
    for size in results:
        for variant in results[size]:
            variants.add(variant)

    console = Console()
    table = Table(title="Inference Results")
    table.add_column("Variant", style="cyan", no_wrap=True)
    for size in results.keys():
        for method in methods:
            table.add_column(f"{size} {method}", justify="right", min_width=6)
            table.add_column(f"{size} {method} cpu", justify="right", min_width=6)

    for variant in variants:
        data = []
        for size in results.keys():
            for method in methods:
                if method not in results[size][variant]:
                    data.append("-")
                    data.append("-")
                else:
                    data.append(f"{results[size][variant][method][0]:.2f}")
                    data.append(f"{results[size][variant][method][1]:.2f}")
        table.add_row(variant, *data)
    console.print(table)
