import multiprocessing
import os
import sys

import lightning

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
import time

import torch
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.datasets import MTVRPDataset
from mavrp.env.models import TransformerModel
from mavrp.env.problems import MTVRP

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch.jit.enable_onednn_fusion(True)
except AttributeError:
    pass

console = Console(record=True, width=300, file=open("output/results.txt", "a+", encoding="utf-8"))
# print the date and time at the beginning of the console
console.print(f"[bold green]Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[/bold green]")


def print_tables(problem, results):
    graph_sizes = results.keys()
    from rich.table import Table

    table = Table(title=problem)
    table.add_column("Model", style="cyan", no_wrap=True)
    for n in graph_sizes:
        table.add_column(f"Avg Cost {n}", style="magenta", min_width=6)
        table.add_column(f"CPU {n}", style="blue", min_width=6)
        table.add_column(f"GAP {n}", style="red", min_width=6)
    ex_key = "50 greedy"
    models = results[ex_key].keys()
    for model in models:
        data = []
        for n in graph_sizes:
            if model not in results[n]:
                data.append("-")
                data.append("-")
            else:
                data.append(f"{results[n][model][0]:.2f}")
                data.append(f"{results[n][model][1]:.2f}")
                data.append(f"{results[n][model][2]:.2f}")
        table.add_row(model, *data)
    console.print(table)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    lightning.seed_everything(Config().seed)
    main_folder = "output"
    variants = MTVRP.get_variants()
    graph_sizes = ["50"]
    inference_modes = ["greedy"]  # , 'beam_search', 'topk', 'topp', 'sample', 'sample_multi_start']
    # inference_modes = ['greedy']
    for problem in variants:
        print("*" * 80)
        print(f"Problem: {problem}")
        print("*" * 80)

        all_combinations = [
            (n + " " + inference_mode + augment)
            for n in graph_sizes
            for inference_mode in inference_modes
            for augment in ["", "-augmented"]
        ]
        results = {key: {} for key in all_combinations}

        # generate datasets
        datasets = dict()
        for n in graph_sizes:
            if "-" not in n:
                filepath = os.path.join("data", "mavrp", n, problem, "test.npz")
                if os.path.exists(filepath):
                    print(f"Loading dataset from {filepath}")
                    datasets[str(n)] = MTVRPDataset.load(filepath)

        for n in graph_sizes:
            config = Config()
            config.graph_size = int(n.split("-")[0])

            test_dataset = datasets[str(config.graph_size)]

            test_loader = DataLoader(
                test_dataset, batch_size=multiprocessing.cpu_count(), shuffle=False, collate_fn=test_dataset.collate_fn
            )

            if "-" not in n:
                if (test_dataset, "pyvrp") and test_dataset.pyvrp["cost"] is not None:
                    cpu = 0.0 if test_dataset.pyvrp["cpu"] is None else test_dataset.pyvrp["cpu"].sum()
                    avg_tl, min_tl, cpu = test_dataset.pyvrp["cost"].mean(), test_dataset.pyvrp["cost"].min(), cpu
                else:
                    avg_tl, min_tl, cpu = 0, 0, 0
                print("Avg Cost: ", avg_tl, " Best Cost: ", min_tl, " CPU: ", cpu)
                for inference_decode_mode in inference_modes:
                    for augment in ["", "-augmented"]:
                        key = n + " " + inference_decode_mode + augment
                        results[key]["pyvrp"] = (avg_tl, cpu, 0.0)

            else:
                for inference_decode_mode in inference_modes:
                    key = n + " " + inference_decode_mode
                    okey = key.split("-")[0] + " " + inference_decode_mode
                    results[key]["pyvrp"] = results[okey]["pyvrp"]

            # if "-" not in n:
            #     pyvrpSolver = PySolver(config)
            #     avg_tl, min_tl, cpu = pyvrpSolver.solve(test_loader, parallel=True)
            #     print("Avg Cost: ", avg_tl, " Best Cost: ", min_tl, " CPU: ", cpu)
            #     for inference_decode_mode in inference_modes:
            #         key = n + ' ' + inference_decode_mode
            #         results[key]['pyvrp'] = (avg_tl, cpu)
            #
            # else:
            #     for inference_decode_mode in inference_modes:
            #         key = n + ' ' + inference_decode_mode
            #         okey = key.split('-')[0] + ' ' + inference_decode_mode
            #         results[key]['pyvrp'] = results[okey]['pyvrp*']

            test_loader = DataLoader(
                test_dataset, batch_size=config.nb_test_samples, shuffle=False, collate_fn=test_dataset.collate_fn
            )

            for inference_decode_mode in inference_modes:
                for config in Config.all():
                    config.inference_decode_mode = inference_decode_mode
                    config.dropout = 0.0
                    # config.device = 'mps'

                    model_path = None
                    loaded = n.split("-")[1] if "-" in n else n

                    loaded = int(loaded)
                    config.graph_size = loaded  # temporary to get the right repr
                    folder = os.path.join(main_folder, config.problem, str(loaded), repr(config))
                    if not os.path.exists(folder):
                        print(f"Folder {folder} does not exist")
                        continue
                    model_path = os.path.join(folder, "baseline.pt")

                    if os.path.exists(model_path):
                        print(f"Loading model from {model_path}")
                        config.dropout = 0.0
                        model = TransformerModel(config)
                        # model.load_from_ckpt(model_path, baseline=True)
                        if inference_decode_mode == "greedy_multi_start":
                            model.use_multi_start()
                        elif inference_decode_mode == "beam_search":
                            model.use_beam_search()
                        state = torch.load(model_path, map_location=config.device, weights_only=True)
                        model.load_state_dict(state, strict=True, assign=True)
                        model = model.to(config.device)
                        model.freeze()
                        # torch.jit.optimize_for_inference(torch.jit.script(model))
                        best_avg_cost = float("+inf")
                        nb_exec = 10 if "sample" in inference_decode_mode else 1
                        for augment in [False, True]:
                            if config.device == "cuda":
                                torch.cuda.synchronize()
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                            elif config.device == "mps":
                                torch.mps.synchronize()
                                start = torch.mps.Event(enable_timing=True)
                                end = torch.mps.Event(enable_timing=True)
                                start.record()
                            else:
                                start_time = time.process_time()
                            for _ in range(nb_exec):
                                costs = torch.tensor([], device=config.device)
                                for i, batch in enumerate(
                                    tqdm(test_loader, desc=model.get_name() + "-" + inference_decode_mode)
                                ):
                                    with torch.inference_mode():
                                        batch = [d.to(config.device) for d in batch]
                                        if augment:
                                            _, solutions, model_costs = model.inference(
                                                batch, decode_mode=inference_decode_mode
                                            )
                                        else:
                                            _, solutions, model_costs = model(batch, decode_mode=inference_decode_mode)
                                        costs = torch.cat((costs, model_costs), dim=0)
                                avg_cost = torch.mean(costs).detach().item()
                                best_avg_cost = min(best_avg_cost, avg_cost)
                                gap_pyvrp = ((best_avg_cost - avg_tl) * 100 / avg_tl).item()
                            if config.device == "cuda":
                                end.record()
                                torch.cuda.synchronize()
                                cpu = start.elapsed_time(end) / 1000
                            elif config.device == "mps":
                                end.record()
                                torch.mps.synchronize()
                                cpu = start.elapsed_time(end) / 1000
                            else:
                                cpu = time.process_time() - start_time
                            key = n + " " + inference_decode_mode + ("-augmented" if augment else "")
                            results[key][config.encoder.__name__] = (best_avg_cost, cpu, gap_pyvrp)
                            print(key, config.encoder.__name__, best_avg_cost, cpu, gap_pyvrp)

                        if config.device == "cuda":
                            torch.cuda.empty_cache()

        print_tables(problem, results)
