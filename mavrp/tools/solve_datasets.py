import multiprocessing
import os
import time

import lightning
import numpy
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.datasets import MTVRPDataset
from mavrp.env.problems import MTVRP
from mavrp.env.solvers import PySolver


def solve_single_instance(args):
    """
    args is a tuple: (index, instance)
    """
    idx, instance = args
    pyvrpSolver = PySolver(Config())
    start_time = time.time()
    cost = pyvrpSolver.process_instance(*instance)
    cpu_time = time.time() - start_time
    actions = pyvrpSolver.sequence

    # Return the index alongside the results
    return idx, (actions, cost, cpu_time)


def process_dataset(data_folder, graph_size, variant, dataset_type):
    folder = os.path.join(data_folder, str(graph_size), variant)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{dataset_type}.npz")

    print(f"Processing {file_path}", flush=True)

    dataset = list(MTVRPDataset.load(file_path))

    indexed_dataset = list(enumerate(dataset))
    num_instances = len(indexed_dataset)

    results = []
    timeout = 5  # Timeout in seconds for each instance

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        async_results = [pool.apply_async(solve_single_instance, (args,)) for args in indexed_dataset]

        for idx, async_res in enumerate(
            tqdm(async_results, total=num_instances, desc=f"Solving Instances {dataset_type} {graph_size} {variant}")
        ):
            try:
                res = async_res.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                res = (idx, ([], float("inf"), timeout))  # Return inf cost on timeout
            results.append(res)

    results.sort(key=lambda x: x[0])

    actions, costs, cpus = zip(*[r[1] for r in results])

    # Save solutions
    solution_path = os.path.join(folder, f"{dataset_type}_sol.npz")
    numpy.savez(solution_path, actions=actions, cost=costs, cpu=cpus)


if __name__ == "__main__":
    lightning.seed_everything(1234)
    data_folder = "data/mtvrp/"

    for graph_size in [26, 51, 101]:
        for variant in MTVRP.get_variants():
            for dataset_type in ["val", "test"]:
                process_dataset(data_folder, graph_size, variant, dataset_type)
