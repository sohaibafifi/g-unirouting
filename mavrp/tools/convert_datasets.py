import multiprocessing
import os
import time

import lightning
import numpy
import numpy as np
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.datasets import MTVRPDataset
from mavrp.env.problems import MTVRP
from mavrp.env.solvers import PySolver


def process_dataset(data_folder, target_folder, graph_size, variant, dataset_type):
    folder = os.path.join(data_folder, str(graph_size), variant)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{dataset_type}.npz")

    print(f"Processing {file_path}", flush=True)

    dataset = MTVRPDataset.load(file_path)
    target_path = file_path.replace(data_folder, target_folder)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    data = dataset.to_data()
    np.savez(target_path, **data)


if __name__ == "__main__":
    lightning.seed_everything(1234)
    data_folder = "data/mtvrp/"
    target_folder = "rf-data/mtvrp/"
    os.makedirs(target_folder, exist_ok=True)

    for graph_size in [50]:
        for variant in MTVRP.get_variants():
            for dataset_type in ["test"]:
                process_dataset(data_folder, target_folder, graph_size, variant, dataset_type)
