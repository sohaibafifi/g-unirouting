import os

import lightning
import numpy

from mavrp.env.datasets import MTVRPDataset
from mavrp.env.problems import MTVRP

if __name__ == "__main__":
    lightning.seed_everything(1234)
    data_folder = "data/mavrp/"
    for graph_size in [50, 100]:
        for variant in MTVRP.get_variants():
            for type in ["test"]:
                folder = os.path.join(data_folder, str(graph_size), variant)
                os.makedirs(folder, exist_ok=True)
                file_path = os.path.join(folder, f"{type}.npz")
                dataset = MTVRPDataset(graph_size=graph_size, variant=variant, num_samples=2**10)
                dataset.save(file_path)
