import pytest
import human_readable
import numpy as np
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.datasets import MTVRPDataset
from mavrp.env.problems import MTVRP


@pytest.mark.parametrize("variant", MTVRP.get_variants())
def test_mtvrp_dataset(variant):
    """Test MTVRPDataset with different variants."""
    graph_size = 20
    num_samples = 1000
    problem = MTVRP()
    dataset = MTVRPDataset(graph_size=graph_size, num_samples=num_samples, variant=variant)
    assert len(dataset) == 1000

    for i in range(len(dataset)):
        sample = dataset[i]
        node_features, global_features = sample
        num_features = problem.num_node_features()
        num_global_features = problem.num_global_features()
        assert node_features.size(0) == graph_size, \
            f"Expected {graph_size} but got {node_features.size(1)}"
        assert node_features.size(1) == num_features, \
            f"Expected {num_features} but got {node_features.size(2)}"
        assert global_features.size(0) == num_global_features, \
            f"Expected {num_global_features} but got {global_features.size(1)}"
        # check for nan and inf values
        assert not node_features.isnan().any(), "Node features contain nan values"
        assert not global_features.isnan().any(), "Global features contain nan values"


def test_dataloader():
    """Test DataLoader with MTVRPDataset."""
    config = Config()
    problem = MTVRP()
    config.graph_size = 100
    dataset = MTVRPDataset(
        graph_size=config.graph_size,
        num_samples=config.nb_train_samples,
        device=config.device
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=MTVRPDataset.collate_fn
    )
    size = dataset.get_size()
    print(f"The size of a Dataset of {config.graph_size} nodes : {human_readable.file_size(size)}")

    for data in loader:
        node_features, global_features = [x.to(config.device) for x in data]
        num_features = problem.num_node_features()
        num_global_features = problem.num_global_features()
        assert node_features.size(0) == config.batch_size, \
            f"Expected {config.batch_size} but got {node_features.size(0)}"
        assert node_features.size(1) == config.graph_size, \
            f"Expected {config.graph_size} but got {node_features.size(1)}"
        assert node_features.size(2) == num_features, \
            f"Expected {num_features} but got {node_features.size(2)}"
        assert global_features.size(0) == config.batch_size, \
            f"Expected {config.batch_size} but got {global_features.size(0)}"
        assert global_features.size(1) == num_global_features, \
            f"Expected {num_global_features} but got {global_features.size(1)}"
        # check for nan and inf values
        assert not node_features.isnan().any(), "Node features contain nan values"
        assert not global_features.isnan().any(), "Global features contain nan values"


def test_save_and_load():
    """Test saving and loading dataset in different formats."""
    import os
    config = Config()
    config.graph_size = 100
    dataset = MTVRPDataset(
        graph_size=config.graph_size,
        num_samples=config.nb_train_samples,
        device=config.device
    )

    # Test saving and loading with .pt format
    dataset.save("test.pt")
    loaded = MTVRPDataset.load("test.pt")
    assert len(dataset) == len(loaded), "Dataset length mismatch when saving using tensors"
    os.remove("test.pt")

    # Test saving and loading with .npz format
    dataset.save("test.npz")
    loaded = MTVRPDataset.load("test.npz")
    assert len(dataset) == len(loaded), "Dataset length mismatch when saving using numpy"
    os.remove("test.npz")

    # Test converting to numpy and saving
    data = dataset.to_data()
    # convert data to numpy
    data = {k: v.cpu().numpy() for k, v in data.items()}
    np.savez("test-rf.npz", **data)

