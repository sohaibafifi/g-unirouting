import pytest
import lightning
import torch

from mavrp.env.datasets import MTVRPDataset
from mavrp.env.problems import MTVRP


@pytest.fixture
def problem_and_dataset():
    """Fixture to create problem and dataset for tests."""
    lightning.seed_everything(1234)
    problem = MTVRP()
    dataset = MTVRPDataset(graph_size=25, num_samples=100)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    instance = dataset[0]
    return problem, dataset, dataloader, instance


def test_mtvrp(problem_and_dataset):
    """Test MTVRP problem node and global features."""
    problem, dataset, dataloader, instance = problem_and_dataset
    node_features, global_features = instance

    assert node_features.size(1) == problem.num_node_features(), \
        f"Expected {problem.num_node_features()} but got {node_features.size(1)}"
    assert global_features.size(0) == problem.num_global_features(), \
        f"Expected {problem.num_global_features()} but got {global_features.size(1)}"


def extract_routes(solution):
    """Helper function to extract routes from solution."""
    routes = []
    current_route = []
    for node in solution:
        if node == 0:
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(node)
    return routes
