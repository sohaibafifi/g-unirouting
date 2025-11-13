import pytest
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.normalization import CostNormalization


@pytest.fixture
def config_and_loader():
    """Fixture to create config and data loader for normalization tests."""
    config = Config()
    problem = config.get_problem()
    config.batch_size = 256
    dataset = problem.dataset(
        graph_size=config.graph_size,
        num_samples=config.batch_size * 4,
        device=config.device
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    return config, loader


def test_cost_normalization(config_and_loader):
    """Test cost normalization functionality."""
    config, loader = config_and_loader
    encoder = config.encoder(config).to(config.device)
    decoder = config.decoder(config).to(config.device)
    normalization = CostNormalization()

    for data in loader:
        encoded_data, global_embeddings = encoder([x.to(config.device) for x in data])
        log_probabilities, solutions, costs = decoder(data, encoded_data, global_embeddings)

        normalized_cost, norm_vals = normalization(data, costs)
        assert normalized_cost.shape == costs.shape, \
            "Normalized cost should have the same shape as cost"

        norm_vals = normalization.norm_vals
        if config.problem == 'MTVRP':
            for variant in norm_vals:
                assert norm_vals[variant]["mean"] > 0.0, \
                    f"Mean value should be greater than 0 for instance {variant}"
                assert norm_vals[variant]["count"] > 0.0, \
                    f"Count should be greater than 0 for instance {variant}"
