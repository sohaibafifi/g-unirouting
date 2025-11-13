import pytest
import torch
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.baselines import RolloutBaseline
from mavrp.env.models import TransformerModel


@pytest.fixture
def baseline_and_loader():
    """Fixture to create baseline and data loader for tests."""
    config = Config()
    config.batch_size = 16
    model = TransformerModel(config)
    baseline = RolloutBaseline(config, model=model)
    problem = config.get_problem()
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
    return baseline, model, loader


def test_evaluate(baseline_and_loader):
    """Test baseline evaluation on data."""
    baseline, model, loader = baseline_and_loader
    for i, data in enumerate(loader):
        tour_lengths = baseline.evaluate(data)
        assert not torch.isnan(tour_lengths).any()


def test_evaluate_raises_error_on_none_inputs(baseline_and_loader):
    """Test that baseline raises ValueError on None inputs."""
    baseline, model, loader = baseline_and_loader
    with pytest.raises(ValueError):
        baseline.evaluate(None)


def test_state(baseline_and_loader):
    """Test baseline state is a dictionary."""
    baseline, model, loader = baseline_and_loader
    state = baseline.state()
    assert isinstance(state, dict)


def test_load_state(baseline_and_loader):
    """Test baseline state loading."""
    baseline, model, loader = baseline_and_loader
    state = model.state_dict()
    baseline.load_state(state)
    loaded_state = baseline.state()
    assert state.keys() == loaded_state.keys()
