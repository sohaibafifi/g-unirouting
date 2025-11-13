import pytest
import torch
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.encoders import GATv2Encoder
from mavrp.env.models import TransformerModel
from mavrp.env.models import CriticModel


@pytest.fixture
def config_and_loader():
    """Fixture to create config and data loader for model tests."""
    config = Config()
    problem = config.get_problem()
    config.encoder = GATv2Encoder
    config.batch_size = 4
    config.graph_size = 16
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


def test_transformers(config_and_loader):
    """Test TransformerModel output format."""
    config, loader = config_and_loader
    model = TransformerModel(config).to(config.device)
    model.eval()

    for data in loader:
        output = model(data, decode_mode='sample')
        assert isinstance(output, tuple), "TransformerModel must return a tuple"
        assert len(output) == 3, "TransformerModel must return a tuple of size 3 if return_raw_logits is False"
        assert output[0].shape == (config.batch_size,), \
            f"log_prob must have shape (batch_size,) but got {output[0].shape}"
        assert output[1].shape[0] >= config.batch_size, \
            f"solutions must have shape (batch_size, seq_len) but got {output[1].shape}"
        assert output[1].shape[1] >= config.graph_size, \
            f"solutions must have shape (batch_size, seq_len >= graph_size) but got {output[1].shape}"


def test_critic(config_and_loader):
    """Test that CriticModel returns a value tensor of shape (batch_size,)."""
    config, loader = config_and_loader
    critic = CriticModel(config).to(config.device)
    critic.eval()

    for data in loader:
        values = critic(data)
        assert isinstance(values, torch.Tensor), "CriticModel must return a torch.Tensor"
        assert values.shape == (config.batch_size,), (
            f"CriticModel output shape must be (batch_size,) but got {values.shape}"
        )
        break

