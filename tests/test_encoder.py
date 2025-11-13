import pytest
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.encoders.gps import GPSEncoder
from mavrp.env.encoders.matnet import MixedScoresEncoder
from mavrp.env.encoders.gatv2 import GATv2Encoder
from mavrp.env.encoders.gat import GATEncoder
from mavrp.env.encoders.sage import SageEncoder
from mavrp.env.encoders.transformerconv import TransformerEncoder
from mavrp.env.encoders import PerformerEncoder, AttentionEncoder


@pytest.fixture
def config_and_loader():
    """Fixture to create config and data loader for encoder tests."""
    config = Config()
    config.graph_size = 16
    config.batch_size = 4
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
    return config, loader


def assert_valid_encoder_output(encoded_data, batch_size, graph_size, embedding_dim):
    """Helper function to validate encoder output."""
    assert encoded_data.shape == (batch_size, graph_size, embedding_dim), \
        f"Expected {(batch_size, graph_size, embedding_dim)} but got {encoded_data.shape}"
    assert not encoded_data.isnan().any(), "Encoded data contains nan values"


def test_performer_encoder(config_and_loader):
    """Test PerformerEncoder."""
    config, loader = config_and_loader
    encoder = PerformerEncoder(config).to(config.device)
    encoder.reset_parameters()

    for data in loader:
        encoded_data, global_embeddings = encoder([x.to(config.device) for x in data])
        assert_valid_encoder_output(
            encoded_data,
            loader.batch_size,
            config.graph_size,
            config.embedding_dim
        )


@pytest.mark.parametrize("use_moe", [False, True])
def test_attention_encoder(config_and_loader, use_moe):
    """Test AttentionEncoder with and without MoE."""
    config, loader = config_and_loader
    config.use_moe = use_moe
    encoder = AttentionEncoder(config).to(config.device)
    encoder.reset_parameters()

    for data in loader:
        encoded_data, global_embeddings = encoder([x.to(config.device) for x in data])
        assert_valid_encoder_output(
            encoded_data,
            loader.batch_size,
            config.graph_size,
            config.embedding_dim
        )


@pytest.mark.parametrize("encoder_class", [
    SageEncoder,
    GPSEncoder,
    MixedScoresEncoder,
])
def test_basic_encoders(config_and_loader, encoder_class):
    """Test encoders that return (encoded_data, global_embeddings)."""
    config, loader = config_and_loader
    encoder = encoder_class(config).to(config.device)
    encoder.reset_parameters()

    for data in loader:
        encoded_data, global_embeddings = encoder(data)
        assert_valid_encoder_output(
            encoded_data,
            loader.batch_size,
            config.graph_size,
            config.embedding_dim
        )


@pytest.mark.parametrize("encoder_class", [
    GATEncoder,
    GATv2Encoder,
    TransformerEncoder,
])
def test_attention_based_encoders(config_and_loader, encoder_class):
    """Test encoders that return (encoded_data, global_embeddings, edge_index, edge_attn_scores)."""
    config, loader = config_and_loader
    encoder = encoder_class(config).to(config.device)
    encoder.reset_parameters()

    for data in loader:
        encoded_data, global_embeddings, edge_index, edge_attn_scores = encoder(data)
        assert_valid_encoder_output(
            encoded_data,
            loader.batch_size,
            config.graph_size,
            config.embedding_dim
        )
        # Additional checks for attention-based encoders
        assert edge_index is not None, "edge_index should not be None"
        assert edge_attn_scores is not None, "edge_attn_scores should not be None"

