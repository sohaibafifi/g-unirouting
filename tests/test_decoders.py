import pytest
import lightning
import torch
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.decoders import EndToEndDecoder
from mavrp.env.decoders.multistart import MultiStartDecoder
from mavrp.env.encoders import GATEncoder, GATv2Encoder
from mavrp.env.encoders.transformerconv import TransformerEncoder


@pytest.fixture
def config_and_loader():
    """Fixture to create config and data loader for tests."""
    config = Config()
    lightning.seed_everything(config.seed)
    config.graph_size = 16
    config.batch_size = 9
    problem = config.get_problem()
    dataset = config.get_problem().dataset(
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


def assert_valid_decoder_output(output, batch_size, graph_size):
    """Helper function to validate decoder output."""
    assert isinstance(output, tuple), "Decoder must return a tuple"
    assert len(output) == 3, "Decoder must return a tuple of size 3"
    log_probabilities, solutions, costs = output

    assert not torch.isinf(costs).any(), "Costs should not be infinite"
    assert not (costs >= 1e20).any(), "Costs should be less than 1e20"
    assert not (costs == 0).any(), "Costs should not be zero"
    assert not torch.isnan(costs).any(), "Costs should not be NaN"
    assert not torch.isinf(solutions).any(), "Solutions should not be infinite"

    sum_cost = costs.sum()
    assert solutions.shape[0] == batch_size
    assert solutions.shape[1] >= graph_size, "Solutions should have at least graph_size columns"
    assert costs.shape == (batch_size,)
    assert isinstance(sum_cost, torch.Tensor)


@pytest.mark.parametrize("decode_mode", ["greedy", "sample"])
def test_decoder(config_and_loader, decode_mode):
    config, loader = config_and_loader
    encoder = config.encoder(config).to(config.device)
    decoder = EndToEndDecoder(config).to(config.device)

    for data in loader:
        encoded_data, global_embeddings = encoder([x.to(config.device) for x in data])
        output = decoder(data, encoded_data, global_embeddings, decode_mode=decode_mode)

        assert_valid_decoder_output(output, loader.batch_size, config.graph_size)

        log_probabilities, solutions, costs = output
        _, new_solutions, new_costs = decoder(
            data, encoded_data, global_embeddings,
            decode_mode=decode_mode,
            actions=solutions
        )
        assert new_costs.shape == (loader.batch_size,)
        assert torch.equal(solutions, new_solutions)


@pytest.mark.parametrize("encoder_class", [GATEncoder, GATv2Encoder, TransformerEncoder])
@pytest.mark.parametrize("decode_mode", ["greedy", "sample"])
def test_decoder_with_edge_attention(config_and_loader, encoder_class, decode_mode):
    config, loader = config_and_loader
    config.use_edge_attn = True
    encoder = encoder_class(config).to(config.device)
    decoder = EndToEndDecoder(config).to(config.device)

    for data in loader:
        encoded_data, global_embeddings, edge_index, edge_attn_scores = encoder(
            [x.to(config.device) for x in data]
        )
        output = decoder(
            data, encoded_data, global_embeddings,
            decode_mode=decode_mode,
            edge_index=edge_index,
            edge_attn_scores=edge_attn_scores
        )

        assert_valid_decoder_output(output, loader.batch_size, config.graph_size)


@pytest.mark.parametrize("decode_mode", ["greedy", "sample"])
def test_ms_decoder(config_and_loader, decode_mode):
    config, loader = config_and_loader
    encoder = config.encoder(config).to(config.device)
    decoder = MultiStartDecoder(config).to(config.device)

    for data in loader:
        encoded_data, global_embeddings = encoder([x.to(config.device) for x in data])
        output = decoder(data, encoded_data, global_embeddings, decode_mode=decode_mode)

        assert_valid_decoder_output(output, loader.batch_size, config.graph_size)

