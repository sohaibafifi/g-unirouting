import torch

from mavrp.configs.config import Config
from mavrp.env.decoders import EndToEndDecoder, RecourseDecoder


def _build_case() -> tuple[torch.Tensor, torch.Tensor]:
    node_features = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1_000.0, 0.0],  # depot
                [1.0, 0.0, 2.0, 0.0, 0.0, 1_000.0, 0.0],  # infeasible by capacity
                [2.0, 0.0, 1.0, 0.0, 0.0, 1_000.0, 0.0],  # feasible
            ]
        ],
        dtype=torch.float32,
    )
    global_features = torch.tensor(
        [[1.0, 0.0, 0.0, 1_000.0, 1_000.0, 0.0]],
        dtype=torch.float32,
    )
    return node_features, global_features


def test_recourse_decoder_adds_rescue_trip_and_preserves_main_route_state():
    config = Config()
    config.device = "cpu"
    config.decoder = RecourseDecoder

    standard_decoder = EndToEndDecoder(config)
    recourse_decoder = RecourseDecoder(config)

    node_features, global_features = _build_case()
    batch_size, seq_len, _ = node_features.shape
    node_embeddings = torch.randn(batch_size, seq_len, config.embedding_dim)
    global_embeddings = torch.randn(batch_size, config.embedding_dim)
    forced_actions = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)

    _, solution_std, cost_std = standard_decoder(
        (node_features, global_features),
        node_embeddings,
        global_embeddings,
        decode_mode="greedy",
        actions=forced_actions,
    )
    _, solution_rec, cost_rec, _ = recourse_decoder(
        (node_features, global_features),
        node_embeddings,
        global_embeddings,
        decode_mode="greedy",
        actions=forced_actions,
    )

    assert torch.equal(solution_std, forced_actions)
    assert torch.equal(solution_rec, forced_actions)
    assert torch.allclose(cost_std, torch.tensor([4.0]))
    assert torch.allclose(cost_rec, torch.tensor([6.0]))


def test_recourse_decoder_open_route_only_charges_outbound_leg():
    config = Config()
    config.device = "cpu"
    config.decoder = RecourseDecoder

    recourse_decoder = RecourseDecoder(config)
    node_features, global_features = _build_case()
    global_features[:, 1] = 1.0  # open route
    batch_size, seq_len, _ = node_features.shape
    node_embeddings = torch.randn(batch_size, seq_len, config.embedding_dim)
    global_embeddings = torch.randn(batch_size, config.embedding_dim)
    forced_actions = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)

    log_probabilities, solution, cost, metrics = recourse_decoder(
        (node_features, global_features),
        node_embeddings,
        global_embeddings,
        decode_mode="greedy",
        actions=forced_actions,
    )

    assert torch.equal(solution, forced_actions)
    assert torch.allclose(cost, torch.tensor([3.0]))
