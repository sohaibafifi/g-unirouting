import pytest
from torch.utils.data import DataLoader

from mavrp.configs.config import Config
from mavrp.env.encoders.graph import GraphEmbeddingLayer, Graph
from mavrp.env.encoders.init import InitialEmbeddingLayer


@pytest.fixture
def config_and_loader():
    """Fixture to create config and data loader for layer tests."""
    config = Config()
    problem = config.get_problem()
    config.batch_size = 16
    config.graph_size = 10
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


def test_scalenorm(config_and_loader):
    """Test ScaleNorm layer."""
    config, loader = config_and_loader
    from mavrp.env.normalization import ScaleNorm
    layer = ScaleNorm(config.embedding_dim).to(config.device)

    for data in loader:
        node_features, global_features = [x.to(config.device) for x in data]
        embedder = InitialEmbeddingLayer(config).to(config.device)
        node_embeddings, global_embeddings = embedder(node_features, global_features)
        encoded_data = layer(node_embeddings)

        assert encoded_data.shape == (loader.batch_size, config.graph_size, config.embedding_dim), \
            f"Expected {(loader.batch_size, config.graph_size, config.embedding_dim)} but got {encoded_data.shape}"
        assert layer.scale.item() == config.embedding_dim, \
            f"Expected scale to be {config.embedding_dim} but got {layer.scale.item()}"


def test_initial_embedding(config_and_loader):
    """Test InitialEmbeddingLayer."""
    config, loader = config_and_loader
    layer = InitialEmbeddingLayer(config).to(config.device)

    for data in loader:
        node_features, global_features = [x.to(config.device) for x in data]
        node_embeddings, global_node_embedding = layer(node_features, global_features)

        assert node_embeddings.shape == (loader.batch_size, config.graph_size, config.embedding_dim), \
            f"Expected {(config.batch_size, config.graph_size, config.embedding_dim)} but got {node_embeddings.shape}"
        assert global_node_embedding.shape == (loader.batch_size, config.embedding_dim), \
            f"Expected {(config.batch_size, config.embedding_dim)} but got {global_node_embedding.shape}"


@pytest.mark.parametrize("sampling", [True, False])
def test_graph_embedding(config_and_loader, sampling):
    """Test GraphEmbeddingLayer with and without neighbor sampling."""
    config, loader = config_and_loader
    initial_layer = InitialEmbeddingLayer(config).to(config.device)
    graph_layer = GraphEmbeddingLayer(config).to(config.device)
    graph_layer.sample_neighbors = sampling

    for data in loader:
        node_features, global_features = [x.to(config.device) for x in data]
        encoded_data, global_embeddings = initial_layer(node_features, global_features)
        encoded_data, global_embeddings = graph_layer(
            node_features,
            global_features,
            encoded_data,
            global_embeddings
        )

        assert isinstance(encoded_data, Graph), "Expected a graph but got something else"
        encoded_data, edge_index, edge_attr = encoded_data.x, encoded_data.edge_index, encoded_data.edge_attr
        assert encoded_data.shape == (loader.batch_size * config.graph_size, config.embedding_dim), \
            f"Expected {(loader.batch_size * config.graph_size, config.embedding_dim)} but got {encoded_data.shape}"
        num_edges = loader.batch_size * ((config.graph_size) * (config.nb_neighbors + 1))
        assert edge_index.shape[0] == 2, f"Expected 2 but got {edge_index.shape[0]}"
        assert edge_index.shape[1] <= num_edges, f"Expected <={num_edges} but got {edge_index.shape[1]}"
        assert edge_attr.shape[0] <= num_edges, f"Expected <={num_edges} but got {edge_attr.shape[0]}"
        assert edge_attr.shape[1] == config.embedding_dim, \
            f"Expected {config.embedding_dim} but got {edge_attr.shape[1]}"
