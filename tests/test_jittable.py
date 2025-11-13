import pytest
import torch

from mavrp.configs.config import Config
from mavrp.env.decoders import EndToEndDecoder
from mavrp.env.decoders.multistart import MultiStartDecoder
from mavrp.env.decoders.pointer import PointerAttention
from mavrp.env.encoders import MixedScoresEncoder, GATEncoder, GATv2Encoder, PerformerEncoder, GPSEncoder
from mavrp.env.encoders import AttentionEncoder
from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.encoders.graph import GraphEmbeddingLayer
from mavrp.env.encoders.matnet import MixedScoresMHA, MixedScoresSDPA
from mavrp.env.encoders.mlp import MLP
from mavrp.env.encoders.attention import MultiHeadAttention, TransformerBlock
from mavrp.env.encoders.sage import SageTransformerBlock, SageEncoder
from mavrp.env.models import TransformerModel
from mavrp.env.normalization import Normalization
from mavrp.env.train.trainers import ReinforceTrainer


@pytest.fixture
def jit_config():
    """Fixture to get a config for JIT tests."""
    return Config.all()[0]


def test_rf_mha_is_jittable(jit_config):
    """Test that MultiHeadAttention is JIT-scriptable."""
    config = jit_config
    mha = MultiHeadAttention(config.embedding_dim, config.n_heads, False)
    mha = torch.jit.script(mha)
    print(mha.code)


def test_mixed_sdpa_jittable(jit_config):
    """Test that MixedScoresSDPA is JIT-scriptable."""
    config = jit_config
    sdpa = MixedScoresSDPA(config.embedding_dim, config.n_heads, False)
    sdpa = torch.jit.script(sdpa)
    print(sdpa.code)


def test_rf_mixed_mha_is_jittable(jit_config):
    """Test that MixedScoresMHA is JIT-scriptable."""
    config = jit_config
    mha = MixedScoresMHA(config.embedding_dim, config.n_heads, False)
    mha = torch.jit.script(mha)
    print(mha.code)


def test_rf_mlp_is_jittable(jit_config):
    """Test that MLP is JIT-scriptable."""
    config = jit_config
    mlp = MLP(input_dim=config.embedding_dim, output_dim=config.embedding_dim, num_neurons=[config.embedding_dim])
    mlp = torch.jit.script(mlp)
    print(mlp.code)


@pytest.mark.parametrize("norm", ['layer', 'batch', 'rms', 'scale', 'identity', 'instance'])
def test_rf_normalization_is_jittable(jit_config, norm):
    """Test that Normalization layers are JIT-scriptable."""
    config = jit_config
    normalizer = Normalization(config.embedding_dim, norm)
    normalizer = torch.jit.script(normalizer)
    print(normalizer.code)


def test_rf_transformer_block_is_jittable(jit_config):
    """Test that TransformerBlock is JIT-scriptable."""
    config = jit_config
    block = TransformerBlock()
    block = torch.jit.script(block)
    print(block.code)


def test_mixed_transformer_block_is_jittable(jit_config):
    """Test that MixedTransformerBlock is JIT-scriptable."""
    config = jit_config
    from mavrp.env.encoders.matnet import MixedTransformerBlock
    block = MixedTransformerBlock()
    block = torch.jit.script(block)
    print(block.code)


def test_rf_gnn_transformer_block_is_jittable(jit_config):
    """Test that SageTransformerBlock is JIT-scriptable."""
    config = jit_config
    block = SageTransformerBlock()
    block = torch.jit.script(block)
    print(block.code)


def test_initial_embedding_is_jittable(jit_config):
    """Test that InitialEmbeddingLayer is JIT-scriptable."""
    config = jit_config
    init_embedding = InitialEmbeddingLayer(config)
    init_embedding = torch.jit.script(init_embedding)
    print(init_embedding.code)


def test_graph_embedding_is_jittable(jit_config):
    """Test that GraphEmbeddingLayer is JIT-scriptable."""
    config = jit_config
    graph_embedding = GraphEmbeddingLayer(config)
    graph_embedding = torch.jit.script(graph_embedding)
    print(graph_embedding.code)


def test_attention_encoder_jittable(jit_config):
    """Test that AttentionEncoder is JIT-scriptable."""
    config = jit_config
    encoder = AttentionEncoder(config)
    encoder = torch.jit.script(encoder)
    print(encoder.code)


@pytest.mark.skip(reason="PerformerEncoder is not jittable")
def test_performer_encoder_jittable(jit_config):
    """Test that PerformerEncoder is JIT-scriptable (disabled - not jittable)."""
    config = jit_config
    encoder = PerformerEncoder(config)
    encoder = torch.jit.script(encoder)
    print(encoder.code)


def test_gnn_encoder_jittable(jit_config):
    """Test that SageEncoder is JIT-scriptable."""
    config = jit_config
    encoder = SageEncoder(config)
    encoder = torch.jit.script(encoder)
    print(encoder.code)


@pytest.mark.skip(reason="GPSEncoder is not jittable")
def test_gps_encoder_jittable(jit_config):
    """Test that GPSEncoder is JIT-scriptable (disabled - not jittable)."""
    config = jit_config
    encoder = GPSEncoder(config)
    encoder = torch.jit.script(encoder)
    print(encoder.code)


@pytest.mark.parametrize("encoder_class", [GATEncoder, GATv2Encoder, MixedScoresEncoder])
def test_graph_encoders_jittable(jit_config, encoder_class):
    """Test that graph encoders are JIT-scriptable."""
    config = jit_config
    encoder = encoder_class(config)
    encoder = torch.jit.script(encoder)
    print(encoder.code)


def test_rf_pointer_jittable(jit_config):
    """Test that PointerAttention is JIT-scriptable."""
    config = jit_config
    pointer = PointerAttention(config)
    pointer = torch.jit.script(pointer)
    print(pointer.code)


def test_rf_decoder_jittable(jit_config):
    """Test that EndToEndDecoder is JIT-scriptable."""
    config = jit_config
    decoder = EndToEndDecoder(config)
    decoder = torch.jit.script(decoder)
    print(decoder.code)


@pytest.mark.skip(reason="MultiStartDecoder not jittable (uses super())")
def test_msdecoder_jittable(jit_config):
    """Test that MultiStartDecoder is JIT-scriptable (disabled - uses super())."""
    config = jit_config
    decoder = MultiStartDecoder(config)
    decoder = torch.jit.script(decoder)
    print(decoder.code)


def test_trainer_jittable(jit_config):
    """Test that ReinforceTrainer can be converted to TorchScript."""
    config = jit_config
    config.encoder = AttentionEncoder
    config.decoder = EndToEndDecoder
    model = ReinforceTrainer(config)
    model = model.to_torchscript()
    print(model.code)


@pytest.mark.parametrize("encoder_class", [AttentionEncoder, SageEncoder])
def test_model_jittable(jit_config, encoder_class):
    """Test that TransformerModel is JIT-scriptable with different encoders."""
    config = jit_config
    config.encoder = encoder_class
    config.decoder = EndToEndDecoder
    model = TransformerModel(config)
    model = torch.jit.script(model)
    print(model.code)

