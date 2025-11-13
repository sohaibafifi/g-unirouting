import pytest
import torch

from mavrp.configs.config import Config
from mavrp.env.decoders import EndToEndDecoder
from mavrp.env.encoders import MixedScoresEncoder
from mavrp.env.train.trainers.actor_critic import ActorCriticTrainer
from mavrp.env.train.trainers.reinforce import ReinforceTrainer


@pytest.fixture
def trainer_config():
    """Fixture to create config for trainer tests."""
    config = Config()
    config.disable_logger = False
    config.nb_train_samples = 16 * 4
    config.nb_val_samples = 16
    config.batch_size = 16
    config.n_epochs = 5
    config.graph_size = 10
    config.warmup_epochs = 2
    return config


@pytest.mark.skip(reason="Test disabled - takes too long")
def test_reinforce(trainer_config):
    """Test REINFORCE trainer."""
    config = trainer_config
    config.encoder = MixedScoresEncoder
    config.decoder = EndToEndDecoder
    trainer = ReinforceTrainer(config)
    trainer = torch.compile(trainer, fullgraph=True, dynamic=True)
    trainer.fit(resume=False)


@pytest.mark.skip(reason="Test disabled - takes too long")
def test_actor_critic(trainer_config):
    """Test Actor-Critic trainer."""
    config = trainer_config
    config.encoder = MixedScoresEncoder
    config.decoder = EndToEndDecoder
    trainer = ActorCriticTrainer(config)
    trainer = torch.compile(trainer, fullgraph=True, dynamic=True)
    trainer.fit(resume=False)

