import importlib

import human_readable
import torch
from rich.console import Console
from rich.table import Table

from mavrp.env.decoders import EndToEndDecoder, RecourseDecoder
from mavrp.env.encoders import (
    AttentionEncoder,
    GATEncoder,
    GATv2Encoder,
    GPSEncoder,
    MixedScoresEncoder,
    PerformerEncoder,
    SageEncoder,
    TransformerEncoder,
)
from mavrp.env.models import TransformerModel


class Config:
    def __init__(self):
        self.problem = "MTVRP"
        self.normalization = "rms"
        self.activation = torch.nn.ReLU
        self.global_attention = "sum"
        self.pre_global_attention = False
        self.merge_global_features_into_depot = True
        self.use_global_in_context = True
        self.instance_features = False
        self.encoder = AttentionEncoder
        self.decoder = EndToEndDecoder
        self.cost_norm = True
        self.embedding_dim = 128
        self.n_layers = 3
        self.n_heads = 8
        self.dropout = 0.1
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        self.hidden_dim = self.embedding_dim * 4
        self.graph_size = 50
        self.batch_size = 2**8
        self.seed = 12341
        self.disable_logger = False
        self.working_dir = "./output"
        self.factor = 1
        self.n_epochs = 301 * self.factor
        self.nb_val_samples = 2**12 // self.factor  # = 4k/factor = 1k
        self.nb_train_samples = 2**17 // self.factor  # = 128k/factor = 32k
        self.nb_test_samples = 2**10  # = 1024
        self.warmup_epochs = 15
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nb_neighbors = 30
        self.sample_neighbors = False
        self.deterministic_train = False
        self.prenorm = True
        self.dist_in_kv = True
        self.bias = True
        self.use_edge_attn = False
        self.use_moe = False
        self.moe_experts = self.n_layers
        self.tune = False
        self.enhanced_scores = False
        self.critic_coef = 0.5
        self.description = ""

    def to_dict(self):
        # Return a dictionary representation of the configuration
        # if an attribute is a class, it is converted to a string
        return {k: v if not isinstance(v, type) else v.__name__ for k, v in vars(self).items()}

    def get_problem(self):
        try:
            return getattr(importlib.import_module("mavrp.env.problems"), self.problem)()
        except (ModuleNotFoundError, AttributeError):
            raise ImportError(f"Problem '{self.problem}' not found.")

    @staticmethod
    def combinations():
        return [
            dict(
                problem=["MTVRP"],
                graph_size=[50],
                encoder=[SageEncoder, AttentionEncoder],
                decoder=[EndToEndDecoder, RecourseDecoder],
            )
        ]

    def __repr__(self):
        all_keys = set()
        for combo in self.combinations():
            all_keys.update(combo.keys())

        acronyms = {key: "".join(word[0] for word in key.split("_")) for key in all_keys}

        return self.description + ",".join(
            f"{acronyms[k]}={v}" if not isinstance(v, type) else f"{acronyms[k]}={v.__name__}"
            for k, v in vars(self).items()
            if k in all_keys and v is not None
        )

    @staticmethod
    def all():
        all_configs = []
        from itertools import product

        for group in Config.combinations():
            keys, values = zip(*group.items())
            for combination in product(*values):
                config = dict(zip(keys, combination))
                _config = Config()
                for k, v in config.items():
                    setattr(_config, k, v)
                all_configs.append(_config)
        return all_configs

    @property
    def checkpoint(self):
        import os

        main_folder = self.working_dir
        folder = os.path.join(main_folder, self.problem, str(self.graph_size), repr(self))

        return os.path.join(folder, "checkpoint.ckpt")

    @staticmethod
    def print_table():
        configs = Config.all()
        table = Table(title="Sizes")
        table.add_column("Config", style="red", no_wrap=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Number of params", style="magenta", justify="right")
        table.add_column("Size", style="green", justify="right")
        table.add_column("checkpoint", style="red", justify="right")
        for i, conf in enumerate(configs):
            conf.device = "cpu"
            model = TransformerModel(conf)

            num_params = human_readable.file_size(model.get_num_params())
            model_size = human_readable.file_size(model.get_model_size())
            import os

            checkpoint = os.path.exists(conf.checkpoint)
            table.add_row(str(i), repr(conf), f"{num_params}", f"{model_size}", f"{checkpoint}")
        Console().print(table)


if __name__ == "__main__":
    Config.print_table()
