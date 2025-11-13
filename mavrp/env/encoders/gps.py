import torch
from torch.nn import (
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.inits import reset

from mavrp.env.encoders.graph import GraphEmbeddingLayer
from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.encoders.performer import RedrawProjection
from mavrp.env.mixins import FreezingMixin


class GPSEncoder(torch.nn.Module, FreezingMixin):
    def __init__(self, config):
        super(GPSEncoder, self).__init__()
        self.config = config
        channels = config.embedding_dim
        self.convs = ModuleList()
        attn_type = 'performer'
        self.initial_embedding = self.config.get_problem().init_embedding(config)
        self.graph_embedding = self.config.get_problem().graph_embedding(config)

        for _ in range(config.n_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=config.n_heads,
                           attn_type=attn_type)
            self.convs.append(conv)
            self.mlp = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.convs)
        reset(self.mlp)
        self.initial_embedding.reset_parameters()
        self.graph_embedding.reset_parameters()

    def forward(self, data):
        self.redraw_projection.redraw_projections()
        batch_size, seq_size, _ = data[0].size()
        node_embeddings, global_embeddings = self.initial_embedding(*data)
        data, global_embeddings = self.graph_embedding(*data, node_embeddings, global_embeddings)
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        x = self.mlp(x)
        return x.view(batch_size, seq_size, -1), global_embeddings
