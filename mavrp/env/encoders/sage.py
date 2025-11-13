from typing import List, Optional, Tuple, Union

import torch
import torch.nn
from torch import Tensor
from torch import nn as nn
from torch.nn import Identity
from torch.nn import functional as F
from torch_geometric.nn import Aggregation, Linear, MessagePassing, MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size

from mavrp.env.encoders.graph import GraphEmbeddingLayer
from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.encoders.mlp import MLP
from mavrp.env.mixins import FreezingMixin, InfoMixin
from mavrp.env.normalization import EdgeNormalization, Normalization


class SageTransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = True,
    ):
        super(SageTransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        ffn = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_neurons=num_neurons,
            hidden_act="ReLU",
        )

        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.gnn = SAGEConvWithEdgeFeatures(in_channels=embed_dim, out_channels=embed_dim, edge_dim=embed_dim)
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.use_prenorm = use_prenorm

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.gnn(self.norm_attn(x), edge_index, edge_attr)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.gnn(x, edge_index, edge_attr))
            h = self.norm_ffn(h + self.ffn(h))
        return h


class SageEncoder(torch.nn.Module, InfoMixin, FreezingMixin):
    def __init__(self, config):
        super(SageEncoder, self).__init__()
        self.config = config
        self.init_embedding = self.config.get_problem().init_embedding(config)
        self.graph_embedding = self.config.get_problem().graph_embedding(config)

        embed_dim = config.embedding_dim
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SageTransformerBlock(
                    embed_dim=embed_dim,
                    normalization=config.normalization,
                    use_prenorm=config.prenorm,
                    feedforward_hidden=feedforward_hidden,
                )
            )

        self.post_layers_norm = EdgeNormalization(embed_dim, config.normalization)

    def reset_parameters(self):
        pass

    def forward(
            self, data: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        node_features, global_features = data
        batch_size, seq_size, _ = node_features.size()
        # Transfer to embedding space
        init_h, g = self.init_embedding(node_features, global_features)  # [B, N, H]
        graph, g = self.graph_embedding(
            node_features, global_features, init_h, g
        )

        # Process embedding
        h = graph.x
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_attr)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h.view(batch_size, seq_size, -1), g  # [B, N, H]


class SAGEConvWithEdgeFeatures(MessagePassing):
    r"""
    Extension of the PyG SAGEConv to handle edge features. We add an extra
    `edge_encoder` layer that transforms edge_attr into a shape that can
    be combined with node embeddings.
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            edge_dim: int,
            aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            eps: float = 1e-7,
            **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.eps = eps
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        if isinstance(in_channels, int):
            node_dim = in_channels
        else:
            node_dim = in_channels[0]

        if edge_dim == node_dim:
            self.edge_encoder = Identity()
        else:
            self.edge_encoder = Linear(edge_dim, node_dim, bias=False)

        self.node_edge_lin = Linear(2 * node_dim, node_dim)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        if hasattr(self.edge_encoder, 'reset_parameters'):
            self.edge_encoder.reset_parameters()
        self.node_edge_lin.reset_parameters()

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: Tensor,
            size: Size = None,
    ) -> Tensor:
        r"""
        Forward pass that additionally takes edge_attr.

        """
        if isinstance(x, Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        # If SAGEConv's `project=True`, apply the linear transform to x_src.
        if self.project and hasattr(self, 'lin'):
            x_src = self.lin(x_src).relu()

        # We pass edge_attr to `propagate` as a named argument, e.g. edge_attr=edge_attr.
        # Then inside `message(...)`, we can use it via the same name
        out = self.propagate(
            edge_index,
            x=x_src,
            edge_attr=edge_attr,
            size=size,
        )

        # Then apply the `lin_l` transform from the parent class:
        out = self.lin_l(out)

        # Add the root node contribution if root_weight is True:
        x_r = x_dst
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        # Optionally normalize the output:
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if hasattr(self, 'edge_encoder'):
            edge_attr = self.edge_encoder(edge_attr)

        assert x_j.size(-1) == edge_attr.size(-1)

        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu() + self.eps  # TODO: check if relu is not also applied after

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'edge_encoder_dim={self.edge_encoder.in_features if hasattr(self.edge_encoder, "in_features") else 1}, )')
