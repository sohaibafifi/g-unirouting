from typing import Any, Optional, Tuple

import torch.nn
from torch import Tensor
from torch_geometric.nn import GATConv

from mavrp.env.encoders.sage import SageEncoder, SageTransformerBlock


class GATTransformerBlock(SageTransformerBlock):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = True,
    ):
        super(GATTransformerBlock, self).__init__(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=use_prenorm,
        )

        self.gnn = GATConv(in_channels=embed_dim, out_channels=embed_dim // num_heads, heads=num_heads, concat=True,
                           add_self_loops=False,
                           dropout=0.1)

    def forward(self, x, edge_index, edge_attr=None):

        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            out, (edge_index_out, attn_scores) = self.gnn(self.norm_attn(x), edge_index, edge_attr, return_attention_weights=True)
            h = x + out
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            out, (edge_index_out, attn_scores) = self.gnn(x, edge_index, edge_attr, return_attention_weights=True)

            h = self.norm_attn(x + out)
            h = self.norm_ffn(h + self.ffn(h))
        return h, edge_index_out, attn_scores


class GATEncoder(SageEncoder):
    def __init__(self, config):
        super(GATEncoder, self).__init__(config)
        self.layers = torch.nn.ModuleList()
        embed_dim = config.embedding_dim
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers

        for i in range(num_layers):
            self.layers.append(
                GATTransformerBlock(
                    embed_dim=embed_dim,
                    normalization=config.normalization,
                    use_prenorm=config.prenorm,
                    feedforward_hidden=feedforward_hidden,
                )
            )

    def forward(
            self, data: Tuple[Tensor, Tensor]) \
            -> tuple[Any, Any, Any | None, Any | None]:
        node_features, global_features = data
        batch_size, seq_size, _ = node_features.size()
        # Transfer to embedding space
        init_h, g = self.init_embedding(node_features, global_features)  # [B, N, H]
        graph, g = self.graph_embedding(
            node_features, global_features, init_h, g
        )
        # get scores from the last layer
        attn_scores = None
        attn_edge_index = None

        # Process embedding, optionally capturing attention weights
        h = graph.x
        for layer in self.layers:
            h, edge_index, attn = layer(h, graph.edge_index, graph.edge_attr)
            attn_scores = attn
            attn_edge_index = edge_index

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)
        output = h.view(batch_size, seq_size, -1)
        # Return latent representations, plus attention if requested
        return output, g, attn_edge_index, attn_scores
