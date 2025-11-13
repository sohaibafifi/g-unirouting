from typing import Optional

import torch.nn
from torch_geometric.nn import GATv2Conv

from mavrp.env.encoders.gat import GATEncoder, GATTransformerBlock
from mavrp.env.encoders.sage import SageEncoder, SageTransformerBlock


class GATv2TransformerBlock(GATTransformerBlock):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = True,
    ):
        super(GATv2TransformerBlock, self).__init__(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=use_prenorm,
        )

        self.gnn = GATv2Conv(in_channels=embed_dim,
                             add_self_loops=False,
                             out_channels=embed_dim // num_heads, heads=num_heads, concat=True, dropout=0.1,
                             edge_dim=embed_dim)




class GATv2Encoder(GATEncoder):
    def __init__(self, config):
        super(GATv2Encoder, self).__init__(config)
        embed_dim = config.embedding_dim
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GATv2TransformerBlock(
                    embed_dim=embed_dim,
                    normalization=config.normalization,
                    use_prenorm=config.prenorm,
                    feedforward_hidden=feedforward_hidden,
                )
            )
