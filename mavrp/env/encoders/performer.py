from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.attention import PerformerAttention

from mavrp.env.encoders.attention import TransformerBlock
from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.mixins import FreezingMixin, InfoMixin
from mavrp.env.normalization import Normalization


class PerformerBlock(TransformerBlock):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = False,
            bias: bool = True
    ):
        super(PerformerBlock, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=use_prenorm,
            bias=bias
        )
        self.attention = PerformerAttention(
            embed_dim, num_heads, dropout=0.1
        )
        self.redraw_projection = RedrawProjection(
            self.attention,
            redraw_interval=1000)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        assert mask is None, "Masking not supported in PerformerBlock"
        self.redraw_projection.redraw_projections()
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.attention(self.norm_attn(x))
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.attention(x))
            h = self.norm_ffn(h + self.ffn(h))
        return h


class PerformerEncoder(torch.nn.Module, InfoMixin, FreezingMixin):
    def __init__(self, config):
        super(PerformerEncoder, self).__init__()
        self.config = config
        self.init_embedding = self.config.get_problem().init_embedding(config)

        embed_dim = config.embedding_dim
        num_heads = config.n_heads
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers

        self.layers = torch.nn.Sequential(
            *(
                PerformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=config.normalization,
                    use_prenorm=self.config.prenorm,
                    feedforward_hidden=feedforward_hidden,
                )
                for _ in range(num_layers)
            )
        )

        self.post_layers_norm = Normalization(embed_dim, config.normalization)

    def reset_parameters(self):
        pass

    def forward(
            self, data
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h, g = self.init_embedding(*data)  # [B, N, H]

        # Process embedding
        h = init_h

        # --- Transformer LAYERS ---
        for layer in self.layers:
            h = layer(h)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, g  # [B, N, H]


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = []
            if isinstance(self.model, PerformerAttention):
                fast_attentions = [self.model]
            elif isinstance(self.model, torch.nn.ModuleList):
                fast_attentions = [
                    module for module in self.model.modules()
                    if isinstance(module, PerformerAttention)
                ]

            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
