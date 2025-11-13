from typing import Optional, Tuple

import torch
import torch.nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn as nn
from torch._C._nn import scaled_dot_product_attention

from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.encoders.mlp import MLP
from mavrp.env.encoders.moe import MixtureOfExperts
from mavrp.env.mixins import FreezingMixin, InfoMixin
from mavrp.env.normalization import Normalization


class AttentionEncoder(torch.nn.Module, InfoMixin, FreezingMixin):
    def __init__(self, config):
        super(AttentionEncoder, self).__init__()
        self.config = config
        self.init_embedding = self.config.get_problem().init_embedding(config)

        embed_dim = config.embedding_dim
        num_heads = config.n_heads
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=config.normalization,
                    use_prenorm=config.prenorm,
                    feedforward_hidden=feedforward_hidden,
                    use_moe=config.use_moe,
                    num_experts=config.moe_experts,
                )
            )

        self.post_layers_norm = Normalization(embed_dim, config.normalization)

    def reset_parameters(self):
        pass

    def forward(
            self, data: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h, g = self.init_embedding(data[0], data[1])  # [B, N, H]

        # Process embedding
        h = init_h

        # --- Transformer LAYERS ---
        for i, layer in enumerate(self.layers):
            h = layer(h)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        h = self.post_layers_norm(h)

        # Return latent representation
        return h, g  # [B, N, H]


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            device: str = None,
            dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attention_dropout = attention_dropout

        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        assert (
                head_dim % 8 == 0 and head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.out_rearrange = Rearrange(
            "b h s d -> b s (h d)"
        )
        self.in_rearrange = Rearrange("b s (three h d) -> three b h s d", three=3, h=self.num_heads)

    def forward(self, x, attn_mask: Optional[Tensor] = None) -> Tensor:
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = self.in_rearrange(
            self.Wqkv(x)
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(self.out_rearrange(out))


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = False,
            bias: bool = True,
            use_moe: bool = False,
            num_experts: Optional[int] = None,
    ):
        super(TransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if use_moe:
            ffn = MixtureOfExperts(
                input_dim=embed_dim,
                hidden_dim=feedforward_hidden,
                output_dim=embed_dim,
                num_experts=num_experts or 1,
            )
        else:
            ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        self.norm_attn = Normalization(embed_dim, normalization)

        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias
        )
        self.norm_ffn = Normalization(embed_dim, normalization)
        self.ffn = ffn
        self.use_prenorm = use_prenorm

    def forward(self, x, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.attention(self.norm_attn(x), mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.attention(x, mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h
