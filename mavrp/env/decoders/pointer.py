import math
from typing import Optional

import torch
import torch.nn
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


class PointerAttention(nn.Module):
    def __init__(
            self, config
    ):
        super(PointerAttention, self).__init__()
        self.config = config
        self.num_heads = config.n_heads
        self.attenton_dropout = config.dropout
        embed_dim = config.embedding_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        # Buffers used to store pre‑computed tensors filled by `precalculate`
        self.register_buffer("glimpse_key_fixed", torch.empty(0), persistent=False)
        self.register_buffer("glimpse_val_fixed", torch.empty(0), persistent=False)
        self.register_buffer("logit_key_fixed", torch.empty(0), persistent=False)
        self.register_buffer("graph_context", torch.empty(0), persistent=False)

        if self.config.dist_in_kv:
            self.W_dist = nn.Linear(1, embed_dim, bias=False)
        else:
            self.W_dist = None

        self.in_rearrange = Rearrange("... g (h s) -> ... h g s", h=self.num_heads)
        self.out_rearrange = Rearrange("... h n g -> ... n (h g)", h=self.num_heads)

    def reset_parameters(self):
        if self.config.dist_in_kv:
            self.W_dist.reset_parameters()
        self.project_node_embeddings.reset_parameters()
        self.project_fixed_context.reset_parameters()
        self.project_out.reset_parameters()

    def precalculate(
            self, embeddings: torch.Tensor
    ):
        (
            self.glimpse_key_fixed,
            self.glimpse_val_fixed,
            self.logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)
        self.logit_key_fixed = self.logit_key_fixed.squeeze(-2).transpose(-2, -1)
        self.graph_context = self.project_fixed_context(embeddings.mean(1))

    def forward(self, h_c: Tensor, mask: Tensor, dist: Optional[Tensor] = None) -> Tensor:

        key = self.glimpse_key_fixed
        value = self.glimpse_val_fixed
        logit_key = self.logit_key_fixed

        # Dynamic distance term
        if self.W_dist is not None and dist is not None:
            delta_kv = self.W_dist(dist)
            key = key + delta_kv  # [B, S, E]
            value = value + delta_kv

        q = self.in_rearrange(h_c)
        k = self.in_rearrange(key)
        v = self.in_rearrange(value)

        attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
        heads = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attenton_dropout)
        heads = self.out_rearrange(heads)
        glimpse = self.project_out(heads)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key)).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        assert not torch.isnan(logits).any(), "Logits contain NaNs"
        logits = torch.tanh(logits) * 10

        return logits
