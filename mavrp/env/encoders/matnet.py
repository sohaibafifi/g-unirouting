from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mavrp.env.encoders.attention import MultiHeadAttention, TransformerBlock
from mavrp.env.encoders.init import InitialEmbeddingLayer
from mavrp.env.mixins import FreezingMixin, InfoMixin
from mavrp.env.normalization import Normalization


class MixedScoresSDPA(nn.Module):
    def __init__(
            self,
            num_heads: int,
            num_scores: int = 1,
            mixer_hidden_dim: int = 16,
            mix1_init: float = (1 / 2) ** (1 / 2),
            mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_scores = num_scores
        mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, self.num_scores + 1, mixer_hidden_dim)
        )
        mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, mixer_hidden_dim)
        )
        self.mix_W1 = nn.Parameter(mix_W1)
        self.mix_b1 = nn.Parameter(mix_b1)

        mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, mixer_hidden_dim, 1)
        )
        mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, 1)
        )
        self.mix_W2 = nn.Parameter(mix_W2)
        self.mix_b2 = nn.Parameter(mix_b2)

    def forward(self, q, k, v, attn_mask:Optional[Tensor]=None, dmat:Optional[Tensor]=None, dropout_p:float=0.0) -> Tensor:
        """Scaled Dot-Product Attention with MatNet Scores Mixer"""
        assert dmat is not None
        b, m, n = dmat.shape[:3]
        dmat = dmat.reshape(b, m, n, self.num_scores)

        # Calculate scaled dot product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        # [b, h, m, n, num_scores+1]
        mix_attn_scores = torch.cat(
            [
                attn_scores.unsqueeze(-1),
                dmat[:, None, ...].expand(b, self.num_heads, m, n, self.num_scores),
            ],
            dim=-1,
        )
        # [b, h, m, n]
        attn_scores = (
            (
                    torch.matmul(
                        F.relu(
                            torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
                            + self.mix_b1[None, None, :, None, :]
                        ),
                        self.mix_W2,
                    )
                    + self.mix_b2[None, None, :, None, :]
            )
            .transpose(1, 2)
            .squeeze(-1)
        )

        # Apply the provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask[~attn_mask.any(-1)] = True
                attn_scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)


class MixedScoresMHA(MultiHeadAttention):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_scores: int = 4,
            mixer_hidden_dim: int = 16,
            bias: bool = True,
            attention_dropout: float = 0.0,
            device: str = None,
            dtype: torch.dtype = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, bias, attention_dropout, device, dtype)
        self.mix_scores_layer = MixedScoresSDPA(
            num_heads=num_heads,
            num_scores=num_scores,
            mixer_hidden_dim=mixer_hidden_dim
        )
    @torch.compile
    def forward(self, x, deltas:Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        deltas: (batch, seqlen, seqlen)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
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
        out = self.mix_scores_layer(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dmat=deltas,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(self.out_rearrange(out))


class MixedTransformerBlock(TransformerBlock):
    def __init__(
            self,
            embed_dim: int = 128,
            num_scores: int = 4,
            num_heads: int = 8,
            feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
            normalization: Optional[str] = "instance",
            use_prenorm: bool = False,
            bias: bool = True,
            use_moe: bool = False,
            num_experts: Optional[int] = None,
    ):
        super(MixedTransformerBlock, self).__init__(embed_dim, num_heads, feedforward_hidden, normalization, use_prenorm, bias,
                                                    use_moe, num_experts)

        self.attention = MixedScoresMHA(
            embed_dim, num_heads, num_scores,  bias=bias
        )


    def forward(self, x: Tensor, deltas: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.attention(self.norm_attn(x), deltas=deltas, attn_mask=mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.attention(x, deltas=deltas, attn_mask=mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h


class MixedScoresEncoder(torch.nn.Module, InfoMixin, FreezingMixin):
    def __init__(self, config):
        super(MixedScoresEncoder, self).__init__()
        self.config = config
        self.init_embedding = self.config.get_problem().init_embedding(config)
        self.enhanced_scores = config.enhanced_scores
        embed_dim = config.embedding_dim
        num_heads = config.n_heads
        feedforward_hidden = config.hidden_dim
        num_layers = config.n_layers
        num_scores = 4 if config.enhanced_scores else 1
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MixedTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_scores=num_scores,
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
            self, data
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        node_features, global_features = data[0], data[1]
        init_h, g = self.init_embedding(node_features, global_features)  # [B, N, H]
        deltas = self.get_deltas(global_features, node_features)
        if not self.enhanced_scores:
            deltas = deltas[..., 0:1]

        # Process embedding
        h = init_h

        # --- Transformer LAYERS ---
        for i, layer in enumerate(self.layers):
            h = layer(h, deltas=deltas)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, g  # [B, N, H]

    @staticmethod
    def get_deltas(global_features, node_features):
        # Compute pairwise Euclidean distances
        locations = node_features[..., 0:2]  # [batch, seq_len, 2]
        dist_mat = torch.cdist(locations, locations)  # [batch, seq_len, seq_len]

        # Compute demand differences as additional score
        demands = node_features[..., 2]  # [batch, seq_len]
        demand_diff = torch.abs(demands.unsqueeze(-1) - demands.unsqueeze(1))  # [batch, seq_len, seq_len]

        # Compute window slack feature
        tw_start = node_features[..., 4]  # [batch, seq_len]
        tw_end   = node_features[..., 5]  # [batch, seq_len]
        window_slack = torch.relu(
            torch.min(tw_end.unsqueeze(-1), tw_end.unsqueeze(1))
          - torch.max(tw_start.unsqueeze(-1), tw_start.unsqueeze(1))
        )  # [batch, seq_len, seq_len]

        # Compute orientation angle of each arc
        x = locations[..., 0]  # [batch, seq_len]
        y = locations[..., 1]  # [batch, seq_len]
        dx = x.unsqueeze(-1) - x.unsqueeze(-2)  # [batch, seq_len, seq_len]
        dy = y.unsqueeze(-1) - y.unsqueeze(-2)  # [batch, seq_len, seq_len]
        angle = torch.atan2(dy, dx)  # [batch, seq_len, seq_len]


        # Apply existing masks and penalties on the distance channel
        open_routes = global_features[:, 1].to(torch.bool)
        mixed_backhauls = global_features[:, 2].to(torch.bool)
        demands_b = node_features[:, :, 3]

        # Zero out distances for open routes on first column
        dist_mat[open_routes, :, 0] = 0.0
        # Backhaul/linehaul invalid combinations
        backhauls_inst = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
        backhauls_inst = backhauls_inst.unsqueeze(-1).expand_as(demands_b)
        backhauls_mask = (demands_b > 0) & backhauls_inst
        linehauls_mask = (demands > 0) & backhauls_inst
        invalid_mask = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)
        dist_mat.masked_fill_(invalid_mask, 1e9)

        window_slack = torch.nan_to_num(window_slack, nan=1e9, posinf=1e9, neginf=1e9)
        # Stack features into dmat: distance, demand diff, window slack, angle
        dmat = torch.stack([dist_mat, demand_diff, window_slack, angle],
                           dim=-1)  # [batch, seq_len, seq_len, num_scores]

        return dmat
