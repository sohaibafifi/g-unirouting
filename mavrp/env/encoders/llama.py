# mtvrp/env/moe.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class MixtureOfExperts(nn.Module):
    """
    A Mixture-of-Experts layer following the Llama 4 MoE design:
    https://github.com/meta-llama/llama-models/blob/main/models/llama4/moe.py
      - Shared base feed-forward network (MLP) applied first to produce a backbone output.
      - Sparse top-k routing: a router linear layer computes logits over experts, selecting the top_k per input.
      - Gating via sigmoid on the selected top_k logits to get expert weights.
      - Experts implemented with fused SwiGLU activations using parameters w1, w3, and w2:
          expert_i(x) = (SiLU(x @ w1_i) * (x @ w3_i)) @ w2_i
      - Final output = shared_FFN(x) + sum_{i=1..top_k} gate_i * expert_i(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        gate_dropout: float = 0.0,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts)
        # fused SwiGLU expert weights
        self.w1 = nn.Parameter(torch.empty(num_experts, input_dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, input_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, output_dim))
        # initialize expert weights
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.w2)
        # store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # shared final MLP applied to the combined MoE output
        self.shared_expert = MLP(
            input_dim=output_dim,
            output_dim=output_dim,
            num_neurons=[hidden_dim],
            hidden_act="ReLU",
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (..., input_dim)
        returns: Tensor of shape (..., output_dim)
        """
        # flatten all leading dims
        orig_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.input_dim)  # [B, input_dim]
        # shared FFN first
        shared = self.shared_expert(flat_x)     # [B, output_dim]
        # routing
        logits = self.router(flat_x)            # [B, num_experts]
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # [B, top_k]
        gates = torch.sigmoid(topk_logits)      # [B, top_k]
        # MoE computation
        moe_out = flat_x.new_zeros(flat_x.size(0), self.output_dim)
        for i in range(self.top_k):
            idx = topk_idx[:, i]   # [B]
            gate = gates[:, i].unsqueeze(1)
            w1_i = self.w1[idx]    # [B, input_dim, hidden_dim]
            w3_i = self.w3[idx]
            w2_i = self.w2[idx]    # [B, hidden_dim, output_dim]
            # fused SwiGLU
            h1 = torch.bmm(flat_x.unsqueeze(1), w1_i).squeeze(1)
            h3 = torch.bmm(flat_x.unsqueeze(1), w3_i).squeeze(1)
            act = F.silu(h1) * h3
            y = torch.bmm(act.unsqueeze(1), w2_i).squeeze(1)
            moe_out += gate * y
        # combine shared and MoE outputs
        combined = shared + moe_out             # [B, output_dim]
        # reshape back to original dims
        return combined.view(*orig_shape, self.output_dim)