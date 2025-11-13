# mtvrp/env/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class MixtureOfExperts(nn.Module):
    """
    A simple continuous Mixture-of-Experts layer.
    For each input vector, we compute:
      gate_logits = gate_proj(x)           # (…, num_experts)
      gate_weights = softmax(gate_logits)
      expert_outputs = [E_i(x) for i in experts]
      output = sum_i gate_weights_i * expert_outputs_i
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        gate_dropout: float = 0.0,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts

        # gating network: projects input → num_experts logits
        self.gate_proj = nn.Linear(input_dim, num_experts)

        # optional dropout on gates
        self.gate_dropout = nn.Dropout(gate_dropout)

        # create experts: each is a 2-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(expert_dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])

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
        # compute gating weights
        # flatten all leading dims for efficiency, then restore
        orig_shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.size(-1))             # [B, input_dim]
        logits = self.gate_proj(flat_x)                # [B, num_experts]
        weights = F.softmax(logits, dim=-1)            # [B, num_experts]
        weights = self.gate_dropout(weights)

        # compute each expert’s outputs
        expert_outs = []
        for expert in self.experts:
            y = expert(flat_x)                          # [B, output_dim]
            expert_outs.append(y.unsqueeze(1))         # [B, 1, output_dim]
        # stack → [B, num_experts, output_dim]
        expert_stack = torch.cat(expert_outs, dim=1)

        # weighted sum: batch‐matrix multiply [B,1,num_experts] × [B,num_experts,output_dim]
        combined = torch.bmm(weights.unsqueeze(1), expert_stack)
        combined = combined.squeeze(1)                 # [B, output_dim]

        # reshape back to (..., output_dim)
        combined =  combined.view(*orig_shape, combined.size(-1))
        # apply shared MLP
        shared_out = self.shared_expert(combined)
        return shared_out + combined