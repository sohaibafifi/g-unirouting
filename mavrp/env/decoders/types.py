"""Decode state, cache and step-result dataclasses shared by decoders and XAI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class DecodeCache:
    node_embeddings: torch.Tensor
    global_embeddings: torch.Tensor
    q_global: Union[torch.Tensor, int]
    attn_matrix: Optional[torch.Tensor]


@dataclass
class DecodeState:
    current_node: torch.Tensor       # [B]
    not_served: torch.Tensor         # [B, N]
    leave_time: torch.Tensor         # [B, 1]
    deliveries: torch.Tensor         # [B, 1]
    pickups: torch.Tensor            # [B, 1]
    distance: torch.Tensor           # [B, 1]
    total_distance: torch.Tensor     # [B, 1]
    is_depot: torch.Tensor           # [B]
    hidden: Optional[torch.Tensor] = None  # [1, B, E]; None for EndToEndDecoder


@dataclass
class StepResult:
    logits: torch.Tensor              # [B, N]  raw logits (-inf where masked)
    policy_mask: torch.Tensor         # [B, N]  True = excluded from policy
    full_mask: torch.Tensor           # [B, N]  True = constraint-infeasible
    logprobs: torch.Tensor            # [B, N]
    potential_distance: torch.Tensor  # [B, N]
    hidden: Optional[torch.Tensor] = None           # updated GRU hidden (RecourseDecoder only)
    recourse_triggered: Optional[torch.Tensor] = None  # [B] bool, set by step_update
