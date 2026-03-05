from typing import Dict, Optional

import torch
import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.base import DecoderBase
from mavrp.env.decoders.pointer import PointerAttention
from mavrp.env.decoders.types import DecodeCache, DecodeState, StepResult
from mavrp.env.mixins import FreezingMixin


class EndToEndDecoder(DecoderBase, FreezingMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.problem = config.get_problem()

        self.dynamic_context_embedding = torch.nn.Linear(config.embedding_dim + 7, config.embedding_dim, bias=False)

        if self.config.use_global_in_context:
            self.q_global = torch.nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        else:
            self.q_global = None

        self.glimpse = PointerAttention(config)

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        if self.config.bias:
            self.edge_distance_scale = torch.nn.Parameter(torch.tensor(1.0, device=self.config.device))
        else:
            self.edge_distance_scale = None

        if self.config.use_edge_attn:
            self.edge_attn_scale = torch.nn.Parameter(torch.tensor(1.0, device=self.config.device))
        else:
            self.edge_attn_scale = None

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.dynamic_context_embedding.weight)
        if self.config.use_global_in_context:
            torch.nn.init.xavier_uniform_(self.q_global.weight)
        self.glimpse.reset_parameters()

    # ------------------------------------------------------------------
    # Step-level API
    # ------------------------------------------------------------------

    def step_logits(
        self,
        cache: DecodeCache,
        common: Dict[str, torch.Tensor],
        state: DecodeState,
    ) -> StepResult:
        from mavrp.env.decoders.ops import compute_action_masks

        batch_size = state.current_node.size(0)
        device = state.current_node.device
        batch_indices = torch.arange(batch_size, device=device)
        current = state.current_node

        policy_mask, full_mask, potential_distance = compute_action_masks(
            common, state, recourse_enabled=False
        )

        last = cache.node_embeddings[batch_indices, current].unsqueeze(1)  # [B, 1, E]

        remaining_distance = torch.nan_to_num(
            common["distance_limits"] - state.distance, posinf=10.0
        )  # [B, 1]
        remaining_demand = common["capacities"] - state.deliveries        # [B, 1]
        remaining_demand_b = common["capacities"] - state.pickups         # [B, 1]
        distance_to_depot = torch.nan_to_num(
            common["deltas"][batch_indices, current, 0], posinf=1e9
        ).unsqueeze(-1)  # [B, 1]

        data = torch.cat((
            last.squeeze(1),                              # [B, E]
            state.leave_time,                             # [B, 1]
            remaining_distance,                           # [B, 1]
            remaining_demand,                             # [B, 1]
            remaining_demand_b,                           # [B, 1]
            distance_to_depot,                            # [B, 1]
            common["open_routes"].unsqueeze(-1),          # [B, 1]
            common["mixed_backhauls"].unsqueeze(-1),      # [B, 1]
        ), dim=-1)

        dynamic_context = self.dynamic_context_embedding(data)   # [B, E]
        h_c = cache.q_global + dynamic_context.unsqueeze(1)      # [B, 1, E]

        dist_feat = common["deltas"][batch_indices, current].unsqueeze(-1)  # [B, N, 1]
        logits = self.glimpse(h_c, policy_mask, dist=dist_feat)  # [B, N]

        if self.edge_distance_scale is not None:
            logits = logits - self.edge_distance_scale * common["deltas"][batch_indices, current]

        if cache.attn_matrix is not None and self.edge_attn_scale is not None:
            logits = logits + self.edge_attn_scale * cache.attn_matrix[batch_indices, current]

        logits = logits.masked_fill(policy_mask, float("-inf"))
        logprobs = torch.log_softmax(logits, dim=-1)

        return StepResult(
            logits=logits,
            policy_mask=policy_mask,
            full_mask=full_mask,
            logprobs=logprobs,
            potential_distance=potential_distance,
        )

    def step_update(
        self,
        common: Dict[str, torch.Tensor],
        state: DecodeState,
        selected_node: torch.Tensor,
        step_result: StepResult,
    ) -> DecodeState:
        batch_size = selected_node.size(0)
        device = selected_node.device
        batch_indices = torch.arange(batch_size, device=device)

        # Mark selected as visited
        not_served = state.not_served.clone()
        not_served[batch_indices, selected_node] = False

        # Update accumulated demands
        deliveries = state.deliveries + common["demands"][batch_indices, selected_node].unsqueeze(1)
        pickups = state.pickups + common["demands_b"][batch_indices, selected_node].unsqueeze(1)

        # Update route distance (gather from pre-computed potential_distance)
        distance = step_result.potential_distance.gather(1, selected_node.unsqueeze(1))  # [B, 1]

        # Update leave_time
        arrival = state.leave_time.squeeze(1) + common["deltas"][batch_indices, state.current_node, selected_node]
        selected_earliest = common["earliest_start_time"][batch_indices, selected_node]
        start_time = torch.maximum(arrival, selected_earliest)
        service = common["services"][batch_indices, selected_node]
        leave_time = (start_time + service).unsqueeze(1)  # [B, 1]

        is_depot = (selected_node == 0)

        # Non-depot selections: keep depot alive in not_served so the loop continues
        not_served[~is_depot, 0] = True

        # Depot selection: accumulate total_distance and reset route state
        total_distance = state.total_distance.clone()
        return_dist = common["deltas"][batch_indices, selected_node, 0].unsqueeze(1)  # [B, 1]; = 0 when selected=depot
        total_distance[is_depot] = total_distance[is_depot] + distance[is_depot] + return_dist[is_depot]
        deliveries[is_depot] = 0
        pickups[is_depot] = 0
        distance[is_depot] = 0
        leave_time[is_depot] = common["earliest_start_time"][is_depot, 0].unsqueeze(1)

        return DecodeState(
            current_node=selected_node,
            not_served=not_served,
            leave_time=leave_time,
            deliveries=deliveries,
            pickups=pickups,
            distance=distance,
            total_distance=total_distance,
            is_depot=is_depot,
            hidden=None,
        )

    # ------------------------------------------------------------------

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor],
                node_embeddings: torch.Tensor,
                global_embeddings: torch.Tensor,
                decode_mode: str = "sample",
                actions: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                edge_attn_scores: Optional[torch.Tensor] = None,
                ):
        from mavrp.env.decoders.ops import build_common

        batch_size, seq_len, _ = node_embeddings.size()
        device = node_embeddings.device

        # Build dense edge-attention matrix (optional)
        attn_matrix = None
        if self.edge_attn_scale is not None:
            # Note: TorchScript doesn't support assert with messages
            # We assume edge_index and edge_attn_scores are provided correctly
            # Build dense attention matrix from edge attention scores
            # edge_index: [2, E], edge_attn_scores: [E] or [E, H]
            # average across heads if needed
            edge_scores = edge_attn_scores.mean(-1) if edge_attn_scores.dim() > 1 else edge_attn_scores
            # create a [batch_size, seq_len, seq_len] attention matrix
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
            attn_matrix = torch_geometric.utils.to_dense_adj(
                edge_index=edge_index, edge_attr=edge_scores, batch=batch_vec
            )

        # Pre-compute problem tensors and build encoder cache
        common = build_common(inputs[0], inputs[1])
        h_hat = global_embeddings.unsqueeze(1)
        q_g = self.q_global(h_hat) if self.q_global is not None else 0
        cache = DecodeCache(
            node_embeddings=node_embeddings,
            global_embeddings=global_embeddings,
            q_global=q_g,
            attn_matrix=attn_matrix,
        )
        self.glimpse.precalculate(node_embeddings)

        # Initialize decode state (current_node=0, not_served[:,0]=False)
        state = self.init_decode_state(common, cache)

        if actions is None or actions.size(1) < 1:
            solution = torch.zeros([batch_size, 1], dtype=torch.long, device=device)
        else:
            solution = actions[:, :1].clone()

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=device)

        while state.not_served.any():
            step_result = self.step_logits(cache, common, state)

            if actions is None or actions.size(1) < solution.size(1) + 1:
                probas = torch.softmax(step_result.logits, dim=-1)
                if decode_mode == "greedy":
                    _, selected_node = self.greedy_decoding(probas)
                elif decode_mode == "sample":
                    _, selected_node = self.sample_decoding(probas)
                else:
                    raise NotImplementedError(f"Decoding mode {decode_mode} not implemented.")
            else:
                selected_node = actions[:, solution.size(1)]

            log_probabilities = log_probabilities + self.cross_entropy(step_result.logits, selected_node)
            solution = torch.cat([solution, selected_node.unsqueeze(1)], dim=1)
            state = self.step_update(common, state, selected_node, step_result)

        assert solution.shape[1] >= seq_len
        return -log_probabilities, solution, state.total_distance.squeeze(1)
