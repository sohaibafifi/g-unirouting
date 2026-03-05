from typing import Dict, Optional

import torch
import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.base import DecoderBase
from mavrp.env.decoders.pointer import PointerAttention
from mavrp.env.decoders.types import DecodeCache, DecodeState, StepResult
from mavrp.env.mixins import FreezingMixin


class RecourseDecoder(DecoderBase, FreezingMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.problem = config.get_problem()

        self.dynamic_context_embedding = torch.nn.Linear(
            config.embedding_dim + 7, config.embedding_dim, bias=False
        )

        self.history = torch.nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        if self.config.use_global_in_context:
            self.q_global = torch.nn.Linear(
                config.embedding_dim, config.embedding_dim, bias=False
            )
        else:
            self.q_global = None

        self.glimpse = PointerAttention(config)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        self.edge_distance_scale = (
            torch.nn.Parameter(torch.tensor(1.0, device=self.config.device))
            if self.config.bias
            else None
        )
        self.edge_attn_scale = (
            torch.nn.Parameter(torch.tensor(1.0, device=self.config.device))
            if self.config.use_edge_attn
            else None
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.dynamic_context_embedding.weight)
        if self.config.use_global_in_context:
            torch.nn.init.xavier_uniform_(self.q_global.weight)
        self.glimpse.reset_parameters()
        self.history.reset_parameters()


    def initial_hidden(self, cache: DecodeCache) -> Optional[torch.Tensor]:
        """Return zero GRU hidden state shaped [1, B, E]."""
        batch_size = cache.node_embeddings.size(0)
        embedding_dim = cache.node_embeddings.size(2)
        return cache.node_embeddings.new_zeros(1, batch_size, embedding_dim)

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

        # RecourseDecoder: policy_mask = visited-only (full constraint violations allowed)
        policy_mask, full_mask, potential_distance = compute_action_masks(
            common, state, recourse_enabled=True
        )

        last = cache.node_embeddings[batch_indices, current].unsqueeze(1)  # [B, 1, E]
        last, new_hidden = self.history(last, state.hidden)                 # GRU step

        remaining_distance = torch.nan_to_num(
            common["distance_limits"] - state.distance, posinf=10.0
        )  # [B, 1]
        remaining_demand = common["capacities"] - state.deliveries          # [B, 1]
        remaining_demand_b = common["capacities"] - state.pickups           # [B, 1]
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

        dynamic_context = self.dynamic_context_embedding(data)          # [B, E]
        h_c = cache.q_global.unsqueeze(1) + dynamic_context.unsqueeze(1)  # [B, 1, E]

        dist_feat = common["deltas"][batch_indices, current].unsqueeze(-1)  # [B, N, 1]
        logits = self.glimpse(h_c, policy_mask, dist=dist_feat)

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
            hidden=new_hidden,
        )

    def step_update(
        self,
        common: Dict[str, torch.Tensor],
        state: DecodeState,
        selected_node: torch.Tensor,
        step_result: StepResult,
    ) -> DecodeState:
        from mavrp.env.decoders.ops import estimate_recourse_trip_cost

        batch_size = selected_node.size(0)
        device = selected_node.device
        batch_indices = torch.arange(batch_size, device=device)

        # Detect recourse: selected node is not the depot AND is constraint-infeasible
        selected_is_depot = (selected_node == 0)
        selected_is_feasible = ~step_result.full_mask[batch_indices, selected_node]
        recourse_triggered = ~selected_is_depot & ~selected_is_feasible
        is_normal = ~recourse_triggered

        # Mark selected node as visited regardless of violation
        not_served = state.not_served.clone()
        not_served[batch_indices, selected_node] = False

        # GRU hidden from this step (already updated in step_logits)
        hidden = step_result.hidden

        # Copy mutable state tensors
        deliveries = state.deliveries.clone()
        pickups = state.pickups.clone()
        distance = state.distance.clone()
        leave_time = state.leave_time.clone()
        total_distance = state.total_distance.clone()

        # Recourse: add depot-out-and-back penalty, keep route state unchanged
        if recourse_triggered.any():
            detour = estimate_recourse_trip_cost(common, selected_node)  # [B]
            total_distance[recourse_triggered] = (
                total_distance[recourse_triggered] + detour[recourse_triggered].unsqueeze(1)
            )

        # Normal non-depot: update route accumulators and advance current node
        is_normal_nondepot = is_normal & ~selected_is_depot
        if is_normal_nondepot.any():
            nb = batch_indices[is_normal_nondepot]
            sn = selected_node[is_normal_nondepot]
            travel = common["deltas"][nb, state.current_node[is_normal_nondepot], sn]
            arrival = state.leave_time[is_normal_nondepot].squeeze(1) + travel
            start_time = torch.maximum(arrival, common["earliest_start_time"][nb, sn])

            deliveries[is_normal_nondepot] = deliveries[is_normal_nondepot] + common["demands"][nb, sn].unsqueeze(1)
            pickups[is_normal_nondepot] = pickups[is_normal_nondepot] + common["demands_b"][nb, sn].unsqueeze(1)
            distance[is_normal_nondepot] = state.distance[is_normal_nondepot] + travel.unsqueeze(1)
            leave_time[is_normal_nondepot] = (start_time + common["services"][nb, sn]).unsqueeze(1)

        # End-route (going to depot normally): accumulate total_distance and reset
        end_route = is_normal & selected_is_depot
        if end_route.any():
            eb = batch_indices[end_route]
            travel_to_depot = common["deltas"][eb, state.current_node[end_route], 0]
            route_dist = state.distance[end_route] + travel_to_depot.unsqueeze(1)
            total_distance[end_route] = total_distance[end_route] + route_dist
            deliveries[end_route] = 0
            pickups[end_route] = 0
            distance[end_route] = 0
            leave_time[end_route] = common["earliest_start_time"][end_route, 0].unsqueeze(1)

            # Reset GRU hidden for completed routes
            if hidden is not None:
                new_h = hidden.clone()
                new_h[:, end_route, :] = 0.0
                hidden = new_h

        # Determine new current_node:
        #   recourse → stay at current; normal → advance to selected
        current_node = state.current_node.clone()
        current_node[is_normal] = selected_node[is_normal]

        # Keep depot "alive" in not_served for any batch not currently at depot
        not_served[current_node != 0, 0] = True

        return DecodeState(
            current_node=current_node,
            not_served=not_served,
            leave_time=leave_time,
            deliveries=deliveries,
            pickups=pickups,
            distance=distance,
            total_distance=total_distance,
            is_depot=(current_node == 0),
            hidden=hidden,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
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
        batch_indices = torch.arange(batch_size, device=device)

        # Build dense edge-attention matrix (optional)
        attn_matrix = None
        if edge_index is not None and edge_attn_scores is not None:
            assert self.config.use_edge_attn, "Edge attention scores provided but not used."
            edge_scores = (
                edge_attn_scores.mean(-1) if edge_attn_scores.dim() > 1 else edge_attn_scores
            )
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
            attn_matrix = torch_geometric.utils.to_dense_adj(
                edge_index=edge_index, edge_attr=edge_scores, batch=batch_vec,
            )

        # Pre-compute problem tensors and build encoder cache
        common = build_common(inputs[0], inputs[1])
        q_g = (
            self.q_global(global_embeddings)
            if self.q_global is not None
            else torch.zeros_like(global_embeddings)
        )
        cache = DecodeCache(
            node_embeddings=node_embeddings,
            global_embeddings=global_embeddings,
            q_global=q_g,
            attn_matrix=attn_matrix,
        )
        self.glimpse.precalculate(node_embeddings)

        # Initialize decode state (current_node=0, not_served[:,0]=False, hidden=zeros)
        state = self.init_decode_state(common, cache)

        if actions is not None and actions.size(1) > 0:
            solution = actions[:, :1].clone()
        else:
            solution = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=device)
        violated_nodes_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        while state.not_served.any():
            step_result = self.step_logits(cache, common, state)

            if actions is None or actions.size(1) <= solution.size(1):
                probas = torch.softmax(step_result.logits, dim=-1)
                if decode_mode == "greedy":
                    _, selected_node = self.greedy_decoding(probas)
                else:
                    _, selected_node = self.sample_decoding(probas)
            else:
                selected_node = actions[:, solution.size(1)]

            selected_node = selected_node.long()

            # Track violated nodes for repair_solution
            selected_is_depot = (selected_node == 0)
            selected_is_feasible = ~step_result.full_mask[batch_indices, selected_node]
            recourse = ~selected_is_depot & ~selected_is_feasible
            if recourse.any():
                violated_nodes_mask[batch_indices[recourse], selected_node[recourse]] = True

            log_probabilities = log_probabilities + self.cross_entropy(step_result.logits, selected_node)
            solution = torch.cat([solution, selected_node.unsqueeze(1)], dim=1)
            state = self.step_update(common, state, selected_node, step_result)

        return -log_probabilities, solution, state.total_distance.squeeze(1)

    def repair_solution(self, solution, violated_nodes_mask, deltas):
        batch_size = solution.size(0)
        device = solution.device
        new_solutions = []

        for i in range(batch_size):
            sol = solution[i].tolist()
            violated_nodes = [
                idx
                for idx in range(violated_nodes_mask.size(1))
                if violated_nodes_mask[i, idx] and idx in sol
            ]

            new_sol = [node for node in sol if node not in violated_nodes]

            for v_node in violated_nodes:
                new_sol.append(0)
                new_sol.append(v_node)

            new_sol.append(0)
            new_solutions.append(
                torch.tensor(new_sol, dtype=torch.long, device=device)
            )

        new_solutions = torch.nn.utils.rnn.pad_sequence(
            new_solutions,
            batch_first=True,
            padding_value=0,
        )

        batch_indices = torch.arange(
            new_solutions.size(0),
            device=device,
        ).unsqueeze(1)

        from_nodes = new_solutions[:, :-1]
        to_nodes = new_solutions[:, 1:]
        new_costs = deltas[batch_indices, from_nodes, to_nodes].sum(dim=1)

        return new_solutions, new_costs
