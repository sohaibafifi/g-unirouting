from typing import Optional

import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.endtoend import EndToEndDecoder

from typing import Optional

import torch
import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.base import DecoderBase
from mavrp.env.decoders.pointer import PointerAttention
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
        batch_size, seq_len, embedding_dim = node_embeddings.size()
        device = node_embeddings.device

        attn_matrix = None
        if edge_index is not None and edge_attn_scores is not None:
            assert self.config.use_edge_attn, "Edge attention scores provided but not used."
            edge_scores = (
                edge_attn_scores.mean(-1)
                if edge_attn_scores.dim() > 1
                else edge_attn_scores
            )
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(
                seq_len
            )
            attn_matrix = torch_geometric.utils.to_dense_adj(
                edge_index=edge_index,
                edge_attr=edge_scores,
                batch=batch_vec,
            )

        node_features, global_features = inputs

        locations = node_features[:, :, :2]
        demands = node_features[:, :, 2]
        demands_b = node_features[:, :, 3]
        time_windows = node_features[:, :, 4:6]
        services = node_features[:, :, 6]

        capacities = global_features[:, 0]
        open_routes = global_features[:, 1].to(torch.bool)
        mixed_backhauls = global_features[:, 2].to(torch.bool)
        distance_limits = global_features[:, 3]
        time_limits = global_features[:, 4]

        deltas = torch.cdist(locations, locations)
        deltas[open_routes, :, 0] = 0.0

        backhauls_instances = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
        backhauls_instances = backhauls_instances.unsqueeze(-1).expand_as(demands_b)
        backhauls_mask = (demands_b > 0) & backhauls_instances
        linehauls_mask = (demands > 0) & backhauls_instances
        invalid_arcs = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)

        h_hat = global_embeddings

        if self.q_global is not None:
            q_g = self.q_global(h_hat)
        else:
            q_g = torch.zeros_like(h_hat)


        not_served = torch.ones_like(demands, dtype=torch.bool, device=device)

        if actions is not None and actions.size(1) > 0:
            solution = actions[:, :1].clone()
        else:
            solution = torch.zeros(

                (batch_size, 1),
                dtype=torch.long,
                device=device,
            )

        batch_indices = torch.arange(batch_size, device=device)
        current_node = solution[:, -1].detach().clone()
        not_served[batch_indices, current_node] = False

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=device)
        detour_cost = torch.zeros(batch_size, dtype=torch.float32, device=device)
        total_distance = torch.zeros(batch_size, dtype=torch.float32, device=device)

        earliest_start_time, latest_start_time = time_windows.unbind(-1)
        latest_start_time = torch.where(
            open_routes.unsqueeze(-1),
            torch.full_like(latest_start_time, float("inf")),
            latest_start_time,
        )

        leave_time = earliest_start_time[:, 0].clone()
        deliveries = torch.zeros_like(leave_time)
        pickups = torch.zeros_like(leave_time)
        distance = torch.zeros_like(leave_time)

        open_routes = open_routes.unsqueeze(-1)
        mixed_backhauls = mixed_backhauls.unsqueeze(-1)

        hidden = node_embeddings.new_zeros(1, batch_size, embedding_dim)
        self.glimpse.precalculate(node_embeddings)

        violated_nodes_mask = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )

        while not_served.any():
            mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            mask = mask.masked_fill(~not_served, True)

            is_depot_state = current_node == 0
            mask[:, 0] = is_depot_state & not_served.any(dim=1)

            last = node_embeddings[batch_indices, current_node, :].unsqueeze(1)
            last, hidden = self.history(last, hidden)

            remaining_distance = torch.nan_to_num(
                distance_limits.unsqueeze(1) - distance.unsqueeze(-1),
                posinf=10.0,
            )
            remaining_demand = capacities.unsqueeze(1) - deliveries.unsqueeze(-1)
            remaining_demand_b = capacities.unsqueeze(1) - pickups.unsqueeze(-1)
            distance_to_depot = torch.nan_to_num(
                deltas[batch_indices, current_node, 0],
                posinf=1e9,
            ).unsqueeze(-1)

            data = torch.cat(
                (
                    last.squeeze(1),
                    leave_time.unsqueeze(-1),
                    remaining_distance,
                    remaining_demand,
                    remaining_demand_b,
                    distance_to_depot,
                    open_routes,
                    mixed_backhauls,
                ),
                dim=-1,
            )

            dynamic_context = self.dynamic_context_embedding(data)
            h_c = q_g.unsqueeze(1) + dynamic_context.unsqueeze(1)

            dist_feat = deltas[batch_indices, current_node, :].unsqueeze(-1)
            u = self.glimpse(h_c, mask, dist=dist_feat)

            if self.edge_distance_scale is not None:
                u = u - self.edge_distance_scale * deltas[
                    batch_indices, current_node, :
                ]

            if attn_matrix is not None and self.edge_attn_scale is not None:
                u = u + self.edge_attn_scale * attn_matrix[
                    batch_indices, current_node, :
                ]

            u = u.masked_fill(mask, float("-inf"))

            if actions is None or actions.size(1) <= solution.size(1):
                probas = torch.nn.functional.softmax(u, dim=-1)
                if decode_mode == "greedy":
                    _, selected_node = self.greedy_decoding(probas)
                else:
                    _, selected_node = self.sample_decoding(probas)
            else:
                selected_node = actions[:, solution.size(1)]

            selected_node = selected_node.long()

            travel_to_next = deltas[batch_indices, current_node, selected_node]
            effective_arrival_times = leave_time + travel_to_next
            selected_earliest = earliest_start_time[batch_indices, selected_node]
            effective_start_times = torch.max(
                effective_arrival_times, selected_earliest
            )

            return_to_depot = deltas[batch_indices, selected_node, 0]
            effective_total_time = (
                effective_start_times
                + services[batch_indices, selected_node]
                + return_to_depot
            )

            effective_distance = distance + travel_to_next
            effective_total_dist = effective_distance + return_to_depot

            exceed_latest_start_time = (
                effective_start_times
                > latest_start_time[batch_indices, selected_node]
            )
            exceed_time_limit = effective_total_time > time_limits
            exceed_distance_limit = effective_total_dist > distance_limits
            exceed_capacity = (
                (deliveries + demands[batch_indices, selected_node] > capacities)
                | (pickups + demands_b[batch_indices, selected_node] > capacities)
            )
            cannot_serve_linehaul = (
                mixed_backhauls.squeeze(-1)
                & (demands[batch_indices, selected_node] + pickups > capacities)
            )

            violation_mask = (
                exceed_capacity
                | cannot_serve_linehaul
                | exceed_latest_start_time
                | exceed_time_limit
                | exceed_distance_limit
            )

            step_ce = self.cross_entropy(u, selected_node)
            log_probabilities = log_probabilities + step_ce

            solution = torch.cat(
                (solution, selected_node.unsqueeze(-1)),
                dim=1,
            )

            not_served[batch_indices, selected_node] = False

            is_depot_choice = selected_node == 0
            is_violation = (~is_depot_choice) & violation_mask
            is_normal = ~is_violation

            next_current_node = current_node.clone()

            if is_violation.any():
                vb = batch_indices[is_violation]
                vn = selected_node[is_violation]
                violated_nodes_mask[vb, vn] = True
                detour_cost[is_violation] += (
                    deltas[vb, 0, vn] + deltas[vb, vn, 0]
                )

            if is_normal.any():
                nb = batch_indices[is_normal]
                sn = selected_node[is_normal]

                selected_demands = demands[nb, sn]
                selected_demands_b = demands_b[nb, sn]
                selected_service_time = services[nb, sn]
                selected_start_times = effective_start_times[is_normal]

                deliveries[is_normal] += selected_demands
                pickups[is_normal] += selected_demands_b
                distance[is_normal] = effective_distance[is_normal]
                leave_time[is_normal] = (
                    selected_start_times + selected_service_time
                )
                next_current_node[is_normal] = sn

            end_route = is_normal & (selected_node == 0)

            if end_route.any():
                total_distance[end_route] += distance[end_route]
                deliveries[end_route] = 0.0
                pickups[end_route] = 0.0
                distance[end_route] = 0.0
                leave_time[end_route] = earliest_start_time[end_route, 0]

                new_hidden = hidden.clone()
                new_hidden[:, end_route, :] = 0.0
                hidden = new_hidden

                next_current_node[end_route] = 0

            current_node = next_current_node

            not_served[current_node != 0, 0] = True

        cost = total_distance + detour_cost
        metrics = torch.stack([total_distance, detour_cost], dim=-1)

        return -log_probabilities, solution, cost, metrics

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

