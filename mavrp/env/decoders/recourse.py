from typing import Optional

import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.endtoend import EndToEndDecoder


class RecourseDecoder(EndToEndDecoder):
    """Decoder that allows infeasible customer selections via depot recourse trips."""

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
        batch_size, seq_len, _ = node_embeddings.size()
        device = node_embeddings.device
        attn_matrix = None
        if self.edge_attn_scale is not None:
            edge_scores = (
                edge_attn_scores.mean(-1)
                if edge_attn_scores.dim() > 1
                else edge_attn_scores
            )
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
            attn_matrix = torch_geometric.utils.to_dense_adj(
                edge_index=edge_index, edge_attr=edge_scores, batch=batch_vec
            )

        node_features, global_features = inputs
        locations = node_features[:, :, :2]
        demands = node_features[:, :, 2]
        demands_b = node_features[:, :, 3]
        time_windows = node_features[:, :, 4:6]
        services = node_features[:, :, 6]
        capacities = global_features[:, 0].unsqueeze(1)
        open_routes = global_features[:, 1].to(torch.bool)
        mixed_backhauls = global_features[:, 2].to(torch.bool)
        distance_limits = global_features[:, 3].unsqueeze(1)
        time_limits = global_features[:, 4].unsqueeze(1)
        deltas = torch.cdist(locations, locations)
        deltas[open_routes, :, 0] = 0.0

        backhauls_instances = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
        backhauls_instances = backhauls_instances.unsqueeze(-1).expand_as(demands_b)
        backhauls_mask = (demands_b > 0) & backhauls_instances
        linehauls_mask = (demands > 0) & backhauls_instances
        invalid_arcs = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)

        h_hat = global_embeddings.unsqueeze(1)
        if self.q_global is not None:
            q_g = self.q_global(h_hat)
        else:
            q_g = 0
        not_served = torch.ones_like(demands, dtype=torch.bool)

        if actions is None or actions.size(1) < 1:
            solution = torch.zeros([batch_size, 1], dtype=torch.long, device=device)
        else:
            solution = actions[:, :1]
        batch_indices = torch.arange(batch_size, device=device)
        current_node = solution[:, -1].clone()
        not_served[batch_indices, current_node] = False

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=device)

        earliest_start_time, latest_start_time = time_windows.unbind(-1)
        latest_start_time = torch.where(
            open_routes.unsqueeze(-1), float("inf"), latest_start_time
        )
        leave_time = earliest_start_time[:, 0].view(-1, 1)
        deliveries = torch.zeros_like(leave_time)
        pickups = torch.zeros_like(leave_time)
        distance = torch.zeros_like(leave_time)
        total_distance = torch.zeros_like(leave_time)
        is_depot = current_node == 0
        open_route_flags = open_routes
        open_routes = open_routes.unsqueeze(-1)
        mixed_backhauls = mixed_backhauls.unsqueeze(-1)

        self.glimpse.precalculate(node_embeddings)

        while not_served.any():
            potential_arrival_times = leave_time + deltas[batch_indices, current_node, :]
            potential_start_times = torch.max(potential_arrival_times, earliest_start_time)
            exceed_latest_start_time = potential_start_times > latest_start_time
            exceed_time_limit = (
                potential_start_times
                + services[batch_indices, :]
                + deltas[batch_indices, :, 0]
                > time_limits
            )
            exceed_infinite_time_limit = potential_start_times.isinf()

            potential_distance = distance + deltas[batch_indices, current_node, :]
            potential_distance_to_depot = potential_distance + deltas[batch_indices, :, 0]
            exceed_distance_limit = potential_distance_to_depot > distance_limits
            exceed_infinite_distance_limit = potential_distance_to_depot.isinf()

            exceed_capacity = (deliveries + demands > capacities) | (
                pickups + demands_b > capacities
            )
            cannot_serve_linehaul = mixed_backhauls & (demands + pickups > capacities)

            mask = torch.ones([batch_size, seq_len], device=device).to(torch.bool)
            mask = mask.masked_fill(not_served, False)
            mask = mask.masked_fill(invalid_arcs[batch_indices, current_node, :], True)
            mask = mask.scatter(1, current_node.unsqueeze(1), True)
            mask[:, 0] = is_depot & not_served.any(dim=1)

            full_mask = mask.masked_fill(
                exceed_capacity
                | cannot_serve_linehaul
                | exceed_latest_start_time
                | exceed_infinite_time_limit
                | exceed_distance_limit
                | exceed_infinite_distance_limit
                | exceed_time_limit,
                True,
            )

            last = node_embeddings[batch_indices, current_node, :].unsqueeze(1)
            remaining_distance = torch.nan_to_num(distance_limits - distance, posinf=10.0)
            remaining_demand = capacities - deliveries
            remaining_demand_b = capacities - pickups
            distance_to_depot = deltas[batch_indices, current_node, 0]
            distance_to_depot = torch.nan_to_num(distance_to_depot, posinf=1e9).unsqueeze(-1)
            data = torch.cat(
                (
                    last.squeeze(1),
                    leave_time,
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
            h_c = q_g + dynamic_context.unsqueeze(1)

            dist_feat = deltas[batch_indices, current_node, :].unsqueeze(-1)
            u = self.glimpse(h_c, mask, dist=dist_feat)
            if self.edge_distance_scale is not None:
                edge_dist = deltas[batch_indices, current_node, :]
                u = u - self.edge_distance_scale * edge_dist
            if attn_matrix is not None and self.edge_attn_scale is not None:
                attn_feat = attn_matrix[batch_indices, current_node, :]
                u = u + self.edge_attn_scale * attn_feat

            u = u.masked_fill(mask, float("-inf"))

            if actions is None or actions.size(1) < solution.size(1) + 1:
                probas = torch.nn.functional.softmax(u, dim=-1)
                if decode_mode == "greedy":
                    _, selected_node = self.greedy_decoding(probas)
                elif decode_mode == "sample":
                    _, selected_node = self.sample_decoding(probas)
                else:
                    raise NotImplementedError(f"Decoding mode {decode_mode} not implemented.")
            else:
                selected_node = actions[:, solution.size(1)]

            log_probabilities = log_probabilities + self.cross_entropy(u, selected_node)
            solution = torch.cat((solution, selected_node.unsqueeze(-1)), dim=1)
            not_served[batch_indices, solution[:, -1]] = False

            selected_is_depot = selected_node == 0
            selected_is_feasible = ~full_mask.gather(1, selected_node.unsqueeze(-1)).squeeze(-1)
            recourse_triggered = (~selected_is_depot) & (~selected_is_feasible)

            selected_demands = demands[batch_indices, selected_node]
            selected_demands_b = demands_b[batch_indices, selected_node]
            deliveries_next = deliveries + selected_demands.unsqueeze(1)
            pickups_next = pickups + selected_demands_b.unsqueeze(1)
            distance_next = potential_distance.gather(1, selected_node.unsqueeze(-1))

            selected_service_time = services[batch_indices, selected_node].unsqueeze(1)
            selected_start_times = potential_start_times.gather(1, selected_node.unsqueeze(-1))
            leave_time_next = selected_start_times + selected_service_time

            total_distance_next = total_distance.clone()
            if selected_is_depot.any():
                total_distance_next[selected_is_depot] = (
                    total_distance_next[selected_is_depot]
                    + distance_next[selected_is_depot]
                    + deltas[
                        batch_indices[selected_is_depot],
                        selected_node[selected_is_depot],
                        0,
                    ].unsqueeze(-1)
                )
                deliveries_next[selected_is_depot] = 0
                pickups_next[selected_is_depot] = 0
                distance_next[selected_is_depot] = 0
                leave_time_next[selected_is_depot] = time_windows[
                    selected_is_depot, 0, 0
                ].unsqueeze(1)

            current_node_next = selected_node
            is_depot_next = selected_is_depot
            if recourse_triggered.any():
                dist_depot_to_next = deltas[
                    batch_indices[recourse_triggered], 0, selected_node[recourse_triggered]
                ].unsqueeze(-1)
                dist_next_to_depot = deltas[
                    batch_indices[recourse_triggered], selected_node[recourse_triggered], 0
                ].unsqueeze(-1)
                recourse_open = open_route_flags[recourse_triggered].unsqueeze(-1)
                recourse_cost = dist_depot_to_next + dist_next_to_depot * (~recourse_open)
                total_distance_next[recourse_triggered] = (
                    total_distance_next[recourse_triggered] + recourse_cost
                )
                deliveries_next[recourse_triggered] = deliveries[recourse_triggered]
                pickups_next[recourse_triggered] = pickups[recourse_triggered]
                distance_next[recourse_triggered] = distance[recourse_triggered]
                leave_time_next[recourse_triggered] = leave_time[recourse_triggered]
                current_node_next = torch.where(
                    recourse_triggered, current_node, current_node_next
                )
                is_depot_next = torch.where(recourse_triggered, is_depot, is_depot_next)

            not_served[:, 0] = ~is_depot_next
            deliveries = deliveries_next
            pickups = pickups_next
            distance = distance_next
            leave_time = leave_time_next
            total_distance = total_distance_next
            current_node = current_node_next
            is_depot = is_depot_next

        assert solution.shape[1] >= seq_len
        costs = total_distance.squeeze(1)
        return -log_probabilities, solution, costs
