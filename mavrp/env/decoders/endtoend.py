from typing import Optional

import torch.nn
import torch_geometric.utils

from mavrp.env.decoders.base import DecoderBase
from mavrp.env.decoders.pointer import PointerAttention
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


        # Edge attention scaling (if provided in config)
        if self.config.use_edge_attn:
            self.edge_attn_scale = torch.nn.Parameter(torch.tensor(1.0, device=self.config.device))
        else:
            self.edge_attn_scale = None

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.dynamic_context_embedding.weight)
        if self.config.use_global_in_context:
            torch.nn.init.xavier_uniform_(self.q_global.weight)
        self.glimpse.reset_parameters()

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor],
                node_embeddings: torch.Tensor,
                global_embeddings: torch.Tensor,
                decode_mode: str = "sample",
                actions: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                edge_attn_scores: Optional[torch.Tensor] = None,
                ):
        batch_size, seq_len, embedding_dim = node_embeddings.size()  # seq_len = nb_clients + nb_depot (1)
        device = node_embeddings.device
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
            attn_matrix = torch_geometric.utils.to_dense_adj(edge_index=edge_index, edge_attr=edge_scores, batch=batch_vec)

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
        # for b in range(deltas.size(0)):
        #     deltas[b].fill_diagonal_(float('inf'))
        # deltas[:, 0, 0] = 0.0  # important since SolutionDecoder generates empty routes
        # set deltas[:, :, 0] = 0.0 if open_routes is > 0.5
        deltas[open_routes, :, 0] = 0.0

        # if backhauls sum > 0 and not mixed_backhauls linehauls must be served before backhauls.
        # So set a high value from backhaul to linehaul (avoiding linehaul after backhaul clients).
        backhauls_instances = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
        backhauls_instances = backhauls_instances.unsqueeze(-1).expand_as(demands_b)
        backhauls_mask = (demands_b > 0) & backhauls_instances  # [batch_size, seq_len], True where node is backhaul
        linehauls_mask = (demands > 0) & backhauls_instances  # [batch_size, seq_len], True where node is linehaul

        invalid_arcs = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)
        # instead of deltas[invalid_mask] = 1e9
        # b_idx, i_idx, j_idx = torch.where(invalid_arcs)
        # deltas[b_idx, i_idx, j_idx] = float('inf')

        h_hat = global_embeddings.unsqueeze(1)
        if self.q_global is not None:
            q_g = self.q_global(h_hat)  # [B,1,E]
        else:
            q_g = 0
        not_served = torch.ones_like(demands, dtype=torch.bool)

        # Initialize solution
        if actions is None or actions.size(1) < 1:
            solution = torch.zeros([batch_size, 1], dtype=torch.long, device=device)
        else:
            solution = actions[:, :1]
        batch_indices = torch.arange(batch_size, device=device)
        not_served[batch_indices, solution[:, -1]] = False

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=device)

        earliest_start_time, latest_start_time = time_windows.unbind(-1)  # [batch_size, seq_len]
        # set latest start time of depot to inf  when open_routes is True
        # latest_start_time[open_routes, 0] = float('inf') # in place operation, don't
        latest_start_time = torch.where(open_routes.unsqueeze(-1), float('inf'), latest_start_time)
        leave_time = earliest_start_time[:, 0].view(-1, 1)  # [batch_size, 1, 1]
        deliveries = torch.zeros_like(leave_time)
        pickups = torch.zeros_like(leave_time)

        distance = torch.zeros_like(leave_time)
        total_distance = torch.zeros_like(leave_time)
        is_depot = torch.full((batch_size,), True, dtype=torch.bool, device=node_embeddings.device)
        has_demand_b = torch.sum(demands_b, dim=-1, keepdim=True) > 0
        open_routes = open_routes.unsqueeze(-1)
        mixed_backhauls = mixed_backhauls.unsqueeze(-1)

        self.glimpse.precalculate(node_embeddings)

        while not_served.any():
            potential_arrival_times = leave_time + deltas[batch_indices, solution[:, -1], :]  # [batch_size, seq_len]
            potential_start_times = torch.max(potential_arrival_times, earliest_start_time)  # [batch_size, seq_len]
            exceed_latest_start_time = potential_start_times > latest_start_time  # [batch_size, seq_len]
            exceed_time_limit = potential_start_times + services[batch_indices, :] + deltas[batch_indices, :,
                                                                                     0] > time_limits  # [batch_size, seq_len]
            exceed_infinite_time_limit = potential_start_times.isinf()  # [batch_size, seq_len]

            potential_distance = distance + deltas[batch_indices, solution[:, -1], :]  # [batch_size, seq_len]

            potential_distance_to_depot = potential_distance + deltas[batch_indices, :, 0]
            exceed_distance_limit = potential_distance_to_depot > distance_limits  # [batch_size, seq_len]
            exceed_infinite_distance_limit = potential_distance_to_depot.isinf()  # [batch_size, seq_len]

            exceed_capacity = (deliveries + demands > capacities) | (pickups + demands_b > capacities)

            # no place to serve linehaul
            cannot_serve_linehaul = mixed_backhauls & (demands + pickups > capacities)

            # Create base mask: visited nodes and invalid arcs
            mask = torch.ones([batch_size, seq_len], device=device).to(torch.bool)
            mask = mask.masked_fill(not_served, False)
            mask = mask.masked_fill(invalid_arcs[batch_indices, solution[:, -1], :], True)

            mask = mask.masked_fill(
                exceed_capacity | cannot_serve_linehaul | exceed_latest_start_time | exceed_infinite_time_limit | exceed_distance_limit | exceed_infinite_distance_limit | exceed_time_limit,
                True)  # mask: [batch_size, seq_len]

            mask = mask.scatter(1, solution[:, -1].unsqueeze(1), True)
            mask[:, 0] = is_depot & not_served.any(dim=1)

            last = node_embeddings[batch_indices, solution[:, -1], :].unsqueeze(1)

            remaining_distance = torch.nan_to_num(distance_limits - distance, posinf=10.0)  # - distance

            remaining_demand = capacities - deliveries
            remaining_demand_b = capacities - pickups

            distance_to_depot = deltas[batch_indices, solution[:, -1], 0]
            distance_to_depot = torch.nan_to_num(distance_to_depot, posinf=1e9).unsqueeze(-1)
            data = torch.cat((last.squeeze(1),
                              leave_time,
                              remaining_distance,
                              remaining_demand,
                              remaining_demand_b,
                              distance_to_depot,
                              open_routes,
                              mixed_backhauls), dim=-1)

            dynamic_context = self.dynamic_context_embedding(data)

            h_c = q_g + dynamic_context.unsqueeze(1)

            dist_feat = deltas[batch_indices, solution[:, -1], :].unsqueeze(-1)
            u = self.glimpse(h_c, mask, dist=dist_feat)
            if self.edge_distance_scale is not None:
                edge_dist = deltas[batch_indices, solution[:, -1], :]  # [B,seq_len]
                u = u - self.edge_distance_scale * edge_dist

            # incorporate edge attention bias
            if attn_matrix is not None and self.edge_attn_scale is not None:
                # attention from last selected node to all candidates
                attn_feat = attn_matrix[batch_indices, solution[:, -1] , :]  # [batch_size, seq_len]
                u = u + self.edge_attn_scale * attn_feat

            u = u.masked_fill(mask, float('-inf'))  # [batch_size, seq_len]

            if actions is None or actions.size(1) < solution.size(1) + 1:
                # Sample or select action based on decode_mode
                probas = torch.nn.functional.softmax(u, dim=-1)
                if decode_mode == "greedy":
                    _, selected_node = self.greedy_decoding(probas)
                elif decode_mode == "sample":
                    _, selected_node = self.sample_decoding(probas)
                else:
                    raise NotImplementedError(f"Decoding mode {decode_mode} not implemented.")
            else:
                # Use the provided action
                selected_node = actions[:, solution.size(1)]
                # Note: TorchScript doesn't support assert with messages
                # We assume actions are valid

            log_probabilities = log_probabilities + self.cross_entropy(u, selected_node)
            solution = torch.cat((solution, selected_node.unsqueeze(-1)), dim=1)
            not_served[batch_indices, solution[:, -1]] = False

            selected_demands = demands[batch_indices, selected_node]  # selected_demands: [batch_size]
            selected_demands_b = demands_b[batch_indices, selected_node]  # selected_demands_b: [batch_size]
            deliveries = deliveries + selected_demands.unsqueeze(1)  # [batch_size, 1]
            pickups = pickups + selected_demands_b.unsqueeze(1)  # [batch_size, 1]

            distance = potential_distance.gather(1, selected_node.unsqueeze(-1))  # [batch_size, 1]

            # Demand and time calculations
            selected_service_time = services[batch_indices, selected_node].unsqueeze(1)
            selected_start_times = potential_start_times.gather(1, selected_node.unsqueeze(-1))  # [batch_size]
            leave_time = (selected_start_times + selected_service_time)  # [batch_size, 1]

            is_depot = selected_node == 0
            # if not is_depot set it as not served
            not_served[~is_depot, 0] = True
            if is_depot.any():
                # add return distance to depot
                total_distance[is_depot] = total_distance[is_depot] + distance[is_depot] + deltas[
                    batch_indices[is_depot], selected_node[is_depot], 0].unsqueeze(-1)
                deliveries[is_depot] = 0
                pickups[is_depot] = 0
                distance[is_depot] = 0
                leave_time[is_depot] = time_windows[is_depot, 0, 0].unsqueeze(1)

        # batch_idx = torch.arange(batch_size, device=solution.device)[:, None]
        # deltas[global_features[:, 1].bool(), :, 0] = 0.0
        # costs = torch.sum(deltas[batch_idx, solution[:, :-1], solution[:, 1:]], dim=1)

        assert solution.shape[1] >= seq_len
        costs = total_distance.squeeze(1)

        return -log_probabilities, solution, costs

