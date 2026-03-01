from typing import Optional

import torch
from torch import Tensor

from mavrp.env.normalization import EdgeNormalization

# Try to import knn_graph from torch_cluster, fallback to Python implementation
try:
    from torch_cluster import knn_graph
except ImportError:
    print("Warning: torch_cluster not found. Using fallback Python implementation for knn_graph.")



    def knn_graph_python(x: Tensor, k: int, batch: Optional[Tensor] = None, loop: bool = False) -> Tensor:
        """
        Pure Python implementation of knn_graph as fallback when torch_cluster is not available.

        Args:
            x: Node positions [N, d] where N is number of nodes, d is dimensionality
            k: Number of nearest neighbors
            batch: Batch vector [N] indicating which graph each node belongs to
            loop: Whether to include self-loops

        Returns:
            edge_index: [2, E] tensor of edges where E = N * k (or N * (k+1) if loop=True)
        """
        device = x.device
        num_nodes = x.size(0)

        # If no batch is provided, assume all nodes are in the same graph
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # Compute pairwise distances
        # distances[i, j] = ||x[i] - x[j]||^2
        distances = torch.cdist(x, x, p=2.0)  # [N, N]

        # Create mask for same-batch nodes
        batch_mask = batch.unsqueeze(1) == batch.unsqueeze(0)  # [N, N]

        # Set distances to infinity for nodes in different batches
        distances = distances.masked_fill(~batch_mask, float('inf'))

        # If not including self-loops, set diagonal to infinity
        if not loop:
            distances = distances + torch.diag(torch.full((num_nodes,), float('inf'), device=device))

        # Find k nearest neighbors for each node
        # topk returns (values, indices) where indices are the neighbor node IDs
        _, neighbors = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)  # [N, k]

        # Build edge_index
        src = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k)  # [N, k]
        dst = neighbors  # [N, k]

        # Flatten to get edge list
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # [2, N*k]

        # Remove edges where dst is inf (can happen if k > actual neighbors in batch)
        valid_mask = edge_index[1] < num_nodes
        edge_index = edge_index[:, valid_mask]

        return edge_index
    knn_graph = knn_graph_python


class Graph:
    def __init__(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                 pos: Optional[Tensor] = None,
                 batch: Optional[Tensor] = None,
                 batch_size: Optional[int] = None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.batch_size = batch_size
        self.batch = batch


class GraphEmbeddingLayer(torch.nn.Module):
    """
    Graph Embedding Layer
    Inherits the global node logic. Now, the global node is fully integrated
    into the node set, and edge calculations remain unchanged.
    """

    def __init__(self, p_config):
        super(GraphEmbeddingLayer, self).__init__()
        self.embedding_dim = p_config.embedding_dim
        if p_config.enhanced_scores:
            self.edge_embedding = torch.nn.Linear(4, self.embedding_dim)
        else:
            self.edge_embedding = torch.nn.Linear(2, self.embedding_dim)
        self.edge_norm = EdgeNormalization(self.embedding_dim, p_config.normalization)
        self.config = p_config
        self.nb_neighbors = self.config.nb_neighbors
        self.sample_neighbors = self.config.sample_neighbors

    def _load_from_state_dict(self,
                              state_dict,  # the checkpoint dict
                              prefix,  # e.g. "encoder.init_layer.init_embd."
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs):
        layer = 'edge_embedding'
        w_key = prefix + layer + ".weight"
        # if this layer appears in the checkpoint, pad its weights with zeros
        if w_key in state_dict:
            old_w = state_dict[w_key]
            if old_w.size(1) == getattr(self, layer).weight.size(1):
                # if the size is the same, just copy it
                new_w = old_w
            else:
                # if the size is different, create a new tensor with zeros
                # and copy the “overlap”
                new_w = torch.zeros_like(getattr(self, layer).weight)
                new_w[: old_w.size(0), : old_w.size(1)] = old_w

            state_dict[w_key] = new_w

        # now let nn.Module do its normal loading
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

    @staticmethod
    # @torch.compile()
    def get_deltas(node_features: torch.Tensor, global_features: torch.Tensor):
        # Compute distances and factor
        locations = node_features[:, :, :2]
        open_routes = global_features[:, 1].to(torch.bool)
        mixed_backhauls = global_features[:, 2].to(torch.bool)
        demands_b = node_features[:, :, 3]
        demands = node_features[:, :, 2]
        distances = torch.cdist(locations, locations)
        if open_routes.any():
            distances = distances.clone()
            zeros = torch.zeros_like(distances[:, :, 0])
            distances[:, :, 0] = torch.where(open_routes.unsqueeze(-1), zeros, distances[:, :, 0])
        backhauls_instances = (torch.sum(demands_b, dim=-1) > 0) & (~mixed_backhauls)
        backhauls_instances = backhauls_instances.unsqueeze(-1).expand_as(demands_b)
        backhauls_mask = (demands_b > 0) & backhauls_instances  # [batch_size, seq_len], True where node is backhaul
        linehauls_mask = (demands > 0) & backhauls_instances  # [batch_size, seq_len], True where node is linehaul

        invalid_mask = backhauls_mask.unsqueeze(-1) & linehauls_mask.unsqueeze(1)
        if invalid_mask.any():
            distances = distances.masked_fill(invalid_mask, float('inf'))

        # TODO : filter out the edges that are not valid from time windows

        return distances

    def forward(self, node_features: torch.Tensor, global_features: torch.Tensor,
                node_embeddings: torch.Tensor, global_embeddings: torch.Tensor) -> tuple[Graph, Tensor]:
        nbatch, _, _ = node_features.size()

        distances = self.get_deltas(node_features, global_features)
        graph = self.build_knn_graph(node_features, distances, self.nb_neighbors,
                                     sample_neighbors=self.sample_neighbors)

        x = node_embeddings.view(-1, self.embedding_dim)
        edge_index = graph.edge_index
        edge_attr = self.edge_embedding(graph.edge_attr[:, :self.edge_embedding.in_features])
        edge_attr = self.edge_norm(edge_attr)
        return Graph(x, edge_index, edge_attr, batch=graph.batch, pos=graph.pos), global_embeddings

    @staticmethod
    def build_knn_graph(
            node_features: torch.Tensor,  # shape: (batch_size, seq_len, feat_dim)
            distances: torch.Tensor,  # shape: (batch_size, seq_len, seq_len)
            nb_neighbors: int,
            sample_neighbors: bool = False
    ) -> Graph:

        device = node_features.device
        batch_size, seq_len, feat_dim = node_features.shape
        nb_neighbors = min(nb_neighbors, seq_len - 1)  # TODO: nb_neighbors should be different for each layer step

        # Create a 'batch' vector indicating which graph each node belongs to
        batch_vec = torch.arange(batch_size, device=device).repeat_interleave(seq_len)

        # Build the k-NN graph using only the 2D positions
        if not sample_neighbors:
            if device.type == 'mps':
                # MPS backend does not support knn_graph
                # convert to CPU and use knn_graph
                edge_index = knn_graph(node_features[:, :, :2].view(-1, 2).cpu(), k=nb_neighbors, batch=batch_vec.cpu(), loop=False)
                edge_index = edge_index.to(device)
            else:
                edge_index = knn_graph(node_features[:, :, :2].view(-1, 2), k=nb_neighbors, batch=batch_vec, loop=False)
        else:
            num_nodes = batch_size * seq_len

            temperature = 1.0

            prob = torch.softmax(-distances / temperature, dim=-1)
            mask = torch.eye(seq_len, device=device).to(torch.bool).unsqueeze(0)
            prob = prob.masked_fill(mask, 0)
            prob = prob.view(num_nodes, seq_len)
            samples = torch.multinomial(prob, num_samples=nb_neighbors,
                                        replacement=False)  # shape: [num_nodes, nb_neighbors]
            src = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, nb_neighbors)
            src = src.reshape(-1)
            dst = samples.reshape(-1)

            edge_index = torch.stack([src, dst], dim=0)
            # remove edges going into depot node
            # edge_index = edge_index[:, edge_index[0] % seq_len != 0]

        # For each edge (src, dst), gather the appropriate distance
        src = edge_index[0]
        dst = edge_index[1]

        batch_src = src // seq_len
        node_src = src % seq_len
        node_dst = dst % seq_len

        dist = distances[batch_src, node_src, node_dst].unsqueeze(-1)  # shape => [E, 1]
        angle = (torch.atan2(node_features[batch_src, node_dst, 1] - node_features[batch_src, node_src, 1],  # y
                             node_features[batch_src, node_dst, 0] - node_features[batch_src, node_src, 0])  # x
                 .unsqueeze(-1))

        # Compute demand differences as additional score
        demands = node_features[..., 2]  # [batch, seq_len]
        demand_diff = torch.abs(demands.unsqueeze(-1) - demands.unsqueeze(1))  # [batch, seq_len, seq_len]
        demand_diff = demand_diff[batch_src, node_src, node_dst].unsqueeze(-1)  # [E, 1]

        # Compute window slack feature
        tw_start = node_features[..., 4]  # [batch, seq_len]
        tw_end = node_features[..., 5]  # [batch, seq_len]

        time_distance = tw_end[batch_src, node_dst] - tw_start[batch_src, node_src]  # [E,]
        time_distance = time_distance.unsqueeze(-1)

        time_distance = torch.nan_to_num(time_distance, nan=0.0, posinf=1e9, neginf=1e9)

        edge_attr = torch.cat([dist, angle, demand_diff, time_distance], dim=-1)

        mask = (dist <= 1000).squeeze(-1)  # [E,]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]

        return Graph(node_features.view(-1, feat_dim),
                     edge_index,
                     edge_attr,
                     batch=batch_vec,
                     pos=node_features[:, :, :2].view(-1, 2))
