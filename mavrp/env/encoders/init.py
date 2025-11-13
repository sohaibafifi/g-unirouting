import torch

from mavrp.env.normalization import Normalization
from mavrp.env.problems import MTVRP


class InitialEmbeddingLayer(torch.nn.Module):
    """
    Initial Embedding Layer
    Now we create a dedicated global node from the global features,
    rather than merging them into the depot node.
    """

    def __init__(self, p_config):
        super(InitialEmbeddingLayer, self).__init__()
        self.config = p_config
        self.embedding_dim: int = p_config.embedding_dim
        problem: MTVRP = p_config.get_problem()

        # Node embeddings
        self.features_embedding = torch.nn.Linear(problem.num_node_features() + 2, self.embedding_dim)
        self.global_embedding = torch.nn.Linear(problem.num_global_features(), self.embedding_dim)

        self.features_norm = Normalization(self.embedding_dim, 'batch')
        self.global_norm = Normalization(self.embedding_dim, 'batch')

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.features_embedding.weight)
        torch.nn.init.xavier_uniform_(self.global_embedding.weight)

    def forward(self, node_features: torch.Tensor, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param node_features: torch.Tensor : Node features [batch_size, seq_len, feature_dim]
        :param global_features: torch.Tensor : Global features [batch_size, feature_dim]
        :return: tuple[torch.Tensor, torch.Tensor] :
                 Node embeddings [batch_size, seq_len, embedding_dim],
                 Global node embedding [batch_size, embedding_dim]
        """
        capacities = global_features[:, 0]  # save the capacities first

        # normalize demands
        node_features[:, :, 2] /= capacities.view(-1, 1)
        node_features[:, :, 3] /= capacities.view(-1, 1)
        global_features[:, 0] = 1.0

        if global_features.size(1) == self.global_embedding.in_features - 1:
            # Add missing features from already generated instances
            backhaul_demand = node_features[:, :, 3].sum(dim=1) > 0
            global_features = torch.cat([global_features, backhaul_demand.unsqueeze(-1).type_as(global_features)],
                                        dim=-1)  # [batch, global_features + 1]

        global_features = torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

        global_embeddings = self.global_embedding(global_features)

        # global_embeddings = self.global_norm(global_embeddings)

        x = node_features[:, :, 0]
        y = node_features[:, :, 1]
        distance_from_depot = torch.sqrt(x ** 2 + y ** 2)
        angle = torch.atan2(y, x)

        node_features = torch.cat([distance_from_depot.unsqueeze(-1), angle.unsqueeze(-1), node_features], dim=-1)
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        node_embeddings = self.features_embedding(node_features)

        node_embeddings[:, 0] += global_embeddings
        node_embeddings = self.features_norm(node_embeddings)
        return node_embeddings, global_embeddings
