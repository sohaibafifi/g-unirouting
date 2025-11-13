from typing import Any, Optional

import torch
from torch import Tensor

from mavrp.env.decoders import EndToEndDecoder


class MultiStartDecoder(EndToEndDecoder):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                inputs: Any,
                node_embeddings: torch.Tensor,
                global_embeddings: torch.Tensor,
                decode_mode: str = "sample",
                actions: Any = None,
                **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(inputs, tuple) and len(inputs) == 2, "inputs must be (node_features, global_features)"
        assert actions is None, "actions not supported in multistart decoder"
        batch_size, seq_len, _ = node_embeddings.size()

        device = node_embeddings.device
        n_starts = seq_len - 1  # the depot is already served and always the first node
        # duplicate the batch k times
        node_embeddings = node_embeddings.repeat(n_starts, 1, 1)
        global_embeddings = global_embeddings.repeat(n_starts, 1)
        node_features, global_features = inputs
        # duplicate the batch k times
        node_features = node_features.repeat(n_starts, 1, 1)  # [n_starts * batch_size, seq_len, -1]
        global_features = global_features.repeat(n_starts, 1)  # [n_starts * batch_size, -1]
        inputs = (node_features, global_features)
        # use actions to set the second node to 1 .. seq_len - 1
        # the first node is the depot and is always 0
        # since actions should be of shape [batch_size * n_starts, seq_len] we fill the rest of the actions with -1
        actions = torch.zeros((batch_size * n_starts, 2), device=device, dtype=torch.long)
        actions[:, 1] = torch.arange(1, n_starts + 1, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size,
                                                                                                           1).view(-1)
        log_probabilities, solution, costs = super().forward(inputs,
                                                             node_embeddings,
                                                             global_embeddings,
                                                             actions=actions,
                                                             decode_mode=decode_mode,
                                                             **kwargs)


        # reshape the solution and costs to [n_starts, batch_size, -1] so that we can get the best solution
        solution = solution.view(n_starts, batch_size, -1)
        costs = costs.view(n_starts, batch_size)
        log_probabilities = log_probabilities.view(n_starts, batch_size)

        # get the best solution and costs for each batch
        costs, costs_idx = torch.min(costs, dim=0)
        batch_index = torch.arange(batch_size, device=device)
        solution = solution[costs_idx, batch_index, :]
        # mean log_probabilities over the n_starts
        log_probabilities = log_probabilities.mean(dim=0)

        return log_probabilities, solution, costs
