from typing import Any

import torch

from mavrp.env.decoders.recourse import RecourseDecoder


class MultiStartRecourseDecoder(RecourseDecoder):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        inputs: Any,
        node_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        decode_mode: str = "sample",
        actions: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any, Any]:
        assert isinstance(inputs, tuple) and len(inputs) == 2, "inputs must be (node_features, global_features)"
        assert actions is None, "actions not supported in multistart decoder"
        batch_size, seq_len, _ = node_embeddings.size()

        device = node_embeddings.device
        n_starts = seq_len - 1
        node_embeddings = node_embeddings.repeat(n_starts, 1, 1)
        global_embeddings = global_embeddings.repeat(n_starts, 1)
        node_features, global_features = inputs
        node_features = node_features.repeat(n_starts, 1, 1)
        global_features = global_features.repeat(n_starts, 1)
        inputs = (node_features, global_features)
        actions = torch.zeros((batch_size * n_starts, 2), device=device, dtype=torch.long)
        actions[:, 1] = (
            torch.arange(1, n_starts + 1, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .view(-1)
        )
        log_probabilities, solution, costs, metrics = super().forward(
            inputs,
            node_embeddings,
            global_embeddings,
            actions=actions,
            decode_mode=decode_mode,
            **kwargs,
        )

        solution = solution.view(n_starts, batch_size, -1)
        costs = costs.view(n_starts, batch_size)
        metrics = metrics.view(n_starts, batch_size)
        log_probabilities = log_probabilities.view(n_starts, batch_size)

        costs, costs_idx = torch.min(costs, dim=0)
        batch_index = torch.arange(batch_size, device=device)
        solution = solution[costs_idx, batch_index, :]
        log_probabilities = log_probabilities.mean(dim=0)
        metrics = metrics[costs_idx, batch_index]

        return log_probabilities, solution, costs, metrics
