from abc import ABC, abstractmethod

import torch.distributions
import torch.nn
from torch.distributions import Categorical


class DecoderBase(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                node_embeddings: torch.Tensor,
                global_embeddings: torch.Tensor,
                decode_mode: str = "sample",
                actions=None):
        pass

    @staticmethod
    def xgreedy_decoding(probas: torch.Tensor):
        """
        Greedy decoding with random tie-breaking when multiple max values are found.

        :param probas: Tensor of shape [batch_size, seq_len].
        :return: Tuple (max_values, selected_indices)
        """
        max_values = torch.amax(probas, dim=1, keepdim=True)  # [batch_size, 1]
        is_max = probas == max_values  # [batch_size, seq_len], boolean mask

        # Random values for tie-breaking
        random_noise = torch.rand_like(probas)
        random_noise[~is_max] = -1  # Ensure only max entries are considered

        # Now pick randomly among max-value candidates
        _, selected_node = torch.max(random_noise, dim=1)
        return max_values.squeeze(1), selected_node

    @staticmethod
    def greedy_decoding(probas: torch.Tensor):
        """
        Performs greedy decoding on a batch of probability distributions or logits.

        :param probas: Tensor of shape [batch_size, seq_len] representing
                       probabilities or logits for each category in the sequence.
        :return: Tuple of (max_probabilities, max_indices).
                 max_probabilities: Tensor of shape [batch_size] with maximum probabilities or logits.
                 max_indices: Tensor of shape [batch_size] with indices of categories having maximum values.
        """
        max_values, selected_node = torch.max(probas, dim=1)

        return max_values, selected_node

    @staticmethod
    def sample_decoding(probas: torch.Tensor):
        """
        Samples indices from a batch of categorical distributions.

        :param probas: Tensor of shape [batch_size, seq_len] representing
                    probabilities or logits for each category in the sequence.
        :return: Tuple of (sampled_probabilities, sampled_indices).
                sampled_probabilities: Tensor of shape [batch_size] with probabilities of sampled indices.
                sampled_indices: Tensor of shape [batch_size] with indices of sampled categories.
        """
        selected_node = torch.multinomial(probas, 1).squeeze(1)  # [batch_size]
        selected_probas = probas.gather(1, selected_node.unsqueeze(1)).squeeze(1)

        return selected_probas, selected_node

    @staticmethod
    def topp_decoding(probas: torch.Tensor, top_p=0.9):
        """
        Performs top-p (nucleus) sampling from a batch of probability distributions or logits.

        :param probas: Tensor of shape [batch_size, seq_len] representing
                       probabilities or logits for each category in the sequence.
        :return: Tuple of (sampled_probabilities, sampled_indices).
                 sampled_probabilities: Tensor of shape [batch_size] with probabilities of sampled indices.
                 sampled_indices: Tensor of shape [batch_size, 1] with indices of sampled categories.
        """

        # Sort probabilities and indices in descending order
        sorted_probas, sorted_indices = torch.sort(probas, descending=True, dim=1)

        # Cumulative probabilities
        cumulative_probas = torch.cumsum(sorted_probas, dim=1)

        # Remove all tokens with a cumulative probability above the threshold p
        removed = cumulative_probas - sorted_probas > top_p

        # Shift the indices to the right to keep the first one above the threshold
        removed[:, 1:] = removed[:, :-1].clone()
        removed[:, 0] = 0

        # Zero out the probabilities that are removed
        sorted_probas[removed] = 0

        # Normalize the modified probabilities
        sorted_probas /= torch.sum(sorted_probas, dim=1, keepdim=True)

        # Sample from the filtered distribution
        selected_node = Categorical(sorted_probas).sample()
        selected_probas = sorted_probas.gather(1, selected_node.unsqueeze(1)).squeeze(1)

        # Map sampled indices back to original indices
        selected_node = sorted_indices.gather(1, selected_node.unsqueeze(1)).squeeze(1)
        return selected_probas, selected_node

    @staticmethod
    def topk_decoding(probas: torch.Tensor, top_k=5):
        """
        Performs top-k sampling from a batch of probability distributions.

        Assumes 'probas' is a tensor of shape [batch_size, 1, seq_len] or [1, seq_len] and
        converts it to [batch_size, seq_len] or [seq_len] respectively for processing.

        :param probas: Tensor representing probabilities for each category in the sequence.
        :return: Tuple of (sampled_probabilities, sampled_indices).
                 sampled_probabilities: Tensor with probabilities of sampled indices.
                 sampled_indices: Tensor with indices of sampled categories.
        """
        top_k = min(top_k, probas.size(1))

        # Extract the top-k elements
        top_values, top_indices = torch.topk(probas, top_k, dim=1)

        # Normalize these top-k values so that they sum up to 1
        top_values = top_values / top_values.sum(dim=1, keepdim=True)

        # Sample from the resulting top-k categorical distribution
        selected_sub_idx = torch.distributions.Categorical(top_values).sample()

        # Map the sampled sub-index back to the original indices
        selected_node = top_indices.gather(1, selected_sub_idx.unsqueeze(1)).squeeze(1)

        # Retrieve the selected probabilities from the original distribution
        selected_probas = probas.gather(1, selected_node.unsqueeze(1)).squeeze(1)

        return selected_probas, selected_node

    def decoding(self, probas: torch.Tensor, decode_mode: str = "sample"):
        """
        Decodes the given probabilities using the specified decoding mode.

        Args:
            probas (torch.Tensor): The probabilities to decode. [batch_size, seq_len]
            decode_mode (str, optional): The decoding mode to use. Defaults to "sample".

        Returns:
            selected_probas, selected_node: The selected probabilities and indices based on the decoding mode.
            [batch_size]

        Raises:
            NotImplementedError: If the specified decoding mode is not implemented.
        """
        decoding_method = getattr(self, f"{decode_mode}_decoding")

        return decoding_method(probas)
