from typing import Optional, Tuple

import torch.nn
from torch import Tensor

from .decoders import MultiStartDecoder, MultiStartRecourseDecoder, RecourseDecoder
from .encoders import GATEncoder
from .mixins import FreezingMixin, InfoMixin


class TransformerModel(torch.nn.Module, InfoMixin, FreezingMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = config.encoder(config)
        self.decoder = config.decoder(config)
        self.reset_parameters()
        self.use_edge_attn = config.use_edge_attn and isinstance(self.encoder, GATEncoder)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, inputs: Tuple[Tensor, Tensor], decode_mode: str = "sample", actions: Optional[Tensor] = None):

        encoded = list(self.encoder(inputs))
        if len(encoded) != 4:  # GATEncoder returns edge_index and edge_attn_scores
            encoded_inputs, global_embeddings = encoded
            return self.decoder(inputs, encoded_inputs, global_embeddings, decode_mode=decode_mode, actions=actions)

        else:
            # GATEncoder returns (node_embeddings, global_embeddings, edge_index, edge_attn_scores)
            return self.decoder(inputs, encoded[0], encoded[1], decode_mode=decode_mode,
                                edge_index=encoded[2], edge_attn_scores=encoded[3], actions=actions)

    def inference(self, inputs, decode_mode="greedy"):
        with torch.inference_mode():
            return self.augment_and_apply(inputs, decode_mode=decode_mode)

    def use_multi_start(self):
        decoder_cls = (
            MultiStartRecourseDecoder
            if isinstance(self.decoder, RecourseDecoder)
            else MultiStartDecoder
        )
        ms_decoder = decoder_cls(self.config).to(device=self.config.device)
        ms_decoder.load_state_dict(self.decoder.state_dict())
        ms_decoder.freeze()
        self.decoder = ms_decoder

    def augment_and_apply(self, inputs, decode_mode="greedy"):
        decoder_cls = (
            MultiStartRecourseDecoder
            if isinstance(self.decoder, RecourseDecoder)
            else MultiStartDecoder
        )
        inference_decoder = decoder_cls(self.config).to(device=self.config.device)
        inference_decoder.load_state_dict(self.decoder.state_dict())
        inference_decoder.freeze()

        inputs_augmented = self.config.get_problem().dataset_cls().augment(inputs)
        all_costs = []
        all_routes = []
        all_log_probs = []
        for data in inputs_augmented:
            if self.config.use_edge_attn and isinstance(self.encoder, GATEncoder):
                node_embeddings, global_embeddings, edge_index, edge_attn_scores = self.encoder(data)
                log_probabilities, routes, costs = inference_decoder(data, node_embeddings, global_embeddings,
                                                                     decode_mode=decode_mode,
                                                                     edge_index=edge_index,
                                                                     edge_attn_scores=edge_attn_scores)
            else:
                # run the encoder and decoder
                # data is a tuple of (node_features, global_features)
                node_embeddings, global_embeddings = self.encoder(data)[:2]
                log_probabilities, routes, costs = inference_decoder(data, node_embeddings, global_embeddings,
                                                                 decode_mode=decode_mode)

            all_costs.append(costs)
            all_routes.append(routes)  # shape (num_augment, batch_size, seq_len)
            all_log_probs.append(log_probabilities)

        # return the best solution
        all_costs = torch.stack(all_costs)  # shape (num_augment, batch_size)
        all_log_probs = torch.stack(all_log_probs)  # shape (num_augment, batch_size)
        # some routes may be shorter than the longest route, add padding (0) to make them all the same length
        max_len = max([r.size(1) for r in all_routes])
        for i in range(len(all_routes)):
            if all_routes[i].size(1) < max_len:
                all_routes[i] = torch.cat([all_routes[i],
                                           torch.zeros(all_routes[i].size(0), max_len - all_routes[i].size(1),
                                                       dtype=torch.long, device=all_routes[i].device)],
                                          dim=1)

        all_routes = torch.stack(all_routes).to(self.config.device)
        best_idx = torch.argmin(all_costs, dim=0)  # shape (batch_size,)
        best_routes = all_routes[best_idx, torch.arange(all_routes.size(1))]
        best_log_probs = all_log_probs[best_idx, torch.arange(all_log_probs.size(1))]
        return best_log_probs, best_routes, all_costs[best_idx, torch.arange(all_costs.size(1))]

    def load_from_ckpt(self, ckpt_path, baseline=False):
        state_dict = torch.load(ckpt_path, map_location=self.config.device, weights_only=False)
        state_dict = state_dict['state_dict']
        if baseline:
            keyword = 'baseline.model.'
        else:
            keyword = 'model.'
        state_dict = {k.replace(keyword + 'encoder', 'encoder').replace(keyword + 'decoder', 'decoder'): v for k, v in
                      state_dict.items() if k.startswith(keyword)}
        self.load_state_dict(state_dict)


class CriticModel(torch.nn.Module, InfoMixin, FreezingMixin):
    """
    Critic network for actor-critic training: encodes the state and regresses
    to a scalar value estimate of expected tour cost.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Shared encoder to process inputs
        self.encoder = config.encoder(config)
        # MLP head to regress embedding to scalar value
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.embedding_dim, 1)
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        # Encode inputs; take the global embedding (second output)
        encoded = list(self.encoder(inputs))
        node_embeddings, global_embeddings = encoded[:2]
        # Regress to scalar values and remove last dim
        values = self.value_head(global_embeddings).squeeze(-1)
        return values
