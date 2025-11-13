import copy
from abc import ABC
from typing import Optional, Tuple

import torch

from mavrp.env.models import TransformerModel


class RolloutBaseline(torch.nn.Module, ABC):
    def __init__(self, config, model: TransformerModel):
        super().__init__()
        self.config = config
        self.model = copy.deepcopy(model)
        self.model.to(model.config.device)
        self.model.eval()
        self.model.freeze()

    @torch.no_grad()
    def evaluate(self,
                 inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 tour_lengths: Optional[torch.Tensor] = None):
        if inputs is None:
            raise ValueError("Inputs cannot be None in RolloutBaseline.evaluate")
        _, _, tour_costs = self.model(inputs, decode_mode='greedy')
        return tour_costs

    def validate(self, inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 tour_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.evaluate(inputs, tour_lengths)

    def update(self, model : TransformerModel, **kwargs) -> None:
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        self.model.freeze()

    def state(self) -> dict:
        return self.model.state_dict()

    def load_state(self, state) -> None:
        self.model.load_state_dict(state)


class ExponentialBaseline(torch.nn.Module, ABC):

    def __init__(self, config, beta=0.8):
        super().__init__()
        self.beta = beta
        self.avg_baseline = None

    @torch.no_grad()
    def evaluate(self,
                 inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 tour_lengths: Optional[torch.Tensor] = None):
        if tour_lengths is None:
            raise ValueError("Tour lengths  cannot be None in ExponentialBaseline.evaluate")

        if self.avg_baseline is None:
            new_avg_baseline = tour_lengths
        else:
            new_avg_baseline = self.beta * self.avg_baseline + (1 - self.beta) * tour_lengths

        self.avg_baseline = new_avg_baseline.detach()
        return self.avg_baseline

    def validate(self, inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 tour_lengths: Optional[torch.Tensor] = None):
        return tour_lengths

    def reset_baseline(self, new_start: Optional[torch.Tensor] = None):
        """ Reset the baseline average and penalties to new starting values """
        self.avg_baseline = new_start

    def update(self, model, **kwargs):
        new_beta = kwargs.pop('new_beta', False)
        if new_beta is not None:
            self.beta = new_beta

    def state(self):
        return {'avg_baseline': self.avg_baseline, 'beta': self.beta}

    def load_state(self, state):
        self.avg_baseline = state['avg_baseline']
        self.beta = state['beta']
