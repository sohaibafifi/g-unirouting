from enum import Enum
from typing import Any

from torch.utils.data import Dataset

from .datasets import MTVRPDataset


class Objective(Enum):
    MAX = -1.0
    MIN = 1.0


class MTVRP:

    def num_node_features(self):
        return 7

    def num_global_features(self):
        return 6

    def objective(self):
        return Objective.MIN

    def factor(self):
        return self.objective().value

    def dataset_cls(self) -> type[Dataset]:
        return MTVRPDataset

    def dataset(self, *args: Any, **kwargs: Any) -> Dataset:
        kwargs['variant'] = self.__class__.__name__
        return (
            self.dataset_cls()(*args, **kwargs))

    def init_cls(self):
        from mavrp.env.encoders.init import InitialEmbeddingLayer
        return InitialEmbeddingLayer

    def graph_cls(self):
        from mavrp.env.encoders.graph import GraphEmbeddingLayer
        return GraphEmbeddingLayer

    def init_embedding(self, config):
        return self.init_cls()(config)

    def graph_embedding(self, config):
        return self.graph_cls()(config)

    @staticmethod
    def get_variants():
        return [
            f'{o}vrp{m}{b}{l}{tw}'
            for b in ['', 'b']
            for tw in ['', 'tw']
            for o in ['', 'o']
            for m in (['m', ''] if b == 'b' else [''])
            for l in ['', 'l']
        ] + ['mtvrp']


class OVRP(MTVRP):
    pass


class VRP(MTVRP):
    pass


class VRPB(MTVRP):
    pass


class VRPMB(MTVRP):
    pass
