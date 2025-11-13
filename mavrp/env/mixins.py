import os
import os.path as osp
import random
import sys
from abc import ABC, abstractmethod

import torch


class InfoMixin(ABC):
    """
    Mixin to get information about the model
    """

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    def get_num_params(self):
        """
        Get the number of trainable parameters in the model
        :return: int : Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self):
        path = f'{random.randrange(sys.maxsize)}.pt'
        torch.save(self.state_dict(), path)
        model_size = osp.getsize(path)
        os.remove(path)
        return model_size

    def get_name(self):
        """
        Get the name of the model
        :return: str : Name of the model
        """
        if hasattr(self, 'name'):
            return self.name
        return f"{self.__class__.__name__}"


class FreezingMixin(ABC):
    """
    Mixin to freeze and unfreeze the model
    """

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def freeze(self):
        """
        Freeze the model
        :return:
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """
        Unfreeze the model
        :return:
        """
        for param in self.parameters():
            param.requires_grad = True
        self.train()
