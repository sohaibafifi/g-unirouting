import functools
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class MultiStepWithWarmupLR(LambdaLR):
    r"""Creates an LR scheduler with multiple learning rate steps with a warmup period.

    Args:
        optimizer (Optimizer): The optimizer to be scheduled.
        milestones (List[int]): List of integers. Must be increasing.
            Multistep learning rate milestones.
        gamma (float): Multiplicative factor of learning rate decay.
        num_warmup_steps (int): The number of steps for the warmup phase.
        last_epoch (int, optional): The index of the last epoch when resuming
            training. (default: :obj:`-1`)
    """

    def __init__(
            self,
            optimizer: Optimizer,
            milestones: List[int],
            gamma: float,
            num_warmup_epochs: int,
            last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            milestones=milestones,
            gamma=gamma,
            num_warmup_epochs=num_warmup_epochs,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
            current_epoch: int,
            milestones: List[int],
            gamma: float,
            num_warmup_epochs: int,
    ) -> float:
        if current_epoch < num_warmup_epochs:
            return float(current_epoch + 1) / float(max(1.0, num_warmup_epochs))
        return gamma ** sum([current_epoch > milestone for milestone in milestones])
