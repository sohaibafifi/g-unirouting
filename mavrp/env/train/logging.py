import torch


class Stats:
    """
    Track training / validation statistics directly on the same device as the model
    (CPU or GPU).  Everything is stored as 1‑D tensors, so there are **no** CPU↔GPU
    transfers during logging.  Conversion back to NumPy (e.g. for Pandas/SciPy) can
    still be done via the usual `.cpu().numpy()` call if needed.
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._init_tensors()

    # --------------------------------------------------------------------- #
    #  Private helpers
    # --------------------------------------------------------------------- #
    def _init_tensors(self) -> None:
        """Initialise/clear all tracked tensors."""
        empty = lambda: torch.empty(0, device=self.device)

        # training
        self.epoch_training_tour_costs = empty()
        self.epoch_training_baseline_tour_costs = empty()
        self.epoch_training_loss = empty()

        # validation
        self.epoch_validation_tour_costs = empty()
        self.validation_baseline_tour_costs = empty()

        # normalised validation
        self.epoch_validation_tour_costs_normalized = empty()
        self.validation_baseline_tour_costs_normalized = empty()

    def _append(self, attr_name: str, new_vals: torch.Tensor) -> None:
        """Append 1‑D tensor `new_vals` (already detached) to the stored tensor."""
        if new_vals.numel() == 0:
            return  # nothing to do
        new_vals = new_vals.to(self.device).view(-1)
        current = getattr(self, attr_name)
        setattr(self, attr_name, torch.cat((current, new_vals), dim=0))

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def log_training_step(
            self,
            tour_costs: torch.Tensor,
            baseline_tour_costs: torch.Tensor,
            loss: torch.Tensor | float,
    ) -> None:
        self._append("epoch_training_tour_costs", tour_costs.detach())
        self._append("epoch_training_baseline_tour_costs", baseline_tour_costs.detach())

        # ensure loss is a tensor
        loss_tensor = (
            loss.detach() if torch.is_tensor(loss) else torch.tensor(loss, device=self.device)
        )
        self._append("epoch_training_loss", loss_tensor)

    @torch.no_grad()
    def log_validation_step(
            self,
            tour_costs: torch.Tensor,
            baseline_tour_costs: torch.Tensor,
            normalized_tour_costs: torch.Tensor,
            normalized_baseline_tour_costs: torch.Tensor,
    ) -> None:
        self._append("epoch_validation_tour_costs", tour_costs.detach())
        self._append("validation_baseline_tour_costs", baseline_tour_costs.detach())
        self._append("epoch_validation_tour_costs_normalized", normalized_tour_costs.detach())
        self._append(
            "validation_baseline_tour_costs_normalized", normalized_baseline_tour_costs.detach()
        )

    # --------------------------------------------------------------------- #
    #  Reset helpers
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        """Reset *all* stored stats (training *and* validation)."""
        self._init_tensors()

    def reset_training_epoch(self) -> None:
        """Reset only the per‑epoch training statistics."""
        attrs = [
            "epoch_training_tour_costs",
            "epoch_training_baseline_tour_costs",
            "epoch_training_loss",
        ]
        for a in attrs:
            setattr(self, a, torch.empty(0, device=self.device))

    def reset_validation_epoch(self) -> None:
        """Reset only the per‑epoch validation statistics."""
        attrs = [
            "epoch_validation_tour_costs",
            "validation_baseline_tour_costs",
            "epoch_validation_tour_costs_normalized",
            "validation_baseline_tour_costs_normalized",
        ]
        for a in attrs:
            setattr(self, a, torch.empty(0, device=self.device))


import itertools
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

# adapted from https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py

class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def fmt(t: object, print_stats=False) -> Lit | object:
    if isinstance(t, torch.Tensor):
        s = f"torch.tensor(..., size={tuple(t.shape)}, dtype={t.dtype}, device='{t.device}')"
        if print_stats:
            s += f" [with stats min={t.min()}, max={t.max()}, mean={t.mean()}]"
        return Lit(s)
    else:
        return t


class NaNErrorMode(TorchDispatchMode):
    def __init__(self, enabled=True, raise_error=False, print_stats=True, print_nan_index=False):
        super().__init__()
        self.enabled = enabled
        # warning or error
        self.raise_error = raise_error
        # print min/max/mean stats
        self.print_stats = print_stats
        # print indices of invalid values in output
        self.print_nan_index = print_nan_index

    def __torch_dispatch__(self, func, types, args, kwargs):
        out = func(*args, **kwargs)
        if self.enabled:
            if isinstance(out, torch.Tensor):
                if not torch.isfinite(out).all():
                    # fmt_partial = partial(fmt, self.print_stats)
                    fmt_lambda = lambda t: fmt(t, self.print_stats)
                    fmt_args = ", ".join(
                        itertools.chain(
                            (repr(tree_map(fmt_lambda, a)) for a in args),
                            (
                                f"{k}={tree_map(fmt_lambda, v)}"
                                for k, v in kwargs.items()
                            ),
                        )
                    )
                    msg = f"NaN outputs in out = {func}({fmt_args})"
                    if self.print_nan_index:
                        msg += f"\nInvalid values detected at:\n{(~out.isfinite()).nonzero()}"
                    if self.raise_error:
                        raise RuntimeError(msg)
                    else:
                        warnings.warn(msg)

        return out

## warning example
# model = models.resnet18()
#
# for i in range(1000):
#     # randomly set weights to NaNs to trigger warning / error
#     if torch.rand(1) < 0.05:
#         name, param = random.choice(list(dict(model.named_parameters()).items()))
#
#         print(f"setting first weight value of {name} to NaN")
#         with torch.no_grad():
#             param.view(-1)[0].copy_(torch.tensor(float("NaN")))
#
#     with NaNErrorMode(
#         enabled=True, raise_error=False, print_stats=True, print_nan_index=False
#     ):
#         out = model(torch.randn(1, 3, 224, 224))
#     print(f"iter: {i}, out.sum: {out.sum()}")
#     if not torch.isfinite(out).all():
#         break
