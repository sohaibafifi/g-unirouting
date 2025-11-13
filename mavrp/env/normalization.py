# Description: This file contains the CostNormalization class, which is used to normalize costs
# across different problem instances in the MTVRP dataset.
# The idea is inspired by the one used in the RL4CO/RouteFinder

import torch
from torch import Tensor
from torch import nn as nn

from .problems import MTVRP


class CostNormalization:
    def __init__(self, type="exponential", alpha: float = 0.25, epsilon: float = 1e-6) -> None:
        # Track per-variant mean, count, and sum of squared diffs (M2) for Gaussian normalization
        self.norm_vals = {variant: {"mean": 0.0, "count": 0, "M2": 0.0}
                          for variant in MTVRP.get_variants()}
        self.alpha = alpha
        self.epsilon = epsilon
        assert type in ["exponential", "cumulative", "no_norm", "gauss"], \
            "type must be 'exponential', 'cumulative' 'gauss', or 'no_norm'."
        self.type = type

    def __call__(
            self,
            data: tuple[torch.Tensor, torch.Tensor],
            costs: torch.Tensor,
            operation: str = "div"
    ) -> tuple[Tensor, Tensor]:
        """
        Normalize the given 'costs' based on each variant's mean cost
        using either division or subtraction.
        """
        if self.type == "gauss":
            # Gaussian normalization requires 'gauss' operation
            operation = "gauss"

        assert operation in ["div", "sub", "gauss"], \
            "operation must be 'div', 'sub', or 'gauss'."

        # Make copies so we don't overwrite original values
        normalized_costs = costs.clone()
        norm_vals = torch.zeros_like(costs)

        # Temporarily store new means for each variant
        new_means = {}

        for variant in self.norm_vals.keys():
            mask = self.build_mask(variant, data)
            if not mask.any():
                continue

            # If Gaussian normalization is requested, perform z-score update & normalization:
            if operation == "gauss":
                batch_costs = costs[mask]
                batch_count = int(mask.sum().item())
                batch_mean = batch_costs.mean().item()
                batch_var = batch_costs.var(unbiased=False).item()  # population variance

                # Update running mean and M2 for this variant
                self._update_running_stats(variant, batch_mean, batch_var, batch_count)

                # Retrieve updated stats
                stats = self.norm_vals[variant]
                mean = stats["mean"]
                count = stats["count"]
                m2 = stats["M2"]
                var = m2 / max(count, 1)
                std = (var + self.epsilon) ** 0.5

                # Apply z-score normalization: (cost - mean) / std
                normalized_costs[mask] = (batch_costs - mean) / std
                norm_vals[mask] = std
            else:
                # For backward-compatible 'div' or 'sub', use original logic:
                # Compute the new average cost for this variant
                new_means[variant] = costs[mask].mean().item()

                # Update the variant using a per-variant counter (mean only)
                self.update_variant(variant, new_means[variant])

                # Apply normalization (div or sub)
                normalized_costs[mask] = self.normalize_cost(
                    variant, mask, costs, operation
                )

                # Record the final (updated) mean in norm_vals
                norm_vals[mask] = self.norm_vals[variant]["mean"]

        return normalized_costs, norm_vals

    def update_variant(self, variant: str, new_val: float):
        """
        Update the mean cost for the given variant, using exponential smoothing
        *only if* its count > 0. Otherwise, initialize directly.
        """
        vdata = self.norm_vals[variant]
        count = vdata["count"]
        old_mean = vdata["mean"]

        if count == 0:
            # First time we see this variant, set the mean = new_val
            vdata["mean"] = new_val
        else:
            if self.type == "cumulative":
                # Apply cumulative normalization
                vdata["mean"] = (old_mean * count + new_val) / (count + 1)
            elif self.type == "exponential":
                # Apply exponential smoothing
                vdata["mean"] = (1 - self.alpha) * old_mean + self.alpha * new_val
            else:
                # no normalization
                vdata["mean"] = new_val

        # Increase the update counter for this variant
        vdata["count"] += 1

    def normalize_cost(
            self,
            variant: str,
            mask: torch.Tensor,
            costs: torch.Tensor,
            operation: str
    ) -> Tensor | None:
        """
        Apply the given normalization operation for a particular variant.
        """
        mean_cost = self.norm_vals[variant]["mean"]
        if operation == "div":
            return costs[mask] / (abs(mean_cost) + self.epsilon)
        elif operation == "sub":
            return costs[mask] - mean_cost
        # 'gauss' is handled directly in __call__, so this should not be reached.
        return None

    @staticmethod
    def build_mask(variant: str, data: tuple[Tensor, Tensor]) -> Tensor:
        """
        Dynamically build the boolean mask for each problem instance in 'data'
        by parsing the variant name.

        data = (node_features, global_features)
        node_features.shape = [batch_size, num_nodes, num_features_per_node]
        global_features.shape = [batch_size, num_global_features]
        """
        node_features, global_features = data

        # Precompute relevant booleans
        mask_open_route = (global_features[:, 1] == 1)
        mask_mixed_backhaul = (global_features[:, 2] == 1)
        mask_distance_limit = global_features[:, 3] < 1000
        mask_backhaul = (node_features[:, :, 3].sum(dim=1) > 0)
        mask_time_windows = (node_features[:, :, 4].max(dim=1).values > 0)

        # Parse the variant name to see which features are turned on

        # For each feature, if flags[feature] == True, we want the mask to be True.
        # If flags[feature] == False, we want the mask to be False => ~mask.
        def match(flag: bool, condition: torch.Tensor) -> torch.Tensor:
            return condition if flag else ~condition

        # Start with all True, refine step by step
        device = global_features.device
        mask = torch.ones(global_features.size(0), dtype=torch.bool, device=device)
        mask &= match(('o' in variant), mask_open_route)
        mask &= match(('b' in variant), mask_backhaul)
        mask &= match(('tw' in variant), mask_time_windows)
        mask &= match(('l' in variant), mask_distance_limit)
        # Only apply 'mixed_backhaul' if backhaul is active
        if 'b' in variant:
            mask &= match(('m' in variant), mask_mixed_backhaul)
        return mask.cpu()

    def _update_running_stats(
        self,
        variant: str,
        batch_mean: float,
        batch_var: float,
        batch_count: int,
    ):
        """
        Merge existing (mean, M2, count) for 'variant' with a new batch of size batch_count,
        which has its own mean=batch_mean and variance=batch_var.

        Uses Welford's batch update:
          new_count = old_count + batch_count
          delta = batch_mean - old_mean
          new_mean = old_mean + delta * (batch_count / new_count)
          new_M2 = old_M2 + (batch_var * batch_count)
                   + delta^2 * (old_count * batch_count / new_count)
        """
        stats = self.norm_vals[variant]
        old_count = stats["count"]
        old_mean = stats["mean"]
        old_M2 = stats["M2"]
        k = batch_count
        new_count = old_count + k

        if old_count == 0:
            # first batch for this variant
            stats["mean"] = batch_mean
            stats["M2"] = batch_var * k
            stats["count"] = k
        else:
            # 1) compute delta between batch mean and running mean
            delta = batch_mean - old_mean

            # 2) update running mean
            stats["mean"] = old_mean + delta * (k / new_count)

            # 3) update running M2 (sum of squared deviations)
            batch_M2 = batch_var * k
            combined_M2 = old_M2 + batch_M2 + delta * delta * (old_count * k / new_count)
            stats["M2"] = combined_M2

            # 4) update count
            stats["count"] = new_count


class Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str = "batch"):
        super().__init__()
        self.normalization = normalization
        cls = {
            "batch": lambda d: nn.BatchNorm1d(d, affine=True),
            "instance": lambda d: nn.InstanceNorm1d(d, affine=True),
            "scale": lambda d: ScaleNorm(d),
            "rms": lambda d: RMSNorm(d),
        }.get(normalization)
        if cls is None:
            cls = nn.Identity
        self.normalizer : nn.Module = cls(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, embed_dim)
        if self.normalization == "batch":
            batch, seq, dim = x.shape
            y : torch.Tensor = self.normalizer(x.view(-1, dim))
            return y.view(batch, seq, dim)
        elif self.normalization == "instance":
            y = self.normalizer(x.permute(0, 2, 1))
            return y.permute(0, 2, 1)
        elif self.normalization == "layer":
            mean = x.mean([1, 2], keepdim=True)
            var = x.var([1, 2], unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + 1e-5)
        elif self.normalization == "rms":
            return self.normalizer(x)
        elif self.normalization == "scale":
            return self.normalizer(x)
        else:  # identity or fallback
            return x


class EdgeNormalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str = "batch"):
        super().__init__()
        self.normalization = normalization
        cls = {
            "batch": lambda d: nn.BatchNorm1d(d, affine=True),
            "instance": lambda d: nn.InstanceNorm1d(d, affine=True),
            "scale": lambda d: ScaleNorm(d),
            "rms": lambda d: RMSNorm(d),
        }.get(normalization)
        if cls is None:
            cls = nn.Identity
        self.normalizer = cls(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, embed_dim)
        if self.normalization == "batch":
            batch_seq, dim = x.shape
            y = self.normalizer(x.view(-1, dim))
            return y.view(batch_seq, dim)
        elif self.normalization == "instance":
            y = self.normalizer(x.permute(0, 2, 1))
            return y.permute(0, 2, 1)
        elif self.normalization == "layer":
            mean = x.mean([1, 2], keepdim=True)
            var = x.var([1, 2], unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + 1e-5)
        elif self.normalization == "rms":
            return self.normalizer(x)
        elif self.normalization == "scale":
            return self.normalizer(x)
        else:  # identity or fallback
            return x


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ScaleNorm(torch.nn.Module):
    """ScaleNorm Transformers without Tears """

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
