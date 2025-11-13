import copy
import math
import os

import numpy
import torch
from torch.utils.data import Dataset


class MTVRPDataset(Dataset):
    """
    A dataset class that generates multi-task VRP (MTVRP) instances
    following RL4CO's MTVRP generator logic in a single vectorized call.
    Generates all `num_samples` at once (no batching).
    """

    def __init__(
            self,
            graph_size,
            num_samples: int | None,
            min_loc: float = 0.0,
            max_loc: float = 1.0,
            capacity: float | None = None,
            min_demand: int = 1,
            max_demand: int = 10,
            min_backhaul: int = 1,
            max_backhaul: int = 10,
            scale_demand: bool = True,
            max_time: float = 4.6,
            backhaul_ratio: float = 0.2,
            max_distance_limit: float = 2.8,
            variant='mtvrp',
            use_combinations: bool = True,
            device: str = "cpu",
    ):

        super().__init__()
        self.num_samples = num_samples
        self.device = device
        self.pyvrp: dict[str, torch.Tensor | None | float] = {
            'actions': None,
            'cost': None,
            'cpu': None,
        }

        if num_samples is None:
            return
        self.num_loc = graph_size - 1  # total: 1 depot + (graph_size - 1) nodes
        self.min_loc = min_loc
        self.max_loc = max_loc

        # Decide capacity
        if capacity is None:
            capacity = self.get_vehicle_capacity(self.num_loc)
        self.capacity = float(capacity)

        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.max_time = max_time
        self.backhaul_ratio = backhaul_ratio
        self.max_distance_limit = max_distance_limit

        # Handle variant preset
        variant = variant.lower()
        if variant == 'mtvrp':
            self.variant_probs = {
                "O": 0.5,
                "TW": 0.5,
                "L": 0.5,
                "B": 0.5,
                "M": 0.5,
            }
        else:
            self.variant_probs = {
                "O": float('o' in variant),
                "TW": float('tw' in variant),
                "L": float('l' in variant),
                "B": float('b' in variant),
                "M": float('m' in variant),
            }
        self.use_combinations = use_combinations

        # Generate everything at once
        data = self._generate_batch(num_samples)
        data = self._subsample_each(data)
        self.node_features = torch.cat([
            data["locs"],
            data["demand_linehaul"].unsqueeze(-1),
            data["demand_backhaul"].unsqueeze(-1),
            data["time_windows"],
            data["service_time"].unsqueeze(-1)
        ], dim=-1)
        self.global_features = torch.cat([
            data["vehicle_capacity"],
            data["open_route"].float(),
            data["mixed_backhaul"].float(),
            data["distance_limit"],
            data["time_windows"][:, 0, 1].unsqueeze(1),
            data["demand_backhaul"].unsqueeze(-1).sum(dim=1) > 0,
        ], dim=-1)

    def cpu(self):
        self.node_features = self.node_features.cpu()
        self.global_features = self.global_features.cpu()
        return self

    def to(self, device):
        self.node_features = self.node_features.to(device)
        self.global_features = self.global_features.to(device)
        self.device = device
        return self

    def get_size(self):
        return self.node_features.element_size() * self.node_features.nelement() + self.global_features.element_size() * self.global_features.nelement()

    def __len__(self):
        return self.num_samples

    def __getitems__(self, idx):
        return self.node_features[idx], self.global_features[idx]

    @staticmethod
    def collate_fn(batch):
        return batch

    def __getitem__(self, idx):
        return self.node_features[idx], self.global_features[idx]

    def get_instance_features(self):
        return self.get_batch_features((self.node_features, self.global_features))

    def save(self, path):
        extension = path.split('.')[-1]
        if extension == 'npz':
            node_features = self.node_features.cpu().numpy()
            global_features = self.global_features.cpu().numpy()

            numpy.savez(path, node_features=node_features, global_features=global_features)
        else:
            dataset = copy.deepcopy(self)
            dataset.node_features = dataset.node_features.cpu()
            dataset.global_features = dataset.global_features.cpu()
            torch.save(self, path)

    @staticmethod
    def permute_customers(inputs):
        """
        Randomly permutes the *customer* nodes (1..N) for each batch instance,
        keeping the depot (index 0) fixed at the start of the node_features.

        inputs:
            node_features: [B, N, D] or [B, N+1, D]
                - where node_features[:, 0, :] is the depot
                  and node_features[:, 1:, :] are the customers
            global_features: [B, G] (whatever shape your global features have)

        returns:
            A single augmented sample (nf_perm, global_features)
        """
        node_features, global_features = inputs
        B, total_nodes, D = node_features.size()  # e.g. total_nodes = N+1 if depot + N customers

        # Split depot vs. customers
        depot = node_features[:, 0:1, :]  # [B, 1, D]
        customers = node_features[:, 1:, :]  # [B, N, D]

        # We will create a random permutation of range(0, N) for each batch row
        # (This is the customer indices in that row.)
        device = node_features.device
        N = customers.size(1)

        # We'll gather using an index array of shape [B, N],
        # where each row is a permutation of [0..N-1].
        perm_indices = torch.empty((B, N), dtype=torch.long, device=device)
        for b in range(B):
            perm_indices[b] = torch.randperm(N, device=device)

        # Now gather from 'customers' using this permutation
        # We'll do something like: customers[b, perm_indices[b], :]
        # The advanced way to do this in one shot is with torch.gather or torch.take_along_dim,
        # but a simple approach is a loop or using `torch.stack`.
        # A fully vectorized way with gather might look like:
        perm_indices_expanded = perm_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        customers_perm = torch.gather(customers, dim=1, index=perm_indices_expanded)

        # Reassemble depot + permuted customers
        nf_perm = torch.cat([depot, customers_perm], dim=1)  # [B, N+1, D] again

        # Return as a single augmented sample in a list
        return [(nf_perm, global_features)]

    @staticmethod
    def augment(inputs,
                *,
                n_random_rot: int = 16,
                n_random_shift: int = 16,
                n_random_scale: int = 0,  # this is not conservative, (keep s)
                n_permutations: int = 0,
                shift_max: float = 0.2,
                scale_range: tuple = (0.7, 1.3)):
        """
        Loss‑preserving data augmentation.

        Parameters
        ----------
        n_random_rot     : how many uniform‑SO(2) rotations to add
        n_random_shift   : how many uniform translations (in [-shift_max, +shift_max]) to add
        n_random_scale   : how many uniform scalings in `scale_range` to add
        n_permutations   : how many customer‑index permutations to add
        shift_max        : L‑inf bound for translations (post‑shift coords are clamped to [0,1])
        scale_range      : (low, high) multiplicative bounds for uniform scaling
        """
        node_features, global_features = inputs
        B, Np1, D = node_features.shape  # [B, N+1, 7]
        device = node_features.device

        # ------------------------------------------------------------------ #
        # Split coordinates vs. non‑spatial features                          #
        # ------------------------------------------------------------------ #
        coords = node_features[..., :2]  # [B, N+1, 2]
        rest = node_features[..., 2:]  # [B, N+1, D-2]

        augmented = []

        # ------------------------------------------------------------------ #
        # 1) Dihedral‑8                                                      #
        # ------------------------------------------------------------------ #
        x, y = coords[..., 0], coords[..., 1]
        dihedral_coords = [
            coords,  # identity
            torch.stack([-y, x], dim=-1),  # rot  90
            -coords,  # rot 180
            torch.stack([y, -x], dim=-1),  # rot 270
            torch.stack([x, -y], dim=-1),  # reflect x‑axis
            torch.stack([-x, y], dim=-1),  # reflect y‑axis
            torch.stack([y, x], dim=-1),  # reflect line y=x
            torch.stack([-y, -x], dim=-1)  # reflect line y=-x
        ]
        for c in dihedral_coords:
            augmented.append((torch.cat([c, rest], dim=-1), global_features))

        # ------------------------------------------------------------------ #
        # 2) Random SO(2) rotations                                          #
        # ------------------------------------------------------------------ #
        for _ in range(n_random_rot):
            theta = 2 * math.pi * torch.rand(1, device=device)
            cos_t, sin_t = theta.cos(), theta.sin()
            R = torch.tensor([[cos_t, -sin_t],
                              [sin_t, cos_t]], device=device).view(1, 1, 2, 2)  # [1,1,2,2]
            c_rot = (coords.unsqueeze(-2) @ R).squeeze(-2)  # [B,N+1,2]
            augmented.append((torch.cat([c_rot, rest], dim=-1), global_features))

        # ------------------------------------------------------------------ #
        # 3) Random translations                                             #
        # ------------------------------------------------------------------ #
        for _ in range(n_random_shift):
            shift = (2 * torch.rand((1, 1, 2), device=device) - 1) * shift_max
            # c_shift = torch.clamp(coords + shift, 0.0, 1.0)
            c_shift = coords + shift
            augmented.append((torch.cat([c_shift, rest], dim=-1), global_features))

        # ------------------------------------------------------------------ #
        # 4) Uniform scalings (coords + distance_limit)                      #
        # ------------------------------------------------------------------ #
        dist_lim_idx = 3  # distance_limit  in global_features
        depot_tw_end_idx = 4  # depot TW_end    in global_features
        for _ in range(n_random_scale):
            s_low, s_high = scale_range
            s = (s_high - s_low) * torch.rand(1, device=device) + s_low

            # 1) coordinates
            # c_scale = torch.clamp(coords * s, 0.0, 1.0)
            c_scale = coords * s

            # 2) node-level time data  (rest = [d_L, d_B, tw_s, tw_e, svc])
            rest_scale = rest.clone()
            rest_scale[..., 2:4] *= s  # tw_start, tw_end
            rest_scale[..., 4] *= s  # service_time

            # 3) global limits
            g_scale = global_features.clone()
            g_scale[:, dist_lim_idx] *= s  # distance_limit
            g_scale[:, depot_tw_end_idx] *= s  # depot TW_end

            augmented.append((torch.cat([c_scale, rest_scale], dim=-1), g_scale))

        # ------------------------------------------------------------------ #
        # 5) Customer‑index permutations                                     #
        # ------------------------------------------------------------------ #
        for _ in range(n_permutations):
            nf_perm, _ = MTVRPDataset.permute_customers((node_features, global_features))[0]
            augmented.append((nf_perm, global_features))
        return augmented

    @staticmethod
    def load(path) -> 'MTVRPDataset':
        extension = path.split('.')[-1]
        if extension == 'npz':
            data = numpy.load(path)
            data = dict(data)
            node_features = torch.tensor(data['node_features'])
            global_features = torch.tensor(data['global_features'])
            dataset = MTVRPDataset(graph_size=node_features.size(1), num_samples=None)
            dataset.node_features = node_features.to(dataset.device)
            dataset.global_features = global_features.to(dataset.device)
            dataset.num_samples = node_features.size(0)

            solution_path = path.replace('test.npz', 'test_sol.npz')
            if os.path.exists(solution_path):
                solution = numpy.load(solution_path)
                solution = dict(solution)
                dataset.pyvrp['actions'] = torch.tensor(solution['actions'])
                dataset.pyvrp['cost'] = torch.tensor(solution['cost'])
                dataset.pyvrp['cpu'] = solution['cpu']

            return dataset
        return torch.load(path, map_location='cpu', weights_only=False)

    @staticmethod
    def get_batch_features(batch):
        """
        I1 Number of customers
        I2 Number of routes (approx. or derived)
        I3 Degree of capacity utilisation
        I4 Average distance between each pair of customers
        I5 Standard deviation of the pairwise distance between customers
        I6 Average distance from customers to the depot
        I7 Standard deviation of the distance from customers to the depot
        I8 Standard deviation of the radians of customers towards the depot
        I9: backhaul ratio (demand_backhaul / total_demand)
        I10: Average time window length
        I11: Standard deviation of time window length
        I12: Average service time
        I13: Standard deviation of service time
        TODO : filter the efficiency features
        ref: 10.1016/j.cor.2018.02.007
        """
        node_features, global_features = batch
        # node_features shape = [batch_size, num_loc+1, >2], e.g. [B, N+1, 7]
        # Extract coords => [batch_size, N+1, 2]
        locs = node_features[..., :2]
        batch_size = locs.size(0)
        num_loc = locs.size(1) - 1

        # Separate depot from customers
        depot = locs[:, 0:1]  # [B, 1, 2]
        customers = locs[:, 1:]  # [B, N, 2]

        # Demands for customers => [B, N]
        demand_customers = node_features[:, 1:, 2:4].sum(dim=-1)

        # ================
        # 1) Number of customers
        # ================
        # Create [batch_size] float tensor = num_loc repeated
        num_customers = torch.full(
            (batch_size,),
            float(num_loc),
            dtype=torch.float,
            device=locs.device
        )
        # shape => [B]

        # ================
        # 2) Number of routes (approx.)
        # ================
        # This depends on how you want to define "number of routes."
        # For instance, a rough guess might be: total_demand // capacity # TODO: use better LB
        capacity = global_features[:, 0]  # [B]
        total_demand = demand_customers.sum(dim=-1)  # [B]
        routes = torch.ceil(total_demand / capacity)  # [B]

        # ================
        # 3) Degree of capacity utilisation
        # ================
        # E.g. sum of all demands / (capacity * # of routes) or simply sum/ capacity:
        deg_cap = total_demand / capacity
        # shape => [B]

        # ================
        # 4) Average distance between customers
        # 5) Standard deviation of pairwise distance
        # ================
        dist = torch.cdist(customers, customers)  # [B, N, N]
        dist_avg = dist.mean(dim=[-2, -1])  # reduce over (N, N) => [B]
        dist_std = dist.std(dim=[-2, -1])  # [B]

        # ================
        # 6) Average distance from customers to depot
        # 7) Standard deviation distance from customers to depot (if open routes the distance is 0)
        # ================
        dist_depot = torch.cdist(customers, depot)  # [B, N, 1]
        open_mask = global_features[:, 1] > 0.5  # open_route
        dist_depot[open_mask] = 0.0  # Set to 0 if open route
        # Flatten out last dimension:
        dist_depot = dist_depot.squeeze(-1)  # [B, N]
        dist_depot_avg = dist_depot.mean(dim=-1)  # [B]
        dist_depot_std = dist_depot.std(dim=-1)  # [B]

        # ================
        # 8) Standard deviation of radians of customers around depot
        # ================
        rad = torch.atan2(customers[:, :, 1], customers[:, :, 0])  # rad => [B, N]
        rad_std = rad.std(dim=-1)  # => [B]

        # ================
        # 9) Backhaul ratio
        # ================
        ratio_backhaul = node_features[:, 1:, 3].sum(dim=1) / demand_customers.sum(dim=1)

        # ================
        # 10) Average time window length
        # 11) Standard deviation of time window length
        # ================
        time_windows = node_features[:, 1:, 4:6]  # [B, N, 2]
        tw_length = time_windows[:, :, 1] - time_windows[:, :, 0]  # [B, N]
        tw_avg = tw_length.mean(dim=-1)  # [B]
        tw_std = tw_length.std(dim=-1)  # [B]

        # ================
        # 12) Average service time
        # 13) Standard deviation of service time
        # ================
        svc_time = node_features[:, 1:, 6]  # [B, N]
        svc_avg = svc_time.mean(dim=-1)  # [B]
        svc_std = svc_time.std(dim=-1)  # [B]

        instance_features = torch.stack([
            num_customers,
            routes,
            deg_cap,
            dist_avg,
            dist_std,
            dist_depot_avg,
            dist_depot_std,
            rad_std,
            ratio_backhaul,
            tw_avg,
            tw_std,
            svc_avg,
            svc_std
        ], dim=-1)  # => [B, 13]

        return instance_features

    def _generate_batch(self, batch_size: int):
        """
        Vectorized logic for batch_size = num_samples (all at once).
        Returns dict of shape [batch_size, ...].
        """
        # Generate locations => [batch_size, N+1, 2]
        locs = self._generate_locations(batch_size)

        # Vehicle capacity => [batch_size, 1]
        capacity_tensor = torch.full((batch_size, 1), self.capacity, dtype=torch.float32, device=self.device)

        # Demands => linehaul/backhaul, [batch_size, N+1] (0 for depot)
        d_linehaul, d_backhaul = self._generate_demands(batch_size)
        zero_depot = torch.zeros((batch_size, 1), device=self.device)
        d_linehaul = torch.cat([zero_depot, d_linehaul], dim=1)
        d_backhaul = torch.cat([zero_depot, d_backhaul], dim=1)

        # Open route => default True
        open_route = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)

        # Time windows => [batch_size, N+1, 2], service_time => [batch_size, N+1]
        time_windows, service_time = self._generate_time_windows(locs)

        # Distance limit => [batch_size, 1]
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 0.1
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )
        distance_limit_tensor = torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]
        # Scale demands => effectively capacity=1 if self.scale_demand
        capacity_original = capacity_tensor.clone()
        if self.scale_demand:
            d_linehaul /= capacity_tensor
            d_backhaul /= capacity_tensor
            capacity_tensor = capacity_tensor / capacity_tensor  # => 1

        # Mixed backhaul => default True, modify if needed
        mixed_backhaul = torch.ones_like(open_route)

        speed = torch.ones_like(open_route)

        # Package everything
        data = {
            "locs": locs,  # [batch_size, N+1, 2]
            "demand_linehaul": d_linehaul,  # [batch_size, N+1]
            "demand_backhaul": d_backhaul,  # [batch_size, N+1]
            "time_windows": time_windows,  # [batch_size, N+1, 2]
            "service_time": service_time,  # [batch_size, N+1]
            "vehicle_capacity": capacity_tensor,  # [batch_size, 1]
            "capacity_original": capacity_original,  # [batch_size, 1]
            "open_route": open_route,  # [batch_size, 1] bool
            "mixed_backhaul": mixed_backhaul,  # [batch_size, 1] bool
            "distance_limit": distance_limit_tensor,  # [batch_size, 1]
            "speed": speed,  # [batch_size, 1]
        }
        return data

    def _subsample_each(self, data: dict):
        """
        For each sample (row) in data, randomly decide which features to keep.
        Then remove or reset them if not kept, but do it in a batched way.
        """
        bs = data["locs"].shape[0]

        locs = data["locs"]
        d_linehaul = data["demand_linehaul"]
        d_backhaul = data["demand_backhaul"]
        tw = data["time_windows"]
        svc = data["service_time"]
        dist_lim = data["distance_limit"]
        open_r = data["open_route"]
        mixed_backhaul = data["mixed_backhaul"]

        p_open = self.variant_probs["O"]
        p_tw = self.variant_probs["TW"]
        p_limit = self.variant_probs["L"]
        p_bh = self.variant_probs["B"]
        p_mixed = self.variant_probs["M"]

        if self.use_combinations:
            # -- Generate 'keep' decisions for each sample independently but in one batch --
            keep_open = (torch.rand(bs, device=self.device) < p_open)
            keep_tw = (torch.rand(bs, device=self.device) < p_tw)
            keep_limit = (torch.rand(bs, device=self.device) < p_limit)
            keep_bh = (torch.rand(bs, device=self.device) < p_bh)

            # keep_mixed is only relevant if keep_bh == True
            random_mixed = (torch.rand(bs, device=self.device) < p_mixed)
            keep_mixed = keep_bh & random_mixed

        else:
            # -- We pick exactly one variant per sample (open, tw, limit, or bh) --
            p = torch.tensor([p_open, p_tw, p_limit, p_bh], device=self.device)
            # Normalize if necessary (depends on your code’s assumptions):
            # p = p / p.sum()
            # Sample a single "choice" per row from [0..3].
            # shape: [bs]
            choice = torch.distributions.Categorical(p).sample((bs,))

            keep_open = (choice == 0)
            keep_tw = (choice == 1)
            keep_limit = (choice == 2)
            keep_bh = (choice == 3)

            # keep_mixed only relevant for samples that are keep_bh
            random_mixed = (torch.rand(bs, device=self.device) < p_mixed)
            keep_mixed = keep_bh & random_mixed

        # -- Now apply the decisions in a vectorized manner --

        # open route
        open_r[~keep_open, 0] = False

        # time windows and service times
        # If not keeping TW, set tw to [0, inf] and service time to 0
        tw[~keep_tw, :, 0] = 0.0
        tw[~keep_tw, :, 1] = float('inf')
        svc[~keep_tw, :] = 0.0

        # distance limit
        dist_lim[~keep_limit, 0] = float('inf')  # or float('inf') if desired

        # backhaul decisions
        # For rows that do NOT keep BH:
        not_keep_bh = ~keep_bh
        d_linehaul[not_keep_bh, :] += d_backhaul[not_keep_bh, :]
        d_backhaul[not_keep_bh, :] = 0.0
        mixed_backhaul[not_keep_bh, 0] = False

        # If BH is kept, decide if it is a "mixed" scenario
        # For BH rows that do NOT keep mixed
        no_mixed = keep_bh & ~keep_mixed
        mixed_backhaul[no_mixed, 0] = False

        return data

    @staticmethod
    def get_vehicle_capacity(num_loc: int) -> float:
        """
        Example capacity logic from RL4CO
        """
        if num_loc > 1000:
            extra_cap = 1000 // 5 + int((num_loc - 1000) // 33.3)
        elif num_loc > 20:
            extra_cap = num_loc // 5
        else:
            extra_cap = 0
        return 30 + extra_cap

    def _generate_locations(self, batch_size: int) -> torch.Tensor:
        """
        Returns locations of shape [batch_size, N+1, 2],
        where index 0 is the depot for each instance.

        We do a fully batched approach for all:
          - random
          - clustered (with uniform #clusters)
          - mixed (half cluster, half random)
        """
        device = self.device
        graph_size = self.num_loc + 1

        locations = torch.rand(
            (batch_size, graph_size, 2), device=device
        ) * (self.max_loc - self.min_loc) + self.min_loc

        return locations  # [B, graph_size, 2]

    def _generate_demands(self, batch_size: int):
        """
        Vectorized linehaul/backhaul demands => shape [batch_size, N].
        We'll add depot=0 outside of this.
        """
        # [batch_size, N]
        linehaul_raw = torch.rand((batch_size, self.num_loc), device=self.device)
        linehaul_raw = linehaul_raw * (self.max_demand - self.min_demand) + (self.min_demand - 1)
        linehaul_demand = (linehaul_raw.int() + 1).float()

        backhaul_raw = torch.rand((batch_size, self.num_loc), device=self.device)
        backhaul_raw = backhaul_raw * (self.max_backhaul - self.min_backhaul) + (self.min_backhaul - 1)
        backhaul_demand = (backhaul_raw.int() + 1).float()

        # Decide linehaul/backhaul
        is_linehaul = torch.rand((batch_size, self.num_loc), device=self.device) > self.backhaul_ratio
        linehaul_demand = linehaul_demand * is_linehaul
        backhaul_demand = backhaul_demand * (~is_linehaul)

        return linehaul_demand, backhaul_demand

    def _generate_time_windows(self, locs: torch.Tensor):
        """
        Vectorized time window & service time generation:
          locs => [batch_size, N+1, 2]
        Return => (time_windows [batch_size, N+1, 2], service_time [batch_size, N+1])
        """
        bs = locs.size(0)
        n_plus_1 = locs.size(1)
        n_loc = n_plus_1 - 1

        a, b, c = 0.15, 0.18, 0.2
        service_time_nodes = a + (b - a) * torch.rand((bs, n_loc), device=self.device)
        tw_length = b + (c - b) * torch.rand((bs, n_loc), device=self.device)

        depot = locs[:, 0:1, :]  # [bs, 1, 2]
        nodes = locs[:, 1:, :]  # [bs, N, 2]
        d_0i = (depot - nodes).norm(dim=-1)  # [bs, N]

        h_max = (self.max_time - service_time_nodes - tw_length) / d_0i - 1
        tw_start = (1 + (h_max - 1) * torch.rand((bs, n_loc), device=self.device)) * d_0i
        tw_end = tw_start + tw_length
        tw_end = torch.max(tw_end, d_0i)

        # Depot's time window => [0, inf] (or [0, self.max_time])
        depot_start = torch.zeros((bs, 1), device=self.device)
        depot_end = torch.full((bs, 1), self.max_time, device=self.device)
        tw_start_full = torch.cat([depot_start, tw_start], dim=1)  # [bs, N+1]
        tw_end_full = torch.cat([depot_end, tw_end], dim=1)  # [bs, N+1]
        time_windows = torch.stack([tw_start_full, tw_end_full], dim=-1)  # [bs, N+1, 2]

        service_time_full = torch.cat([
            torch.zeros((bs, 1), device=self.device),
            service_time_nodes
        ], dim=1)  # [bs, N+1]

        return time_windows, service_time_full

    def to_data(self):
        # 'locs',
        # 'demand_linehaul',
        # 'vehicle_capacity',
        # 'speed',
        # 'num_depots',
        # 'demand_backhaul',
        # 'backhaul_class', 1 if b, 2 if m, else 0
        # 'open_route',
        # 'time_windows',
        # 'service_time',
        # 'distance_limit',
        # REF : backhaul_class: which type of backhaul to use:
        #                 0: no backhaul (note: we don't use 0 since we can efficiently generate CVRP by just transforming backhauls to linehauls)
        #                 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        #                 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        #
        backhaul_class = torch.ones_like(self.global_features[:, 1])
        # mobackhaul_class[self.node_features[:, :, 3].sum(dim=1) > 0] = 1
        backhaul_class[self.global_features[:, 2] > 0] = 2

        # locs: torch.Size([1000, 51, 2])
        # demand_linehaul: torch.Size([1000, 50])
        # vehicle_capacity: torch.Size([1000, 1])
        # speed: torch.Size([1000, 1])
        # num_depots: torch.Size([1000, 1])
        # demand_backhaul: torch.Size([1000, 50])
        # backhaul_class: torch.Size([1000, 1])
        # open_route: torch.Size([1000, 1])
        # time_windows: torch.Size([1000, 51, 2])
        # service_time: torch.Size([1000, 51])
        # distance_limit: torch.Size([1000, 1])

        data = {
            "locs": self.node_features[:, :, :2],
            "demand_linehaul": self.node_features[:, 1:, 2],
            "vehicle_capacity": self.global_features[:, 0].unsqueeze(-1),
            "speed": torch.ones_like(self.global_features[:, 1]).unsqueeze(-1),
            "num_depots": torch.ones_like(self.global_features[:, 1]).unsqueeze(-1),
            "demand_backhaul": self.node_features[:, 1:, 3],
            'backhaul_class': backhaul_class.unsqueeze(-1),
            "open_route": self.global_features[:, 1].unsqueeze(-1).to(torch.bool),
            "time_windows": self.node_features[:, :, 4:6],
            "service_time": self.node_features[:, :, 6],
            "distance_limit": self.global_features[:, 3].unsqueeze(-1),
        }

        return data
