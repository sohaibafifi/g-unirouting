import concurrent.futures
import math
import time
from abc import ABC, abstractmethod

import numpy as np
import pyvrp
import torch
from pyvrp import ProblemData, VehicleType
from tqdm import tqdm


class Solver(ABC):

    def solve(self, loader, parallel=False):
        start_time = time.time()
        all_costs = []
        all_solutions = []
        for batch_id, batch in enumerate((pbar := tqdm(loader))):
            x = [x.cpu() for x in batch]
            costs, solutions = self.evaluate(*x, parallel=parallel)
            all_costs += costs.tolist()
            all_solutions += solutions.tolist()
            pbar.set_postfix(cost=np.mean(all_costs))
        return np.mean(all_costs), np.min(all_costs), time.time() - start_time, all_solutions

    def evaluate(self, node_features, global_features, parallel=False):
        n_instances = node_features.size(0)
        results = []
        solutions = []
        if not parallel:
            for batch_index in range(n_instances):
                tour_length, routes = self.process_instance(node_features[batch_index], global_features[batch_index])
                results.append(tour_length)
                solutions.append(routes)

        else:

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_instances) as executor:
                # Create a list of futures
                futures = [executor.submit(self.process_instance, node_features[index], global_features[index])
                           for index in range(n_instances)]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        tour_length, routes = future.result(timeout=10)
                    except concurrent.futures.TimeoutError:
                        print("Timeout occurred for one of the instances.")
                        tour_length = float('inf')
                    results.append(tour_length)
                    if tour_length == float('inf'):
                        solutions.append([0])
                    else:
                        solutions.append(routes)

        # pad solutions to the same length
        max_length = max(len(route) for route in solutions)
        for i in range(len(solutions)):
            if len(solutions[i]) < max_length:
                solutions[i] = np.concatenate([solutions[i], np.full(max_length - len(solutions[i]), 0)])
            else:
                solutions[i] = solutions[i][:max_length]
        return torch.tensor(results), torch.tensor(np.array(solutions))

    @abstractmethod
    def process_instance(self, node_features, global_features):
        raise NotImplementedError


class NNSolver(Solver):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def solve(self, loader, parallel=False):
        return self.model.test(loader)


class PySolver(Solver):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def process_instance(self, node_features, global_features):
        scale_factor = 100000.0
        import numpy
        from pyvrp import Model
        from pyvrp.constants import MAX_VALUE
        node_features = torch.nan_to_num(node_features, posinf=MAX_VALUE / scale_factor).cpu().numpy()
        global_features = torch.nan_to_num(global_features, posinf=MAX_VALUE / scale_factor).cpu().numpy()

        locations = node_features[:, :2]
        capacity = global_features[0]
        open_route = global_features[1]
        mixed_backhauls = global_features[2]
        max_distance = global_features[3]
        max_duration = global_features[4]

        demands_lh = node_features[:, 2]
        demands_bh = node_features[:, 3]

        model = Model()
        depot = model.add_depot(x=int(locations[0][0] * scale_factor),
                                y=int(locations[0][1] * scale_factor))
        vehicle_type = VehicleType(
            num_available=len(locations) - 1,
            tw_early=0,
            max_distance=int(max_distance * scale_factor),
            capacity=[int(scale_factor * capacity)],
            tw_late=int(min(max_duration, 1e10) * scale_factor),
            # max_duration=int(max_duration  * scale_factor),
            start_depot=0,
            end_depot=0,
        )

        clients = [
            model.add_client(
                x=int(locations[idx][0] * scale_factor),
                y=int(locations[idx][1] * scale_factor),
                delivery=math.ceil(node_features[idx][2] * scale_factor),
                pickup=math.ceil(node_features[idx][3] * scale_factor),
                tw_early=math.ceil(node_features[idx][4] * scale_factor),
                tw_late=math.floor(node_features[idx][5] * scale_factor),
                service_duration=math.ceil(node_features[idx][6] * scale_factor),
                required=True
            )
            for idx in range(1, len(node_features))
        ]
        _locations = [depot] + clients
        deltas = [[0 for _ in range(len(_locations))] for _ in range(len(_locations))]
        for frm_idx, frm in enumerate(_locations):
            for to_idx, to in enumerate(_locations):
                distance = math.sqrt(
                    (locations[frm_idx, 0] - locations[to_idx, 0]) ** 2 + (
                                locations[frm_idx, 1] - locations[to_idx, 1]) ** 2
                )
                distance = math.ceil(distance * scale_factor) if frm_idx != to_idx else 0
                if open_route and to_idx == 0:
                    distance = 0

                if not mixed_backhauls:
                    #  linehauls must be served before backhauls.
                    if demands_bh[frm_idx] > 0 and demands_lh[to_idx] > 0:
                        distance = MAX_VALUE

                deltas[frm_idx][to_idx] = distance
        deltas = numpy.array(deltas)
        self.data = ProblemData(clients, [depot], [vehicle_type], [deltas], [deltas])

        class DeltaNoImprovement:
            """
            Criterion that stops if the best solution has not been improved for a fixed
            number of iterations. The criterion is based on the delta between the best
            solution and the current best solution.

            Parameters
            ----------
            max_iterations
                The maximum number of non-improving iterations.
            delta
                The minimum improvement required to reset the counter.
            """

            def __init__(self, max_iterations: int, delta: int, max_runtime: float):
                if max_iterations < 0:
                    raise ValueError("max_iterations < 0 not understood.")

                if max_runtime < 0:
                    raise ValueError("max_runtime < 0 not understood.")

                self._max_runtime = max_runtime
                self._start_runtime: float | None = None

                self._max_iterations = max_iterations
                self._target: float | None = None
                self._counter = 0
                self.delta = delta

            def __call__(self, best_cost: float) -> bool:
                if self._target is None or best_cost < self._target - self.delta:
                    self._target = best_cost
                    self._counter = 0
                else:
                    self._counter += 1

                if self._start_runtime is None:
                    self._start_runtime = time.perf_counter()

                return time.perf_counter() - self._start_runtime > self._max_runtime or self._counter >= self._max_iterations

        self.res = pyvrp.solve(self.data, DeltaNoImprovement(301, delta=int(scale_factor * 0.01), max_runtime=5),
                               seed=self.config.seed)
        solution = self.res.best
        print('---------------------------------')
        print("solution.is_feasible()", solution.is_feasible())
        print(solution)
        deltas = self.data.distance_matrix(0)
        # print("deltas", deltas)
        total_distance = 0.0
        for route in solution.routes():
            print("route", route)

            deliveries = 0
            pickups = 0
            distance = deltas[0, route.visits()[0]]
            for i in range(len(route.visits()) - 1):
                distance += deltas[route.visits()[i], route.visits()[i + 1]]
            distance += deltas[route.visits()[-1], 0]
            total_distance += distance
            print("distance", route.distance() / scale_factor, distance / scale_factor)
            for visit in route.visits():
                deliveries += node_features[visit][2]
                print("visit", visit, node_features[visit][2])
                pickups += node_features[visit][3]
            print("deliveries", route.delivery()[0] / scale_factor, deliveries)
            assert math.fabs((route.delivery()[
                                  0] / scale_factor) - deliveries) < 1e-3, f"route.delivery()[0] {route.delivery()[0] / scale_factor} != deliveries {deliveries} ({math.fabs((route.delivery()[0] / scale_factor) - deliveries)})"
            assert deliveries <= 1, f"deliveries {deliveries} > 1"
        print("total_distance", self.res.cost(), total_distance / scale_factor)

        # plot_result(self.res, self.data)
        # import matplotlib.pyplot as plt
        # plt.show()
        sequence = [0]
        for route in solution.routes():
            for visit in route.visits():
                sequence.append(visit)
            sequence += [0]
        print("sequence", sequence)
        print('---------------------------------')

        return self.res.cost() / scale_factor if self.res.is_feasible() else float('inf'), np.array(sequence)
