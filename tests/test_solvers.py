import os
import sys

import lightning
import pytest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.models import TransformerModel
from mavrp.env.solvers import PySolver


@pytest.fixture
def solver_config_and_loader():
    """Fixture to create config and data loader for solver tests."""
    config = Config()
    lightning.seed_everything(config.seed)
    problem = config.get_problem()
    config.batch_size = 4
    config.graph_size = 16
    dataset = problem.dataset(
        graph_size=config.graph_size,
        num_samples=config.batch_size * 4,
        device=config.device
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    return config, loader


def test_pyvrp(solver_config_and_loader):
    """Test PySolver with PyVRP backend."""
    config, loader = solver_config_and_loader
    solver = PySolver(config)

    for batch_id, batch in enumerate((pbar := tqdm(loader))):
        x = [x.cpu() for x in batch]
        costs, solutions = solver.evaluate(*x, parallel=False)
        node_features, global_features = x
        distance_matrix = torch.cdist(node_features[:, :, :2], node_features[:, :, :2])
        open_routes = global_features[:, 1].bool()
        distance_matrix[open_routes, :, 0] = 0.0
        # Check if the solution is feasible
        for b, solution in enumerate(solutions):
            deltas = distance_matrix[b]
            print(costs[b], solution)
            # split the solution into routes using 0 as the separator
            routes = []
            current_route = []
            for node in solution:
                if node == 0:
                    if current_route:
                        routes.append(current_route)
                        current_route = []
                else:
                    current_route.append(node.item())
            print(solution, "solution", routes)
            # print routes
            total_distance = 0.0
            for route in routes:
                print("route", route)
                deliveries = 0
                pickups = 0
                distance = deltas[0, route[0]]
                for i in range(len(route) - 1):
                    distance += deltas[route[i], route[i + 1]]
                distance += deltas[route[-1], 0]
                print("distance", distance)
                total_distance += distance
                for visit in route:
                    deliveries += node_features[b][visit][2]
                    print(visit, ":", node_features[b][visit][2].item(), end=" ")
                    pickups += node_features[b][visit][3]
                print("deliveries", deliveries)
                assert deliveries <= global_features[b][0], \
                    f"Deliveries exceed capacity {deliveries} < {global_features[b][0]}"
                assert pickups <= global_features[b][0], \
                    f"Pickups exceed capacity {pickups} < {global_features[b][0]}"

            print("total_distance", costs[b].item(), total_distance)


def test_pyvrp_parallel(solver_config_and_loader):
    """Test PySolver with parallel execution."""
    config, loader = solver_config_and_loader
    solver = PySolver(config)
    avg_tl, min_tl, cpu, solutions = solver.solve(loader, parallel=True)
    assert min_tl <= avg_tl


@pytest.mark.skip(reason="Test requires specific model checkpoints")
def test_pyvrp_solution():
    """Test PyVRP solution with model checkpoints (requires specific files)."""
    config = Config.all()[0]
    lightning.seed_everything(config.seed)

    config.problem = 'VRP'
    config.graph_size = 10
    config.batch_size = 1
    problem = config.get_problem()
    dataset = problem.dataset(graph_size=config.graph_size, num_samples=config.batch_size, device=config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    solver = PySolver(config)
    batch = next(iter(loader))
    node_features, global_features = batch
    tour_length = solver.process_instance(node_features[0], global_features[0])
    solution = solver.res.best
    print("solution.is_feasible()", solution.is_feasible())
    print(solution)
    deltas = solver.data.distance_matrix(0)
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
        print("distance", route.distance() / 100000.0, distance / 100000.0)
        for visit in route.visits():
            deliveries += node_features[0][visit][2]
            pickups += node_features[0][visit][3]
        print("deliveries", route.delivery()[0] / 100000.0, deliveries)
    print("total_distance", tour_length, total_distance / 100000.0)

    # transform to actions
    actions = [0]
    for route in solution.routes():
        for visit in route.visits():
            actions.append(visit)
        actions.append(0)
    actions = torch.tensor([actions])

    #plot_instance(solver.data)
    #plot_result(solver.res, solver.data)
    #plt.show()
    for config in Config.all():
        print('__________________________________________________________')
        print(config.__repr__())
        print('__________________________________________________________')
        torch.serialization.safe_globals([Config])
        model = TransformerModel(config)
        loaded_config = config
        loaded_config.graph_size = 50
        model_path = os.path.join(config.working_dir, 'MTVRP',
                              '50',
                              repr(loaded_config),
                              'checkpoint.ckpt')


        sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../mavrp'))
        if not os.path.exists(model_path):
            print(f"Folder {model_path} does not exist")
            continue
        with torch.serialization.safe_globals([Config]):
            print('loading model from', model_path)
            model.load_from_ckpt(model_path, baseline=True)


        for m in [model]:
            print('-------------------------------------------------------')
            print('Solving with ', m.decoder.__class__.__name__)
            best_log_probs, best_routes, all_costs = m(batch, decode_mode="greedy")

            # split the solution into routes using 0 as the separator
            solution = []
            current_route = []
            for node in best_routes[0]:
                if node == 0:
                    if current_route:
                        solution.append(current_route)
                        current_route = []
                else:
                    current_route.append(node.item())
            print(best_routes[0], "solution", solution)
            # print routes
            total_distance = 0.0
            for route in solution:
                print("route", route)
                deliveries = 0
                pickups = 0
                distance = deltas[0, route[0]]
                for i in range(len(route) - 1):
                    distance += deltas[route[i], route[i + 1]]
                distance += deltas[route[-1], 0]
                print("distance", distance / 100000.0)
                total_distance += distance
                for visit in route:
                    deliveries += node_features[0][visit][2]
                    pickups += node_features[0][visit][3]
                print("deliveries", deliveries)

            print("total_distance", all_costs[0].item(), total_distance / 100000.0)

        # check the pyvrp solution through the model
        log_probs, routes, all_costs = model(batch, actions=actions, decode_mode="greedy")
        print('pyvrp  through the model ' , all_costs)
        print('solution ', routes)



