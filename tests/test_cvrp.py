from vrp_lib import euclidean_distance_matrix, solve_cvrp


def test_cvrp_respects_capacity_and_visits_everyone():
    points = [(0, 0), (1, 1), (2, 2), (3, 1), (1, -2), (-1, 1), (-2, 0)]
    demands = [0, 3, 4, 2, 5, 3, 4]  # total 21
    capacities = [12, 12]

    matrix = euclidean_distance_matrix(points, scale=100)
    result = solve_cvrp(matrix, demands, capacities, depot=0, time_limit_seconds=2)

    assert result.routes, "solver should return at least one route"
    for route in result.routes:
        assert route[0] == 0 and route[-1] == 0

    visited = {node for route in result.routes for node in route}
    assert visited == set(range(len(points))), "every customer must be visited"

    for route, capacity in zip(result.routes, capacities):
        load = sum(demands[i] for i in route if i != 0)
        assert load <= capacity, f"route load {load} exceeds capacity {capacity}"

    assert result.loads, "loads should be populated"
    assert sum(result.loads) == sum(demands)


def test_cvrp_drops_nodes_when_capacity_too_tight():
    points = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (2, 2), (-2, 2), (2, -2), (-2, -2)]
    demands = [0, 5, 5, 5, 5, 5, 5, 5, 5]
    capacities = [5]  # one tiny vehicle, can only serve one customer
    matrix = euclidean_distance_matrix(points, scale=100)

    result = solve_cvrp(
        matrix, demands, capacities,
        depot=0, time_limit_seconds=2, drop_node_penalty=1_000_000,
    )

    assert result.dropped, "expected some nodes to be dropped"
    assert 0 not in result.dropped, "depot is never dropped"
    served = [i for route in result.routes for i in route if i != 0]
    assert sum(demands[i] for i in served) <= capacities[0]
