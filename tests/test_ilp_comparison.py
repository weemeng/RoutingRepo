import math

from vrp_lib import solve_toptw_ilp, solve_vrptw_ilp


def _make_distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def test_120_vrptw_and_toptw_share_arc_flow_structure():
    points = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)]
    demands = [0, 5, 5, 5, 5]
    windows = [(0, 480), (20, 60), (80, 120), (140, 200), (220, 280)]
    distance = _make_distance(points)

    vrptw = solve_vrptw_ilp(
        distance, demands, windows,
        vehicle_capacities=[20],
        service_time=5,
        horizon=480,
        time_limit_seconds=10,
    )

    # Loose budget + uniform profits -> TOPTW should visit everyone too.
    profits = [0, 1, 1, 1, 1]
    toptw = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=480,
        service_time=5,
        horizon=480,
        time_limit_seconds=10,
    )

    assert vrptw.status == 1
    assert toptw.status == 1
    assert toptw.dropped == []
    assert toptw.total_profit == sum(profits[1:])
    vrptw_visited = {n for r in vrptw.routes for n in r}
    toptw_visited = {n for r in toptw.routes for n in r}
    assert vrptw_visited == toptw_visited == {0, 1, 2, 3, 4}


def test_121_toptw_drops_when_vrptw_infeasible():
    # Same data as test_102_vrptw_ilp_infeasible.
    points = [(0, 0), (100, 0), (0, 100)]
    demands = [0, 1, 1]
    windows = [(0, 480), (30, 60), (30, 60)]
    distance = _make_distance(points)

    vrptw = solve_vrptw_ilp(
        distance, demands, windows,
        vehicle_capacities=[10],
        service_time=10,
        horizon=480,
        time_limit_seconds=10,
    )

    profits = [0, 5, 5]
    toptw = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=480,
        service_time=10,
        horizon=480,
        time_limit_seconds=10,
    )

    # VRPTW must visit both within tight windows -> infeasible.
    assert vrptw.status != 1 or not vrptw.routes
    # TOPTW can skip the unreachable customer and still return a feasible answer.
    assert toptw.status == 1
    assert toptw.dropped, "expected TOPTW to drop at least one customer"


def test_122_toptw_total_distance_le_vrptw_when_budget_tight():
    points = [(0, 0), (100, 0), (0, 100), (-100, 0), (0, -100)]
    demands = [0, 1, 1, 1, 1]
    windows = [(0, 1000)] * 5
    distance = _make_distance(points)

    vrptw = solve_vrptw_ilp(
        distance, demands, windows,
        vehicle_capacities=[10],
        service_time=5,
        horizon=1000,
        time_limit_seconds=10,
    )
    assert vrptw.status == 1

    # Tight budget forces TOPTW to drop at least one customer.
    profits = [0, 10, 10, 10, 1]
    toptw = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=625,
        service_time=5,
        horizon=1000,
        time_limit_seconds=10,
    )

    assert toptw.status == 1
    assert toptw.dropped, "expected TOPTW to drop at least one customer"
    assert toptw.total_distance <= vrptw.total_distance
