import math

from vrp_lib import solve_vrptw


def _make_distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def test_vrptw_respects_time_windows():
    points = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)]
    demands = [0, 5, 5, 5, 5]
    windows = [(0, 480), (20, 60), (80, 120), (140, 200), (220, 280)]
    distance = _make_distance(points)

    result = solve_vrptw(
        distance, demands, windows,
        vehicle_capacities=[20],
        service_time=5,
        horizon=480,
        time_limit_seconds=3,
    )

    assert result.status == 1, f"solver did not succeed, status={result.status}"
    assert result.routes and result.routes[0][0] == 0 and result.routes[0][-1] == 0
    visited = {n for r in result.routes for n in r}
    assert visited == {0, 1, 2, 3, 4}

    by_node = {entry["node"]: entry for entry in result.schedule if entry["node"] != 0}
    for node, win in enumerate(windows):
        if node == 0:
            continue
        arrive = by_node[node]["arrive"]
        assert win[0] <= arrive <= win[1], (
            f"node {node} arrived at {arrive}, outside window {win}"
        )


def test_vrptw_soft_windows_finds_solution_when_hard_infeasible():
    # Hard windows force pickup at node 1 in [30, 60] then node 2 in [30, 60]
    # but they're 100 km apart -- impossible.
    points = [(0, 0), (100, 0), (0, 100)]
    demands = [0, 1, 1]
    windows = [(0, 480), (30, 60), (30, 60)]
    distance = _make_distance(points)

    hard = solve_vrptw(
        distance, demands, windows,
        vehicle_capacities=[10],
        service_time=10, horizon=480,
        time_limit_seconds=2,
    )
    assert hard.status != 1 or not hard.routes

    # With soft windows, the solver should serve both nodes (one late).
    soft = solve_vrptw(
        distance, demands, windows,
        vehicle_capacities=[10],
        service_time=10, horizon=480,
        early_penalty=10, late_penalty=10,
        time_limit_seconds=3,
    )
    assert soft.status == 1
    visited = {n for r in soft.routes for n in r}
    assert visited == {0, 1, 2}

    # At least one node must be late given the geometry.
    by_node = {e["node"]: e for e in soft.schedule}
    total_violation = 0
    for node, (a, b) in enumerate(windows):
        if node == 0:
            continue
        arr = by_node[node]["arrive"]
        if arr < a:
            total_violation += a - arr
        elif arr > b:
            total_violation += arr - b
    assert total_violation > 0


def test_vrptw_returns_failure_when_infeasible():
    points = [(0, 0), (100, 0), (0, 100)]
    demands = [0, 1, 1]
    windows = [(0, 480), (30, 60), (30, 60)]
    distance = _make_distance(points)

    result = solve_vrptw(
        distance, demands, windows,
        vehicle_capacities=[10],
        service_time=10,
        horizon=480,
        time_limit_seconds=2,
    )

    assert result.status != 1 or not result.routes
