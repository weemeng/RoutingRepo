import math

from vrp_lib import solve_pdptw


def _make_distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def test_pdptw_satisfies_pair_constraints():
    # Two requests: P1->D1 and P2->D2.
    # Layout:  depot=(0,0), P1=(10,0), D1=(20,0), P2=(0,10), D2=(0,20)
    points = [(0, 0), (10, 0), (20, 0), (0, 10), (0, 20)]
    demands = [0, +1, -1, +1, -1]
    windows = [(0, 480), (10, 60), (60, 200), (10, 60), (60, 200)]
    pairs = [(1, 2), (3, 4)]
    distance = _make_distance(points)

    result = solve_pdptw(
        distance, demands, windows, pairs,
        vehicle_capacities=[2, 2],
        service_time=5, horizon=480,
        time_limit_seconds=5,
    )

    assert result.status == 1
    visited = {n for r in result.routes for n in r}
    assert visited == {0, 1, 2, 3, 4}, "every node must be visited"

    # For each pair: same vehicle, pickup before delivery.
    for pickup_node, delivery_node in pairs:
        v_p = next(v for v, r in enumerate(result.routes) if pickup_node in r)
        v_d = next(v for v, r in enumerate(result.routes) if delivery_node in r)
        assert v_p == v_d, f"pair ({pickup_node},{delivery_node}) on different vehicles"

        route = result.routes[v_p]
        assert route.index(pickup_node) < route.index(delivery_node), (
            f"pair ({pickup_node},{delivery_node}) violates precedence"
        )


def test_pdptw_respects_time_windows():
    points = [(0, 0), (10, 0), (20, 0)]
    demands = [0, +1, -1]
    windows = [(0, 480), (30, 60), (100, 200)]
    pairs = [(1, 2)]
    distance = _make_distance(points)

    result = solve_pdptw(
        distance, demands, windows, pairs,
        vehicle_capacities=[1], service_time=5, horizon=480,
        time_limit_seconds=3,
    )

    assert result.status == 1
    by_node = {entry["node"]: entry for entry in result.schedule}
    for node, win in enumerate(windows):
        if node == 0:
            continue
        arrive = by_node[node]["arrive"]
        assert win[0] <= arrive <= win[1]
