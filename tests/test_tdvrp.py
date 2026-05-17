import math

from vrp_lib import solve_tdvrp


def _distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def _scale(matrix, mult):
    return [[int(round(v * mult)) for v in row] for row in matrix]


def test_tdvrp_assigns_customers_to_required_shifts():
    # 4 customers, depot at origin.
    # c1 must be morning, c3 must be afternoon, c2 and c4 are flexible.
    points = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)]
    demands = [0, 5, 5, 5, 5]
    base = _distance(points)

    shift_travel = [
        _scale(base, 1.0),   # morning
        _scale(base, 0.85),  # midday
        _scale(base, 1.5),   # afternoon
    ]
    shift_windows = [(0, 240), (240, 420), (420, 600)]

    # c1 morning-only, c3 afternoon-only, others flexible.
    windows = [(0, 600), (0, 240), (0, 600), (420, 600), (0, 600)]

    result = solve_tdvrp(
        shift_travel=shift_travel,
        shift_windows=shift_windows,
        demands=demands,
        time_windows=windows,
        vehicle_capacity=20,
        service_time=10,
        horizon=600,
        time_limit_seconds=3,
    )

    assert result.status == 1
    assert len(result.routes) == 3

    # Locate which shift each customer landed in.
    where = {}
    for v, route in enumerate(result.routes):
        for node in route:
            if node != 0:
                where[node] = v

    assert where[1] == 0, f"c1 must be morning, got shift {where[1]}"
    assert where[3] == 2, f"c3 must be afternoon, got shift {where[3]}"
    # All four customers visited.
    assert set(where.keys()) == {1, 2, 3, 4}


def test_tdvrp_shift_start_respects_window():
    # Single shift, late-only.
    points = [(0, 0), (5, 0), (0, 5)]
    demands = [0, 1, 1]
    base = _distance(points)

    result = solve_tdvrp(
        shift_travel=[_scale(base, 1.0)],
        shift_windows=[(300, 600)],
        demands=demands,
        time_windows=[(0, 600), (300, 600), (300, 600)],
        vehicle_capacity=10,
        service_time=5,
        horizon=600,
        time_limit_seconds=2,
    )
    assert result.status == 1
    # Start cumul must be >= 300.
    start_entries = [e for e in result.schedule if e["node"] == 0]
    assert all(e["arrive"] >= 300 for e in start_entries)
