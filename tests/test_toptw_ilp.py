import math

from vrp_lib import solve_toptw_ilp


def _make_distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def test_110_toptw_visits_all_when_budget_loose():
    points = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)]
    profits = [0, 5, 7, 3, 4]
    windows = [(0, 500)] * 5
    distance = _make_distance(points)

    result = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=500,
        service_time=2,
        horizon=500,
        time_limit_seconds=10,
    )

    assert result.status == 1
    assert result.dropped == []
    assert result.total_profit == sum(profits[1:])
    visited = {n for r in result.routes for n in r}
    assert visited == {0, 1, 2, 3, 4}


def test_111_toptw_skips_low_profit_when_budget_tight():
    # Four customers at distance 100 from depot in four directions.
    # Each visit costs ~200 round-trip + 10 service. With t_max=620 only 3
    # round-trips fit (3 * 210 = 630 > 620 is exactly the boundary; we use
    # t_max=625 to permit any 3-customer tour and force one drop).
    points = [(0, 0), (100, 0), (0, 100), (-100, 0), (0, -100)]
    profits = [0, 10, 10, 10, 1]
    windows = [(0, 1000)] * 5
    distance = _make_distance(points)

    result = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=625,
        service_time=5,
        horizon=1000,
        time_limit_seconds=10,
    )

    assert result.status == 1
    assert 4 in result.dropped, f"expected node 4 dropped, got dropped={result.dropped}"
    assert result.total_profit == 30


def test_112_toptw_window_forces_skip():
    # Customer 2 is reachable but its window closes before we can arrive.
    points = [(0, 0), (10, 0), (200, 0)]
    profits = [0, 5, 100]
    windows = [(0, 500), (0, 500), (0, 50)]  # node 2 needs arrival <= 50, but dist=200
    distance = _make_distance(points)

    result = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=500,
        service_time=0,
        horizon=500,
        time_limit_seconds=10,
    )

    assert result.status == 1
    assert 2 in result.dropped
    assert result.total_profit == 5


def test_113_toptw_mandatory_visit_forces_other_drop():
    # Same tight-budget instance as test_111 — only 3 customers fit. Without
    # mandatory, the low-profit customer (4) is the natural drop. Forcing 4
    # to be visited via mandatory=[4] must instead drop one of {1, 2, 3}.
    points = [(0, 0), (100, 0), (0, 100), (-100, 0), (0, -100)]
    profits = [0, 10, 10, 10, 1]
    windows = [(0, 1000)] * 5
    distance = _make_distance(points)

    result = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=625,
        service_time=5,
        horizon=1000,
        mandatory=[4],
        time_limit_seconds=10,
    )

    assert result.status == 1
    assert 4 not in result.dropped
    dropped_high = set(result.dropped) & {1, 2, 3}
    assert len(dropped_high) == 1, f"expected exactly one of {{1,2,3}} dropped, got {result.dropped}"
    assert result.total_profit == 21  # 10 + 10 + 1


def test_114_toptw_mandatory_infeasible_when_window_unreachable():
    # Reuse test_112: customer 2's window closes before arrival is possible.
    # Forcing it via mandatory must make the model infeasible.
    points = [(0, 0), (10, 0), (200, 0)]
    profits = [0, 5, 100]
    windows = [(0, 500), (0, 500), (0, 50)]
    distance = _make_distance(points)

    result = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=500,
        service_time=0,
        horizon=500,
        mandatory=[2],
        time_limit_seconds=10,
    )

    assert result.status != 1
    assert result.routes == []
