import math

from vrp_lib import solve_toptw_ilp, solve_toptw_teg_ilp


def _make_distance(points):
    n = len(points)
    m = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i != j:
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj)))
    return m


def test_130_teg_collapses_to_toptw_when_shifts_identical():
    # One shift covering the whole horizon, identical travel matrix → the
    # TEG solver must reach the same total profit and visit set as the
    # plain TOPTW ILP on the same instance.
    points = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)]
    profits = [0, 5, 7, 3, 4]
    windows = [(0, 500)] * 5
    distance = _make_distance(points)

    baseline = solve_toptw_ilp(
        distance, profits, windows,
        num_vehicles=1,
        t_max=500,
        service_time=2,
        horizon=500,
        time_limit_seconds=10,
    )

    teg = solve_toptw_teg_ilp(
        shift_travel=[distance],
        shift_windows=[(0, 500)],
        profits=profits,
        time_windows=windows,
        num_vehicles=1,
        t_max=500,
        service_time=2,
        horizon=500,
        time_step=2,
        time_limit_seconds=10,
    )

    assert teg.status == 1
    assert teg.total_profit == baseline.total_profit
    teg_visited = {n for r in teg.routes for n in r if n != 0}
    base_visited = {n for r in baseline.routes for n in r if n != 0}
    assert teg_visited == base_visited


def test_131_teg_morning_fast_afternoon_slow_changes_route():
    # Two shifts. Morning (0..200) is fast; afternoon (200..400) is 4x
    # slower on every customer arc. With t_max=200 the morning shift is
    # the only way to visit all four customers.
    points   = [(0, 0), (40, 0), (0, 40), (-40, 0), (0, -40)]
    base = _make_distance(points)
    morning  = [row[:] for row in base]
    afternoon = [[4 * x for x in row] for row in base]
    profits = [0, 10, 10, 10, 10]
    windows = [(0, 400)] * 5

    teg = solve_toptw_teg_ilp(
        shift_travel=[morning, afternoon],
        shift_windows=[(0, 200), (200, 400)],
        profits=profits,
        time_windows=windows,
        num_vehicles=1,
        t_max=200,
        service_time=5,
        horizon=400,
        time_step=5,
        time_limit_seconds=10,
    )

    afternoon_only = solve_toptw_teg_ilp(
        shift_travel=[afternoon],
        shift_windows=[(0, 400)],
        profits=profits,
        time_windows=windows,
        num_vehicles=1,
        t_max=200,
        service_time=5,
        horizon=400,
        time_step=5,
        time_limit_seconds=10,
    )

    assert teg.status == 1
    assert afternoon_only.status == 1
    assert teg.total_profit > afternoon_only.total_profit, (
        f"two-shift solver should exploit fast morning travel "
        f"(profit={teg.total_profit} vs afternoon-only={afternoon_only.total_profit})"
    )
    # Every leg of the TEG solution should be scheduled within the morning
    # window — the budget is too tight for any afternoon arc.
    for entry in teg.schedule:
        assert entry["arrive"] <= 200


def test_132_teg_mandatory_works():
    # Same tight-budget setup as test_113 but with TEG: forcing node 4 to
    # be visited displaces one of the profit-10 customers.
    points = [(0, 0), (100, 0), (0, 100), (-100, 0), (0, -100)]
    profits = [0, 10, 10, 10, 1]
    windows = [(0, 1000)] * 5
    distance = _make_distance(points)

    result = solve_toptw_teg_ilp(
        shift_travel=[distance],
        shift_windows=[(0, 1000)],
        profits=profits,
        time_windows=windows,
        num_vehicles=1,
        t_max=625,
        service_time=5,
        horizon=1000,
        time_step=5,
        mandatory=[4],
        time_limit_seconds=15,
    )

    assert result.status == 1
    assert 4 not in result.dropped
    dropped_high = set(result.dropped) & {1, 2, 3}
    assert len(dropped_high) == 1, f"expected one high-profit drop, got {result.dropped}"
    assert result.total_profit == 21


def test_133_teg_drops_when_budget_tight():
    # Three customers; budget admits exactly two.
    points = [(0, 0), (50, 0), (0, 50), (-50, 0)]
    profits = [0, 5, 5, 5]
    windows = [(0, 500)] * 4
    distance = _make_distance(points)

    result = solve_toptw_teg_ilp(
        shift_travel=[distance],
        shift_windows=[(0, 500)],
        profits=profits,
        time_windows=windows,
        num_vehicles=1,
        t_max=215,
        service_time=5,
        horizon=500,
        time_step=5,
        time_limit_seconds=10,
    )

    assert result.status == 1
    assert len(result.dropped) == 1
    assert result.total_profit == 10
