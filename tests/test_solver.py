from vrp_lib import euclidean_distance_matrix, solve_routing


def test_solve_small_tsp_visits_every_node():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    matrix = euclidean_distance_matrix(points, scale=100)

    result = solve_routing(matrix, num_vehicles=1, depot=0, time_limit_seconds=2)

    assert len(result.routes) == 1
    route = result.routes[0]
    assert route[0] == 0 and route[-1] == 0
    assert set(route) == {0, 1, 2, 3}
    assert result.total_distance > 0
