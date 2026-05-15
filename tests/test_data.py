from vrp_lib.data import euclidean_distance_matrix, manhattan_distance_matrix


def test_euclidean_symmetric_and_zero_diagonal():
    points = [(0, 0), (3, 4), (6, 8)]
    matrix = euclidean_distance_matrix(points)
    for i in range(len(points)):
        assert matrix[i][i] == 0
        for j in range(len(points)):
            assert matrix[i][j] == matrix[j][i]
    assert matrix[0][1] == 5


def test_manhattan_basic():
    matrix = manhattan_distance_matrix([(0, 0), (3, 4)])
    assert matrix[0][1] == 7
