from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def euclidean_distance_matrix(points: Sequence[Point], scale: int = 1) -> list[list[int]]:
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            matrix[i][j] = int(round(math.hypot(xi - xj, yi - yj) * scale))
    return matrix


def manhattan_distance_matrix(points: Sequence[Point], scale: int = 1) -> list[list[int]]:
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            matrix[i][j] = int(round((abs(xi - xj) + abs(yi - yj)) * scale))
    return matrix
