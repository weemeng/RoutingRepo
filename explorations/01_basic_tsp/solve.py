"""Basic single-vehicle TSP on a small set of points."""

from __future__ import annotations

from pathlib import Path

from vrp_lib import euclidean_distance_matrix, plot_routes, solve_routing

POINTS: list[tuple[float, float]] = [
    (0, 0),
    (2, 3),
    (5, 2),
    (6, 6),
    (8, 3),
    (7, 9),
    (3, 8),
    (1, 6),
    (4, 4),
    (9, 1),
]


def main() -> None:
    matrix = euclidean_distance_matrix(POINTS, scale=100)
    result = solve_routing(matrix, num_vehicles=1, depot=0, time_limit_seconds=5)

    print(f"Status: {result.status}")
    print(f"Total distance (scaled): {result.total_distance}")
    for vid, route in enumerate(result.routes):
        print(f"  vehicle {vid}: {' -> '.join(map(str, route))}")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plot_routes(POINTS, result.routes, save_to=output_dir / "route.png", title="Basic TSP")
    print(f"Saved plot to {output_dir / 'route.png'}")


if __name__ == "__main__":
    main()
