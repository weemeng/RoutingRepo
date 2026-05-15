from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

Point = tuple[float, float]


def plot_routes(
    points: Sequence[Point],
    routes: Sequence[Sequence[int]],
    save_to: str | Path | None = None,
    title: str = "Routes",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c="black", zorder=3)
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5))

    for vehicle_id, route in enumerate(routes):
        rx = [points[node][0] for node in route]
        ry = [points[node][1] for node in route]
        ax.plot(rx, ry, marker="o", label=f"vehicle {vehicle_id}")

    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")

    if save_to:
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
