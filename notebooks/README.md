# Notebooks

Interactive scratch space for VRP experiments. Launch with:

```bash
uv run jupyter lab
```

Notebooks can import shared utilities directly:

```python
from vrp_lib import solve_routing, euclidean_distance_matrix, plot_routes
```

Promote any logic that stabilises here into `src/vrp_lib/` so it can be tested.
