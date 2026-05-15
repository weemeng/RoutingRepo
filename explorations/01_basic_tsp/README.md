# 01 — Basic TSP

The simplest VRP: a single vehicle visiting every node exactly once and
returning to the depot. Useful as a sanity check that the toolchain works.

## Run

```bash
uv run python explorations/01_basic_tsp/solve.py
```

## What to look at next

- Swap `FirstSolutionStrategy` (e.g. `SAVINGS`, `CHRISTOFIDES`) and compare.
- Try the `LocalSearchMetaheuristic` options.
- Increase the number of nodes and watch solve time grow.
