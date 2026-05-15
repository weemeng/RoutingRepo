# RoutingRepo

A learning sandbox for **Vehicle Routing Problems (VRP)** with [Google OR-Tools](https://developers.google.com/optimization/routing).

The layout is built around small, self-contained **explorations** that share a common utilities library, plus a `tests/` folder for verifying that shared code keeps working.

## Layout

```
.
├── pyproject.toml          # uv-managed project + dependencies
├── src/vrp_lib/            # Shared utilities (data, solver wrappers, viz)
├── explorations/           # Self-contained experiments (one folder each)
│   └── 01_basic_tsp/       # Starter example
├── notebooks/              # Jupyter notebooks for interactive exploration
├── tests/                  # pytest suite for vrp_lib
├── data/                   # raw/ and processed/ datasets (gitignored)
└── docs/                   # Notes, learnings, write-ups
```

## Quickstart

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
uv sync --all-extras                       # install everything (incl. notebooks + dev)
uv run python explorations/01_basic_tsp/solve.py
uv run pytest
```

To launch Jupyter:

```bash
uv run jupyter lab
```

## Adding a new exploration

1. Copy `explorations/01_basic_tsp/` to `explorations/NN_your_topic/`.
2. Update its `README.md` with the problem statement and what you're trying.
3. Put any reusable code into `src/vrp_lib/` and import from there.
4. Add a test under `tests/` if it's library-level logic.

## Conventions

- **One folder per exploration.** Number prefixes (`01_`, `02_`) keep them ordered.
- **Pure scripts** under `solve.py`, **interactive work** under `notebooks/`.
- **Shared logic** lives in `src/vrp_lib/` — never copy-paste between explorations.
- **Data** stays out of git (see `.gitignore`); commit small fixtures only.
