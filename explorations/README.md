# Explorations

Each subfolder is a self-contained experiment. Keep them numbered so the
chronological learning order is obvious.

| Folder            | Topic                  |
| ----------------- | ---------------------- |
| `01_basic_tsp/`   | Single-vehicle TSP     |

## Template

```
NN_topic/
├── README.md     # Problem, approach, findings
├── solve.py      # Entry point: `uv run python explorations/NN_topic/solve.py`
└── data/         # Optional; small fixtures only (large data goes in /data)
```

Reusable logic belongs in `src/vrp_lib/`, not duplicated across explorations.
