from __future__ import annotations

from collections.abc import Sequence

from ortools.sat.python import cp_model

from vrp_lib.solver import SolveResult


# Maps CP-SAT status codes onto the integer convention used by the
# routing-solver based functions (status == 1 means success in the existing
# test suite). We deliberately collapse OPTIMAL and FEASIBLE into 1 so that
# callers can treat both as a successful solve; switch to a richer signal
# (e.g. an `optimal: bool`) on SolveResult if the distinction matters later.
_CPSAT_STATUS_MAP = {
    cp_model.OPTIMAL: 1,
    cp_model.FEASIBLE: 1,
    cp_model.INFEASIBLE: 3,
    cp_model.MODEL_INVALID: 5,
    cp_model.UNKNOWN: 0,
}


def _arc_keys(n: int, num_vehicles: int) -> list[tuple[int, int, int]]:
    return [
        (i, j, k)
        for k in range(num_vehicles)
        for i in range(n)
        for j in range(n)
        if i != j
    ]


def _reconstruct_routes(
    solver: cp_model.CpSolver,
    x: dict[tuple[int, int, int], cp_model.IntVar],
    n: int,
    num_vehicles: int,
    depot: int,
) -> list[list[int]]:
    routes: list[list[int]] = []
    for k in range(num_vehicles):
        route = [depot]
        current = depot
        visited_local: set[int] = {depot}
        while True:
            next_node = None
            for j in range(n):
                if j == current:
                    continue
                if solver.Value(x[(current, j, k)]) == 1:
                    next_node = j
                    break
            if next_node is None or next_node == depot:
                route.append(depot)
                break
            if next_node in visited_local:
                # defensive: should not occur given time/load propagation,
                # but avoid infinite loops if the model is ever changed.
                route.append(depot)
                break
            route.append(next_node)
            visited_local.add(next_node)
            current = next_node
        routes.append(route)
    return routes


def solve_vrptw_ilp(
    distance: Sequence[Sequence[int]],
    demands: Sequence[int],
    time_windows: Sequence[tuple[int, int]],
    vehicle_capacities: Sequence[int],
    service_time: int,
    horizon: int,
    depot: int = 0,
    time_limit_seconds: int = 10,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> SolveResult:
    """Arc-flow ILP for VRPTW using CP-SAT.

    All inputs must be integers (CP-SAT is integer-only). Scale floats
    by a constant factor before calling.
    """
    n = len(distance)
    num_vehicles = len(vehicle_capacities)
    model = cp_model.CpModel()

    arcs = _arc_keys(n, num_vehicles)

    x: dict[tuple[int, int, int], cp_model.IntVar] = {
        (i, j, k): model.NewBoolVar(f"x_{i}_{j}_{k}") for (i, j, k) in arcs
    }

    depot_open, depot_close = time_windows[depot]
    t: dict[tuple[int, int], cp_model.IntVar] = {}
    for k in range(num_vehicles):
        for i in range(n):
            if i == depot:
                lo, hi = int(depot_open), int(depot_close)
            else:
                lo, hi = int(time_windows[i][0]), int(time_windows[i][1])
            t[(i, k)] = model.NewIntVar(lo, hi, f"t_{i}_{k}")

    t_end: dict[int, cp_model.IntVar] = {
        k: model.NewIntVar(int(depot_open), int(horizon), f"t_end_{k}")
        for k in range(num_vehicles)
    }

    load: dict[tuple[int, int], cp_model.IntVar] = {}
    for k in range(num_vehicles):
        cap = int(vehicle_capacities[k])
        for i in range(n):
            load[(i, k)] = model.NewIntVar(0, cap, f"load_{i}_{k}")
        model.Add(load[(depot, k)] == 0)

    # Each customer visited exactly once across all vehicles.
    for i in range(n):
        if i == depot:
            continue
        model.Add(
            sum(x[(i, j, k)] for k in range(num_vehicles) for j in range(n) if j != i)
            == 1
        )

    # Each vehicle leaves and returns to the depot exactly once.
    for k in range(num_vehicles):
        model.Add(sum(x[(depot, j, k)] for j in range(n) if j != depot) == 1)
        model.Add(sum(x[(i, depot, k)] for i in range(n) if i != depot) == 1)

    # Flow conservation per vehicle at every customer.
    for k in range(num_vehicles):
        for h in range(n):
            if h == depot:
                continue
            inflow = sum(x[(i, h, k)] for i in range(n) if i != h)
            outflow = sum(x[(h, j, k)] for j in range(n) if j != h)
            model.Add(inflow == outflow)
            model.Add(inflow <= 1)

    # Time propagation. Subtour elimination is implicit: any cycle disjoint
    # from the depot would require t to strictly increase around it.
    for (i, j, k), arc in x.items():
        st = 0 if i == depot else service_time
        if j == depot:
            model.Add(t_end[k] >= t[(i, k)] + st + int(distance[i][j])).OnlyEnforceIf(arc)
        else:
            model.Add(
                t[(j, k)] >= t[(i, k)] + st + int(distance[i][j])
            ).OnlyEnforceIf(arc)

    # Capacity propagation (MTZ-style, also reinforces subtour elimination).
    for (i, j, k), arc in x.items():
        if j == depot:
            continue
        model.Add(load[(j, k)] >= load[(i, k)] + int(demands[j])).OnlyEnforceIf(arc)

    model.Minimize(
        sum(int(distance[i][j]) * x[(i, j, k)] for (i, j, k) in arcs)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    cpsat_status = solver.Solve(model)
    mapped = _CPSAT_STATUS_MAP.get(cpsat_status, 0)
    if mapped != 1:
        return SolveResult(routes=[], total_distance=0, status=mapped)

    routes = _reconstruct_routes(solver, x, n, num_vehicles, depot)

    schedule: list[dict] = []
    loads_out: list[int] = []
    total = 0
    for k, route in enumerate(routes):
        route_load = 0
        for idx, node in enumerate(route):
            if idx == len(route) - 1:
                arrive = solver.Value(t_end[k])
                leave = arrive
            else:
                arrive = solver.Value(t[(node, k)])
                leave = arrive + (0 if node == depot else service_time)
            schedule.append({
                "vehicle": k,
                "node": node,
                "arrive": arrive,
                "leave": leave,
            })
            if node != depot:
                route_load += int(demands[node])
        for a, b in zip(route, route[1:]):
            total += int(distance[a][b])
        loads_out.append(route_load)

    return SolveResult(
        routes=routes,
        total_distance=total,
        status=1,
        loads=loads_out,
        schedule=schedule,
    )


def solve_toptw_ilp(
    distance: Sequence[Sequence[int]],
    profits: Sequence[int],
    time_windows: Sequence[tuple[int, int]],
    num_vehicles: int,
    t_max: int,
    service_time: int,
    horizon: int,
    depot: int = 0,
    demands: Sequence[int] | None = None,
    vehicle_capacities: Sequence[int] | None = None,
    time_limit_seconds: int = 10,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> SolveResult:
    """Arc-flow ILP for TOPTW (Team Orienteering Problem with Time Windows).

    Customers carry a profit and visits are optional. Each vehicle has a
    per-route duration budget `t_max`. Capacity constraints are activated
    only when both `demands` and `vehicle_capacities` are provided.
    """
    n = len(distance)
    if (demands is None) != (vehicle_capacities is None):
        raise ValueError("demands and vehicle_capacities must be provided together")
    use_capacity = demands is not None
    if use_capacity and len(vehicle_capacities) != num_vehicles:
        raise ValueError("vehicle_capacities length must equal num_vehicles")

    model = cp_model.CpModel()
    arcs = _arc_keys(n, num_vehicles)

    x: dict[tuple[int, int, int], cp_model.IntVar] = {
        (i, j, k): model.NewBoolVar(f"x_{i}_{j}_{k}") for (i, j, k) in arcs
    }
    y: dict[int, cp_model.IntVar] = {
        i: model.NewBoolVar(f"y_{i}") for i in range(n) if i != depot
    }

    depot_open, depot_close = time_windows[depot]
    t: dict[tuple[int, int], cp_model.IntVar] = {}
    for k in range(num_vehicles):
        for i in range(n):
            if i == depot:
                lo, hi = int(depot_open), int(depot_close)
            else:
                lo, hi = int(time_windows[i][0]), int(time_windows[i][1])
            t[(i, k)] = model.NewIntVar(lo, hi, f"t_{i}_{k}")

    t_end: dict[int, cp_model.IntVar] = {
        k: model.NewIntVar(int(depot_open), int(t_max), f"t_end_{k}")
        for k in range(num_vehicles)
    }

    load: dict[tuple[int, int], cp_model.IntVar] = {}
    if use_capacity:
        for k in range(num_vehicles):
            cap = int(vehicle_capacities[k])
            for i in range(n):
                load[(i, k)] = model.NewIntVar(0, cap, f"load_{i}_{k}")
            model.Add(load[(depot, k)] == 0)

    # Visit indicator: y[i] == sum over vehicles/predecessors of x[*,i,*].
    for i in range(n):
        if i == depot:
            continue
        model.Add(
            sum(x[(j, i, k)] for k in range(num_vehicles) for j in range(n) if j != i)
            == y[i]
        )

    # Each vehicle leaves the depot at most once and returns the same number
    # of times. A vehicle that visits zero customers stays at the depot.
    for k in range(num_vehicles):
        depot_out = sum(x[(depot, j, k)] for j in range(n) if j != depot)
        depot_in = sum(x[(i, depot, k)] for i in range(n) if i != depot)
        model.Add(depot_out <= 1)
        model.Add(depot_out == depot_in)

    # Flow conservation per vehicle at every customer.
    for k in range(num_vehicles):
        for h in range(n):
            if h == depot:
                continue
            inflow = sum(x[(i, h, k)] for i in range(n) if i != h)
            outflow = sum(x[(h, j, k)] for j in range(n) if j != h)
            model.Add(inflow == outflow)
            model.Add(inflow <= 1)

    # Time propagation + per-vehicle duration budget.
    for (i, j, k), arc in x.items():
        st = 0 if i == depot else service_time
        if j == depot:
            model.Add(t_end[k] >= t[(i, k)] + st + int(distance[i][j])).OnlyEnforceIf(arc)
        else:
            model.Add(
                t[(j, k)] >= t[(i, k)] + st + int(distance[i][j])
            ).OnlyEnforceIf(arc)

    if use_capacity:
        for (i, j, k), arc in x.items():
            if j == depot:
                continue
            model.Add(load[(j, k)] >= load[(i, k)] + int(demands[j])).OnlyEnforceIf(arc)

    model.Maximize(sum(int(profits[i]) * y[i] for i in y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    cpsat_status = solver.Solve(model)
    mapped = _CPSAT_STATUS_MAP.get(cpsat_status, 0)
    if mapped != 1:
        return SolveResult(routes=[], total_distance=0, status=mapped)

    routes = _reconstruct_routes(solver, x, n, num_vehicles, depot)

    visited: set[int] = set()
    for route in routes:
        for node in route:
            if node != depot:
                visited.add(node)
    dropped = sorted(i for i in range(n) if i != depot and i not in visited)

    schedule: list[dict] = []
    loads_out: list[int] = []
    total_distance = 0
    for k, route in enumerate(routes):
        route_load = 0
        for idx, node in enumerate(route):
            if idx == len(route) - 1:
                arrive = solver.Value(t_end[k])
                leave = arrive
            else:
                arrive = solver.Value(t[(node, k)])
                leave = arrive + (0 if node == depot else service_time)
            schedule.append({
                "vehicle": k,
                "node": node,
                "arrive": arrive,
                "leave": leave,
            })
            if use_capacity and node != depot:
                route_load += int(demands[node])
        for a, b in zip(route, route[1:]):
            total_distance += int(distance[a][b])
        loads_out.append(route_load)

    total_profit = int(round(solver.ObjectiveValue()))

    return SolveResult(
        routes=routes,
        total_distance=total_distance,
        status=1,
        loads=loads_out if use_capacity else [],
        dropped=dropped,
        schedule=schedule,
        total_profit=total_profit,
    )
