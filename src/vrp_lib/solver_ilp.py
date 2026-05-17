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
    mandatory: Sequence[int] | None = None,
    time_limit_seconds: int = 10,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> SolveResult:
    """Arc-flow ILP for TOPTW (Team Orienteering Problem with Time Windows).

    Customers carry a profit and visits are optional. Each vehicle has a
    per-route duration budget `t_max`. Capacity constraints are activated
    only when both `demands` and `vehicle_capacities` are provided.

    Customers listed in `mandatory` must appear on some vehicle's route;
    the problem becomes infeasible if a mandatory customer cannot be
    reached within its time window and the duration budget.
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

    if mandatory:
        for i in mandatory:
            if i == depot:
                raise ValueError("depot cannot be marked mandatory")
            if not 0 <= i < n:
                raise ValueError(f"mandatory index {i} out of range [0, {n})")
            model.Add(y[i] == 1)

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


# ---------------------------------------------------------------------------
# Time-expanded graph (TEG) TOPTW
# ---------------------------------------------------------------------------

TEGNode = tuple[int, int]
TEGArc = tuple[TEGNode, TEGNode]


def _shift_of(tau: int, shift_windows: Sequence[tuple[int, int]]) -> int:
    """Return the index of the shift containing tau (half-open `[open, close)`).

    Tau exactly at the close of the final shift is treated as belonging to
    that final shift so that arcs landing on the horizon boundary are valid.
    """
    for s, (lo, hi) in enumerate(shift_windows):
        if lo <= tau < hi:
            return s
    if shift_windows and tau == shift_windows[-1][1]:
        return len(shift_windows) - 1
    raise ValueError(f"time {tau} falls outside shift_windows {list(shift_windows)}")


def _round_up_to_grid(value: int, step: int) -> int:
    if value <= 0:
        return 0
    return ((value + step - 1) // step) * step


def _build_teg(
    shift_travel: Sequence[Sequence[Sequence[int]]],
    shift_windows: Sequence[tuple[int, int]],
    time_windows: Sequence[tuple[int, int]],
    n: int,
    depot: int,
    service_time: int,
    horizon: int,
    t_max: int,
    time_step: int,
) -> tuple[dict[int, list[int]], list[TEGArc]]:
    """Construct the time-expanded graph.

    Returns `(nodes_at, arcs)` where `nodes_at[i]` is the sorted list of
    valid time copies for customer i, and `arcs` is the flat list of
    physically feasible arcs `((i, tau_i), (j, tau_j))`.
    """
    taus_all = list(range(0, horizon + 1, time_step))

    nodes_at: dict[int, list[int]] = {}
    for i in range(n):
        a, b = time_windows[i]
        if i == depot:
            upper = min(int(b), int(t_max))
            nodes_at[i] = [tau for tau in taus_all if 0 <= tau <= upper]
        else:
            nodes_at[i] = [tau for tau in taus_all if int(a) <= tau <= int(b)]

    valid_set = {(i, tau) for i, taus in nodes_at.items() for tau in taus}

    arcs: list[TEGArc] = []
    for i in range(n):
        # Depot only emits outgoing arcs from tau=0; the other depot copies
        # are return-only sinks.
        source_taus: Sequence[int] = (0,) if i == depot else nodes_at[i]
        for tau in source_taus:
            if (i, tau) not in valid_set:
                continue
            shift_idx = _shift_of(tau, shift_windows)
            d_row = shift_travel[shift_idx][i]
            st = 0 if i == depot else int(service_time)
            for j in range(n):
                if j == i:
                    continue
                arrive_raw = tau + st + int(d_row[j])
                arrive = _round_up_to_grid(arrive_raw, time_step)
                if j == depot:
                    if arrive > t_max:
                        continue
                else:
                    a_j, b_j = time_windows[j]
                    if arrive < int(a_j):
                        arrive = _round_up_to_grid(int(a_j), time_step)
                    if arrive > int(b_j):
                        continue
                if (j, arrive) not in valid_set:
                    continue
                arcs.append(((i, tau), (j, arrive)))

    return nodes_at, arcs


def solve_toptw_teg_ilp(
    shift_travel: Sequence[Sequence[Sequence[int]]],
    shift_windows: Sequence[tuple[int, int]],
    profits: Sequence[int],
    time_windows: Sequence[tuple[int, int]],
    num_vehicles: int,
    t_max: int,
    service_time: int,
    horizon: int,
    time_step: int,
    depot: int = 0,
    mandatory: Sequence[int] | None = None,
    time_limit_seconds: int = 10,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> SolveResult:
    """Time-dependent TOPTW solved as an arc-flow ILP on a time-expanded graph.

    Each TEG node is a pair `(customer, tau)`. Arcs are pre-filtered so that
    `tau_dst = round_up(tau_src + service + d_ij[shift(tau_src)])` lands
    inside the destination's time window and (for return-to-depot) within
    `t_max`. Arrival earlier than a customer's window-open is bumped up,
    encoding implicit waiting (loiter) at the destination.

    `time_step` is the TEG resolution; finer grids approach the continuous
    problem at the cost of more variables. Travel times and shift windows
    should be commensurable with `time_step` to avoid systematic rounding
    pessimism.
    """
    if len(shift_travel) != len(shift_windows):
        raise ValueError("shift_travel and shift_windows must have the same length")
    if not shift_travel:
        raise ValueError("at least one shift is required")
    n = len(shift_travel[0])
    if any(len(m) != n or any(len(row) != n for row in m) for m in shift_travel):
        raise ValueError("every shift_travel matrix must be n x n")
    if len(profits) != n or len(time_windows) != n:
        raise ValueError("profits and time_windows must have length n")
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    nodes_at, arcs = _build_teg(
        shift_travel, shift_windows, time_windows,
        n=n, depot=depot, service_time=service_time,
        horizon=horizon, t_max=t_max, time_step=time_step,
    )

    model = cp_model.CpModel()

    # x[arc, k] for every feasible TEG arc and every vehicle.
    x: dict[tuple[TEGArc, int], cp_model.IntVar] = {}
    for arc_idx, arc in enumerate(arcs):
        for k in range(num_vehicles):
            x[(arc, k)] = model.NewBoolVar(f"x_{arc_idx}_{k}")

    # y[i] — was customer i visited (by any vehicle, at any time copy)?
    y: dict[int, cp_model.IntVar] = {
        i: model.NewBoolVar(f"y_{i}") for i in range(n) if i != depot
    }

    if mandatory:
        for i in mandatory:
            if i == depot:
                raise ValueError("depot cannot be marked mandatory")
            if not 0 <= i < n:
                raise ValueError(f"mandatory index {i} out of range [0, {n})")
            model.Add(y[i] == 1)

    # Pre-index arcs by source and destination TEG nodes for fast constraint
    # assembly.
    out_of: dict[TEGNode, list[TEGArc]] = {}
    in_to: dict[TEGNode, list[TEGArc]] = {}
    for arc in arcs:
        src, dst = arc
        out_of.setdefault(src, []).append(arc)
        in_to.setdefault(dst, []).append(arc)

    # Visit indicator: y[i] = total inflow across all time copies, vehicles.
    for i in range(n):
        if i == depot:
            continue
        inflow_terms = [
            x[(arc, k)]
            for tau in nodes_at[i]
            for arc in in_to.get((i, tau), [])
            for k in range(num_vehicles)
        ]
        if inflow_terms:
            model.Add(sum(inflow_terms) == y[i])
        else:
            model.Add(y[i] == 0)

    # Flow conservation per vehicle at every customer time copy.
    for k in range(num_vehicles):
        for i in range(n):
            if i == depot:
                continue
            for tau in nodes_at[i]:
                node = (i, tau)
                inflow = [x[(arc, k)] for arc in in_to.get(node, [])]
                outflow = [x[(arc, k)] for arc in out_of.get(node, [])]
                # If neither side has any arc this becomes 0 == 0; harmless.
                model.Add(sum(inflow) == sum(outflow))

    # Vehicle starts at (depot, 0) at most once; depot inflow = depot outflow.
    for k in range(num_vehicles):
        depot_out = [x[(arc, k)] for arc in out_of.get((depot, 0), [])]
        depot_in_all = [
            x[(arc, k)]
            for tau in nodes_at[depot]
            for arc in in_to.get((depot, tau), [])
        ]
        if depot_out:
            model.Add(sum(depot_out) <= 1)
        if depot_out or depot_in_all:
            model.Add(sum(depot_out) == sum(depot_in_all))

    # Objective: maximise collected profit.
    model.Maximize(sum(int(profits[i]) * y[i] for i in y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    cpsat_status = solver.Solve(model)
    mapped = _CPSAT_STATUS_MAP.get(cpsat_status, 0)
    if mapped != 1:
        return SolveResult(routes=[], total_distance=0, status=mapped)

    # Reconstruct routes by following arcs from (depot, 0) per vehicle.
    routes: list[list[int]] = []
    schedule: list[dict] = []
    total_distance = 0
    for k in range(num_vehicles):
        current: TEGNode = (depot, 0)
        route_nodes: list[int] = [depot]
        schedule.append({
            "vehicle": k,
            "node": depot,
            "arrive": 0,
            "leave": 0,
        })
        guard = 0
        while True:
            guard += 1
            if guard > len(arcs) + 2:
                raise RuntimeError("TEG route reconstruction did not terminate")
            next_arc: TEGArc | None = None
            for arc in out_of.get(current, []):
                if solver.Value(x[(arc, k)]) == 1:
                    next_arc = arc
                    break
            if next_arc is None:
                break
            src, dst = next_arc
            j, tau_j = dst
            tau_i = src[1]
            shift_idx = _shift_of(tau_i, shift_windows)
            total_distance += int(shift_travel[shift_idx][src[0]][j])
            arrive = tau_j
            leave = arrive + (0 if j == depot else int(service_time))
            schedule.append({
                "vehicle": k,
                "node": j,
                "arrive": arrive,
                "leave": leave,
            })
            route_nodes.append(j)
            current = dst
            if j == depot:
                break
        routes.append(route_nodes)

    visited: set[int] = set()
    for route in routes:
        for node in route:
            if node != depot:
                visited.add(node)
    dropped = sorted(i for i in range(n) if i != depot and i not in visited)

    total_profit = int(round(solver.ObjectiveValue()))

    return SolveResult(
        routes=routes,
        total_distance=total_distance,
        status=1,
        dropped=dropped,
        schedule=schedule,
        total_profit=total_profit,
    )
