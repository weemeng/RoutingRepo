from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class SolveResult:
    routes: list[list[int]]
    total_distance: int
    status: int
    loads: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    schedule: list[dict] = field(default_factory=list)


def solve_routing(
    distance_matrix: Sequence[Sequence[int]],
    num_vehicles: int = 1,
    depot: int = 0,
    time_limit_seconds: int = 5,
    first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> SolveResult:
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = first_solution_strategy
    params.local_search_metaheuristic = local_search_metaheuristic
    params.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return SolveResult(routes=[], total_distance=0, status=routing.status())

    routes: list[list[int]] = []
    total = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(manager.IndexToNode(index))
        routes.append(route)
        total += route_distance

    return SolveResult(routes=routes, total_distance=total, status=routing.status())


def solve_cvrp(
    distance_matrix: Sequence[Sequence[int]],
    demands: Sequence[int],
    vehicle_capacities: Sequence[int],
    depot: int = 0,
    time_limit_seconds: int = 5,
    drop_node_penalty: int | None = None,
    first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> SolveResult:
    n = len(distance_matrix)
    num_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        list(vehicle_capacities),
        True,
        "Capacity",
    )

    if drop_node_penalty is not None:
        for node in range(n):
            if node == depot:
                continue
            routing.AddDisjunction([manager.NodeToIndex(node)], int(drop_node_penalty))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = first_solution_strategy
    params.local_search_metaheuristic = local_search_metaheuristic
    params.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return SolveResult(routes=[], total_distance=0, status=routing.status())

    routes: list[list[int]] = []
    loads: list[int] = []
    total = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            route_load += demands[node]
            prev = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
        route.append(manager.IndexToNode(index))
        routes.append(route)
        loads.append(route_load)
        total += route_distance

    visited = {node for route in routes for node in route}
    dropped = sorted(i for i in range(n) if i not in visited and i != depot)

    return SolveResult(
        routes=routes,
        total_distance=total,
        status=routing.status(),
        loads=loads,
        dropped=dropped,
    )


def solve_vrptw(
    distance: Sequence[Sequence[int]],
    demands: Sequence[int],
    time_windows: Sequence[tuple[int, int]],
    vehicle_capacities: Sequence[int],
    service_time: int,
    horizon: int,
    depot: int = 0,
    wait_slack: int = 60,
    time_limit_seconds: int = 5,
    first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> SolveResult:
    n = len(distance)
    num_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, list(vehicle_capacities), True, "Capacity",
    )

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        st = 0 if from_node == depot else service_time
        return st + distance[from_node][to_node]

    time_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_idx, wait_slack, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(n):
        if node == depot:
            continue
        a, b = time_windows[node]
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(a), int(b))

    depot_open, depot_close = time_windows[depot]
    for v in range(num_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(int(depot_open), int(depot_close))
        time_dim.CumulVar(routing.End(v)).SetRange(int(depot_open), int(depot_close))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = first_solution_strategy
    params.local_search_metaheuristic = local_search_metaheuristic
    params.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return SolveResult(routes=[], total_distance=0, status=routing.status())

    routes: list[list[int]] = []
    loads: list[int] = []
    schedule: list[dict] = []
    total = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrive = solution.Min(time_dim.CumulVar(index))
            depart = arrive + (0 if node == depot else service_time)
            schedule.append({
                "vehicle": vehicle_id,
                "node": node,
                "arrive": arrive,
                "leave": depart,
            })
            route.append(node)
            route_load += demands[node]
            prev = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
        end_node = manager.IndexToNode(index)
        end_arrive = solution.Min(time_dim.CumulVar(index))
        schedule.append({
            "vehicle": vehicle_id,
            "node": end_node,
            "arrive": end_arrive,
            "leave": end_arrive,
        })
        route.append(end_node)
        routes.append(route)
        loads.append(route_load)
        total += route_distance

    return SolveResult(
        routes=routes,
        total_distance=total,
        status=routing.status(),
        loads=loads,
        schedule=schedule,
    )


def solve_pdptw(
    distance: Sequence[Sequence[int]],
    demands: Sequence[int],
    time_windows: Sequence[tuple[int, int]],
    pairs: Sequence[tuple[int, int]],
    vehicle_capacities: Sequence[int],
    service_time: int,
    horizon: int,
    depot: int = 0,
    wait_slack: int = 60,
    time_limit_seconds: int = 10,
    max_ride_time: int | Sequence[int] | None = None,
    first_solution_strategy: int = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    local_search_metaheuristic: int = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
) -> SolveResult:
    n = len(distance)
    num_vehicles = len(vehicle_capacities)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, list(vehicle_capacities), True, "Capacity",
    )

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        st = 0 if from_node == depot else service_time
        return st + distance[from_node][to_node]

    time_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_idx, wait_slack, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(n):
        if node == depot:
            continue
        a, b = time_windows[node]
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(a), int(b))

    depot_open, depot_close = time_windows[depot]
    for v in range(num_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(int(depot_open), int(depot_close))
        time_dim.CumulVar(routing.End(v)).SetRange(int(depot_open), int(depot_close))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    for pickup_node, delivery_node in pairs:
        p_idx = manager.NodeToIndex(pickup_node)
        d_idx = manager.NodeToIndex(delivery_node)
        routing.AddPickupAndDelivery(p_idx, d_idx)
        routing.solver().Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
        routing.solver().Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

    if max_ride_time is not None:
        if isinstance(max_ride_time, int):
            ride_limits = [max_ride_time] * len(pairs)
        else:
            ride_limits = list(max_ride_time)
        for (pickup_node, delivery_node), limit in zip(pairs, ride_limits):
            p_idx = manager.NodeToIndex(pickup_node)
            d_idx = manager.NodeToIndex(delivery_node)
            routing.solver().Add(
                time_dim.CumulVar(d_idx) - time_dim.CumulVar(p_idx) <= int(limit)
            )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = first_solution_strategy
    params.local_search_metaheuristic = local_search_metaheuristic
    params.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return SolveResult(routes=[], total_distance=0, status=routing.status())

    routes: list[list[int]] = []
    loads: list[int] = []
    schedule: list[dict] = []
    total = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        route_distance = 0
        peak_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrive = solution.Min(time_dim.CumulVar(index))
            depart = arrive + (0 if node == depot else service_time)
            peak_load = max(peak_load, solution.Value(routing.GetDimensionOrDie("Capacity").CumulVar(index)))
            schedule.append({
                "vehicle": vehicle_id,
                "node": node,
                "arrive": arrive,
                "leave": depart,
            })
            route.append(node)
            prev = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
        end_node = manager.IndexToNode(index)
        end_arrive = solution.Min(time_dim.CumulVar(index))
        schedule.append({
            "vehicle": vehicle_id,
            "node": end_node,
            "arrive": end_arrive,
            "leave": end_arrive,
        })
        route.append(end_node)
        routes.append(route)
        loads.append(peak_load)
        total += route_distance

    return SolveResult(
        routes=routes,
        total_distance=total,
        status=routing.status(),
        loads=loads,
        schedule=schedule,
    )
