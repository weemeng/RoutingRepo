from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class SolveResult:
    routes: list[list[int]]
    total_distance: int
    status: int


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
