import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, cos, sin, asin, sqrt

destination = np.array([-33.894, 151.212])


#load the data
df = pd.read_csv('testdata.csv')

#Sorting the data
drivers = df[df['role'] == 'driver']
riders = df[df['role'] == 'rider']

driver_coord = drivers[['x', 'y']].to_numpy()
rider_coord = riders[['x', 'y']].to_numpy()



#Calculating the distance matrix
def haversine(coord1, coord2):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  
    return c * r

all_coords = np.vstack((driver_coord, rider_coord, destination))
distance_matrix = np.zeros((11, 11))

for i in range(11):
    for j in range(11):
        distance_matrix[i][j] = haversine(all_coords[i], all_coords[j])



#Set demands and vehicle capacities
demands = [0] * 3 + [1] * 7 + [0]
vehicle_capacities = [3, 2, 4]



#Initialize the Routing Index Manager
start_nodes = [0, 1, 2]
end_nodes = [10, 10, 10]
manager = pywrapcp.RoutingIndexManager(
    len(all_coords),                    # total number of locations (nodes)
    len(driver_coord),                  # number of vehicles
    start_nodes,                        # list of start nodes (one per vehicle)
    end_nodes                           # list of end nodes (one per vehicle)
)

#Initialize the Routing Model
routing = pywrapcp.RoutingModel(manager)
routing.SetFixedCostOfAllVehicles(100)



#Define and register a transit callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    distance = distance_matrix[from_node, to_node]
    return distance 

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

#Define and register a demand callback
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return demands[from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)



#Define cost of each arc
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)



#Add Distance constraint
dimension_name = "Distance"
routing.AddDimension(
    transit_callback_index,
    0,                          # no slack
    1000,                       # vehicle maximum travel distance
    True,                       # start cumul to zero
    dimension_name,
)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(1000)



#Add Capacity constraint
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # null capacity slack
    vehicle_capacities,  # vehicle maximum capacities
    True,  # start cumul to zero
    "Capacity"
)



#Setting first solution heuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
)

#Solve the problem
solution = routing.SolveWithParameters(search_parameters)



#Print solution on console 
def print_solution(manager, routing, solution):
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    for vehicle in range(len(driver_coord)):
        if not routing.IsVehicleUsed(solution, vehicle):
            continue
        index = routing.Start(vehicle)
        plan_output = f"Route for vehicle {vehicle}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(previous_index)
            to_node = manager.IndexToNode(index)
            route_distance += distance_matrix[from_node][to_node]

        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Maximum of the route distances: {max_route_distance}m")


if solution:
    print_solution(manager, routing, solution)
else:
    print("No solution found !")





#Plotting and visualizing the routes
def plot_routes(manager, routing, solution, all_coords):
    plt.figure(figsize=(10, 8))

    # Plot all points
    for i, (lat, lon) in enumerate(all_coords):
        if i < 3:
            plt.plot(lon, lat, 'bs', markersize=10)  # blue square for drivers
            plt.text(lon, lat, f'D{i}', fontsize=9, color='blue')
        elif i < 10:
            plt.plot(lon, lat, 'go', markersize=8)   # green circle for riders
            plt.text(lon, lat, f'R{i-3}', fontsize=9, color='green')
        else:
            plt.plot(lon, lat, 'r*', markersize=15)  # red star for destination
            plt.text(lon, lat, 'Dest', fontsize=10, color='red')

    # Plot vehicle routes
    colors = ['b', 'orange', 'purple']  # one color per vehicle

    for vehicle in range(routing.vehicles()):
        if not routing.IsVehicleUsed(solution, vehicle):
            continue
        index = routing.Start(vehicle)
        route_lat = []
        route_lon = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            lat, lon = all_coords[node_index]
            route_lat.append(lat)
            route_lon.append(lon)
            index = solution.Value(routing.NextVar(index))
        # Add destination
        node_index = manager.IndexToNode(index)
        lat, lon = all_coords[node_index]
        route_lat.append(lat)
        route_lon.append(lon)

        plt.plot(route_lon, route_lat, color=colors[vehicle], linewidth=2, label=f'Vehicle {vehicle}')

    plt.title("Optimized Routes")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if solution:
    plot_routes(manager, routing, solution, all_coords)