import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, cos, sin, asin, sqrt
import folium
from folium import PolyLine, Marker



#load the data
df = pd.read_csv('rawdata.csv')

#Sorting the data
drivers = df[df['role'] == 'Yes']
riders = df[df['role'] == 'No']

all_coords = list(zip(df["Latitude"], df["Longitude"]))

#Append the final destination to list of coordinates
all_coords.append((-33.915200,151.267898))


#Retrieving the distance matrix
distance_matrix = np.load("distance_matrix.npy")



#Set demands and vehicle capacities
demands = [0]*2 + [1]*6 + [0]*2 + [1]*2 + [0]*1 + [1]*2 + [0]*1 + [1]*1 + [0]*3
vehicle_capacities = [7, 4, 4, 4, 2, 1, 1, 3, 4]



#Initialize the Routing Index Manager
start_nodes = [0, 1, 8, 9, 12, 15, 17, 18, 19]
end_nodes = [20, 20, 20, 20, 20, 20, 20, 20, 20]
manager = pywrapcp.RoutingIndexManager(
    21,                    # total number of locations (nodes)
    len(drivers),                  # number of vehicles
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
    50000,                       # vehicle maximum travel distance
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
    for vehicle in range(len(drivers)):
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



#Plotting and visualizing the routes
def plot_routes_on_map(manager, routing, solution, all_coords):
    avg_lat = np.mean([lat for lat, lon in all_coords])
    avg_lon = np.mean([lon for lat, lon in all_coords])
    route_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

    for i, (lat, lon) in enumerate(all_coords):
        if i == 20:
            folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='flag'),
                          tooltip="Destination").add_to(route_map)
        elif df.loc[i, 'role'] == 'Yes':
            folium.Marker([lat, lon], icon=folium.Icon(color='blue', icon='car'),
                          tooltip=f"Driver {i}").add_to(route_map)
        else:
            folium.Marker([lat, lon], icon=folium.Icon(color='green', icon='user'),
                          tooltip=f"Rider {i}").add_to(route_map)

    colors = ['blue', 'orange', 'purple', 'green', 'black', 'darkred', 'cadetblue', 'pink', 'gray']

    for vehicle_id in range(routing.vehicles()):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        route_coords = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_coords.append(all_coords[node_index])
            index = solution.Value(routing.NextVar(index))
        node_index = manager.IndexToNode(index)
        route_coords.append(all_coords[node_index])  

        folium.PolyLine(
            locations=route_coords,
            color=colors[vehicle_id % len(colors)],
            weight=5,
            opacity=0.7,
            tooltip=f"Vehicle {vehicle_id}"
        ).add_to(route_map)

    route_map.save("optimized_routes_map.html")
    print("Map saved as 'optimized_routes_map.html'")

if solution:
    plot_routes_on_map(manager, routing, solution, all_coords)

