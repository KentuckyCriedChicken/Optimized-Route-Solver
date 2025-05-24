import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

destination = np.array([-33.894, 151.212])


#load the data
df = pd.read_csv('testdata.csv')

#Sorting the data
drivers = df[df['role'] == 'driver']
riders = df[df['role'] == 'rider']

driver_coord = drivers[['x', 'y']].to_numpy()
rider_coord = riders[['x', 'y']].to_numpy()



#Calculating the distance matrix
all_coords = np.vstack((driver_coord, rider_coord, destination))
distance_matrix = np.zeros((11, 11))

for i in range(11):
    for j in range(11):
        distance_matrix[i][j] = np.linalg.norm(all_coords[i] - all_coords[j])



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



#Define and register a transit callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node, to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

#Define cost of each arc
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)



#Add Distance constraint
dimension_name = "Distance"
routing.AddDimension(
    transit_callback_index,
    0,                          # no slack
    3000,                       # vehicle maximum travel distance
    True,                       # start cumul to zero
    dimension_name,
)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)



#Setting first solution heuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

#Solve the problem
solution = routing.SolveWithParameters(search_parameters)



#Print solution on console 









plt.scatter(df['x'], df['y'])
plt.scatter(destination[0], destination[1], c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('map_v1')
plt.show()

