import networkx as nx

cities = {}

# Utility/helper functions

def city_import(file):
    pass

# from the list of cities, build the graph (the empty solution)
def build_graph(cities):
    graph = nx.graph()
    graph.add_nodes_from(cities) #each node has the properties of the dict

# given the solution edges in the graph, calculate the score
def evaluate_solution(graph):
    pass 


# Algorithms

# continually add shortest edges until k miles is reached
def mst_buildup(cities, k):
    pass

# Find shortest path spanning trees (Dijkstra) for highest-weighted trees until k miles is reached
def shortest_path_spanning_tree_buildup(cities, k):
    pass

# Start with empty set of edges and build up to a solution, eliminating bad ones along the way
def backtracking(cities, k):
    pass

# Start with all possible solutions and systematically eliminate them by keeping a running maximum score and eliminating candidates via their upper bound.
def branch_and_bound(cities, k):
    pass

# Local search with a perturbation when a local optimum is found. Perturbation is made to be orthogonal to previous local minima.
def iterated_local_search(cities, k, timeout):
    pass

# Local search with a steadily decreasing chance to perturb to a random solution
def simulated_annealing(cities, k, timeout):
    pass

# Local search that allows for worsening moves at local optima, and keeps a record of previously-visited states
def tabu_search(cities, k, timeout):
    pass

# Local search inspired by evolution
def genetic_algorithm(cities, k, timeout):
    pass

# read in args and stuff
if __name__ == "__main__":
    pass 