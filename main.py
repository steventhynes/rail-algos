import networkx as nx

cities = {}

# Utility/helper functions

def city_import(file):
    pass

# from the list of cities, build the graph with no edges (the empty solution) (O(n) time)
def empty_solution(cities):
    graph = nx.empty_graph(cities) #each node has the properties of the dict
    graph.add_nodes_from(cities)
    return graph

# build graph of cities with all edges and edge weights (O(n^2) time)
def complete_solution(cities):
    graph = empty_solution(cities)
    edges_to_add = []
    for i in graph.nodes:
        for j in graph.nodes:
            if i is not j:
                dist = ((i["lat"] - j["lat"]) ** 2 + (i["lon"] - j["lon"]) ** 2) ** 0.5
                edges_to_add.append((i, j, dist))
    graph.add_weighted_edges_from(edges_to_add, weight="dist")
    return graph

# given the solution edges in the graph, calculate the score (O(n^3) time)
def evaluate_solution(graph):
    all_pairs_shortest_paths = nx.all_pairs_shortest_path_length(graph)
    score = 0.0
    for i in graph.nodes:
        for j in graph.nodes:
            if i is not j:
                score += (i["weight"] * j["weight"]) / all_pairs_shortest_paths[i][j] # multiplied weights of cities divided by distance between them
    return score


# Algorithms

# continually add shortest edges until k miles is reached.
# If k=inf, then keep going until score starts decreasing
def mst_buildup(cities, k="inf"):
    pass

# Find shortest path spanning trees (Dijkstra) for highest-weighted trees until k miles is reached.
# If k=inf, then keep going until score starts decreasing
def shortest_path_spanning_tree_buildup(cities, k="inf"):
    pass

# Start with empty set of edges and build up to a solution, eliminating bad ones along the way.
# If k=inf, then don't disqualify solutions based on k
def backtracking(cities, k="inf"):
    pass

# Start with all possible solutions and systematically eliminate them by keeping a running maximum
# score and eliminating candidates via their upper bound.
# If k=inf, then don't disqualify solutions based on k
def branch_and_bound(cities, k="inf"):
    pass

# Local search with a perturbation when a local optimum is found. Perturbation is made to be orthogonal
# to previous local minima.
# If k=inf, then don't disqualify solutions based on k
def iterated_local_search(cities, timeout, k="inf"):
    pass

# Local search with a steadily decreasing chance to perturb to a random solution
# If k=inf, then don't disqualify solutions based on k
def simulated_annealing(cities, timeout, k="inf"):
    pass

# Local search that allows for worsening moves at local optima, and keeps a record of previously-visited states
# If k=inf, then don't disqualify solutions based on k
def tabu_search(cities, timeout, k="inf"):
    pass

# Local search inspired by evolution
# If k=inf, then don't disqualify solutions based on k
def genetic_algorithm(cities, timeout, k="inf"):
    pass

# read in args and stuff
if __name__ == "__main__":
    pass 