from cmath import inf
import networkx as nx
from collections import deque

# Utility/helper functions

def cities_from_file(filename):
    cities = []
    with open(filename, 'r') as cityfile:
        lines = cityfile.readlines()
        for line in lines[1:]:
            new_city = {}
            splitline = line.strip().split(',')
            new_city['name'], new_city['state'] = splitline[:2]
            new_city['population'] = int(splitline[2])
            new_city['lat'] = float(splitline[3])
            new_city['lon'] = float(splitline[4])
            cities.append(('{}, {}'.format(new_city['name'], new_city["state"]), new_city))
    return cities

# from the list of cities, build the graph with no edges (the empty solution) (O(n) time)
def empty_solution(cities):
    graph = nx.empty_graph(cities) #each node has the properties of the dict
    graph.add_nodes_from(cities)
    return graph

# build graph of cities with all edges and edge weights (O(n^2) time)
def complete_solution(cities):
    graph = empty_solution(cities)
    edges_to_add = []
    visited_edges = set()
    for i in graph.nodes:
        for j in graph.nodes:
            if i is not j and (i, j) not in visited_edges:
                dist = ((graph.nodes[i]["lat"] - graph.nodes[j]["lat"]) ** 2 + (graph.nodes[i]["lon"] - graph.nodes[j]["lon"]) ** 2) ** 0.5
                edges_to_add.append((i, j, dist))
                visited_edges.add((i, j))
                visited_edges.add((j, i))
    graph.add_weighted_edges_from(edges_to_add, weight="dist")
    return graph

# given the solution edges in the graph, calculate the score (O(n^3) time)
def evaluate_solution(graph):
    all_pairs_shortest_paths = nx.floyd_warshall(graph, weight="dist") # This WILL need to be optimized or approximated; takes way too long
    score = 0.0
    for i in graph.nodes:
        for j in graph.nodes:
            if i is not j and all_pairs_shortest_paths[i][j] < inf:
                score += (graph.nodes[i]["population"] * graph.nodes[j]["population"]) / all_pairs_shortest_paths[i][j] # multiplied weights of cities divided by distance between them
    return score


# Algorithms

# continually add shortest edges until k miles is reached.
def greedy_buildup(cities, k):
    empty = empty_solution(cities)
    complete = complete_solution(cities)
    sorted_edges = deque(sorted(complete.edges, key=lambda x: x["dist"]))
    prev_sol = None
    curr_sol = empty
    cost = 0.0
    while sorted_edges:
        prev_sol = curr_sol
        curr_sol = curr_sol.copy()
        new_edge = sorted_edges.popleft()
        curr_sol.add_weighted_edges_from([new_edge])
        cost += new_edge["dist"]
        if cost > k:
            return prev_sol
    return curr_sol

# Find shortest path spanning trees (Dijkstra) for highest-weighted trees until k miles is reached.
def shortest_path_spanning_tree_buildup(cities, k):
    pass

# Start with empty set of edges and build up to a solution, eliminating bad ones along the way.
def backtracking(cities, k):
    pass

# Start with all possible solutions and systematically eliminate them by keeping a running maximum
# score and eliminating candidates via their upper bound.
def branch_and_bound(cities, k):
    pass

# Local search with a perturbation when a local optimum is found. Perturbation is made to be orthogonal
# to previous local minima.
def iterated_local_search(cities, timeout, k):
    pass

# Local search with a steadily decreasing chance to perturb to a random solution
def simulated_annealing(cities, timeout, k):
    pass

# Local search that allows for worsening moves at local optima, and keeps a record of previously-visited states
def tabu_search(cities, timeout, k):
    pass

# Local search inspired by evolution
def genetic_algorithm(cities, timeout, k):
    pass

# read in args and stuff
if __name__ == "__main__":
    pass 