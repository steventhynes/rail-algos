from cmath import inf
import networkx as nx
from collections import deque
import plotly.graph_objects as go

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

# Gets only the connected nodes and gets the all-pairs distance between them
def all_pairs_shortest_paths(graph):
    connected_nodes = [node for node in graph.nodes if graph.edges(node)]
    new_graph = nx.induced_subgraph(graph, connected_nodes)
    return nx.floyd_warshall(new_graph, weight='dist')

# given the solution edges in the graph, calculate the score (O(n^3) time)
def evaluate_solution(graph):
    apsp = all_pairs_shortest_paths(graph) # This WILL need to be optimized or approximated; takes way too long
    score = 0.0
    for i in apsp:
        for j in apsp[i]:
            if i is not j and apsp[i][j] < inf:
                score += (graph.nodes[i]["population"] * graph.nodes[j]["population"]) / apsp[i][j] # multiplied weights of cities divided by distance between them
    return score

# display the solution in a map
def display_solution(graph):
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = [graph.nodes[node]['lon'] for node in graph.nodes],
    lat = [graph.nodes[node]['lat'] for node in graph.nodes],
    hoverinfo = 'text',
    text = [node for node in graph.nodes],
    mode = 'markers',
    marker = dict(
        size = 2,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))
    for edge in graph.edges:
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'USA-states',
                lon = [graph.nodes[edge[0]]['lon'], graph.nodes[edge[1]]['lon']],
                lat = [graph.nodes[edge[0]]['lat'], graph.nodes[edge[1]]['lat']],
                mode = 'lines',
                line = dict(width = 1,color = 'red'),
                opacity = 1,
            )
        )
    fig.show()


# Algorithms

# continually add highest-weight edges until k miles is reached.
def greedy_buildup(cities, k):
    empty = empty_solution(cities)
    complete = complete_solution(cities)
    sorted_edges = deque(sorted(complete.edges.data(), key=lambda x: (complete.nodes[x[0]]['population'] * complete.nodes[x[1]]['population']) / (x[2]['dist'] ** 2), reverse=True))
    prev_sol = None
    curr_sol = empty
    cost = 0.0
    while sorted_edges:
        prev_sol = curr_sol
        curr_sol = curr_sol.copy()
        new_edge = sorted_edges.popleft()
        curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
        cost += new_edge[2]['dist']
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