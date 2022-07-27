from cmath import inf
import networkx as nx
from numpy import average
import plotly.graph_objects as go
from random import sample
from itertools import product


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
def complete_solution(cities, dist_limit=10):
    graph = empty_solution(cities)
    edges_to_add = []
    visited_edges = set()
    for i in graph.nodes:
        for j in graph.nodes:
            if i is not j and (i, j) not in visited_edges:
                dist = ((graph.nodes[i]["lat"] - graph.nodes[j]["lat"]) ** 2 + (graph.nodes[i]["lon"] - graph.nodes[j]["lon"]) ** 2) ** 0.5
                if dist_limit is not None and dist <= dist_limit:
                    edges_to_add.append((i, j, dist))
                visited_edges.add((i, j))
                visited_edges.add((j, i))
    graph.add_weighted_edges_from(edges_to_add, weight="dist")
    return graph

# Gets only the connected nodes and gets the all-pairs distance between them
def all_pairs_shortest_paths(graph):
    connected_nodes = [node for node in graph.nodes if graph.edges(node)]
    new_graph = nx.induced_subgraph(graph, connected_nodes)
    apsp = nx.floyd_warshall(new_graph, weight='dist')
    new_apsp = {}
    for i in graph.nodes:
        new_apsp[i] = {}
        for j in graph.nodes:
            try:
                new_apsp[i][j] = apsp[i][j]
            except KeyError:
                if i == j:
                    new_apsp[i][j] = 0.0
                else:
                    new_apsp[i][j] = inf
    return new_apsp

# Return the score of an edge or node pair
def score_calc(pop1, pop2, distance):
    # return (pop1 * pop2) * (10 ** -10) / (distance ** 2)
    return (pop1 * pop2) * (10 ** -10) / distance

# given the solution edges in the graph, calculate the score (O(n^3) time)
def evaluate_solution(graph):
    apsp = all_pairs_shortest_paths(graph) # This WILL need to be optimized or approximated; takes way too long
    score = 0.0
    for i in apsp:
        for j in apsp[i]:
            if i is not j and apsp[i][j] < inf:
                score += score_calc(graph.nodes[i]["population"], graph.nodes[j]["population"], apsp[i][j])
    return apsp, score

def approx_evaluate_solution(graph, most_important_nodes, num_random_nodes):
    nodes = set(most_important_nodes + sample(graph.nodes, num_random_nodes))
    score = 0
    for node in nodes:
        path_lengths = nx.shortest_path_length(graph, node, weight='dist')
        for target in path_lengths:
            if node == target:
                continue
            score += score_calc(graph.nodes[node]['population'], graph.nodes[target]['population'], path_lengths[target])
    return score * len(graph.nodes)/len(nodes)


# Add an edge to the graph and evaluate the solution more efficiently based on previous all-pairs-shortest-path length dict. O(n^2).
def add_edge_and_eval(graph, new_edge, prev_apsp):
    graph.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
    new_apsp = prev_apsp # copy?
    new_score = 0.0
    connected_nodes = [node for node in graph.nodes if graph.edges(node)]
    endpoint1, endpoint2 = new_edge[:2]
    for node in connected_nodes:
        # if endpoint1 not in new_apsp:
        #     new_apsp[endpoint1] = {}
        new_apsp[endpoint1][node] = min(prev_apsp[endpoint1][node], prev_apsp[endpoint2][node] + new_edge[2]['dist'])
        if endpoint1 != node:
            new_score += score_calc(graph.nodes[endpoint1]["population"], graph.nodes[node]["population"], new_apsp[endpoint1][node])
        new_apsp[endpoint2][node] = min(prev_apsp[endpoint2][node], prev_apsp[endpoint1][node] + new_edge[2]['dist'])
        # if endpoint2 not in new_apsp:
        #     new_apsp[endpoint2] = {}
        if endpoint2 != node:
            new_score += score_calc(graph.nodes[endpoint2]["population"], graph.nodes[node]["population"], new_apsp[endpoint2][node])
    for node1 in connected_nodes:
        if node1 in [endpoint1, endpoint2]:
            continue
        for node2 in connected_nodes:
            new_apsp[node1][node2] = min(prev_apsp[node1][node2], new_apsp[endpoint1][node1] + new_edge[2]['dist'] + new_apsp[endpoint2][node2],
                                        new_apsp[endpoint1][node2] + new_edge[2]['dist'] + new_apsp[endpoint2][node1])
            if node1 is not node2:
                new_score += score_calc(graph.nodes[node1]["population"], graph.nodes[node2]["population"], new_apsp[node1][node2])
    return graph, new_apsp, new_score

def approx_add_edge_and_eval(graph, new_edge):
    graph.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
    approx_evaluate_solution(graph, [])
            
def remove_edge_and_eval(graph, edge_to_remove, prev_apsp):
    connected_nodes = [node for node in graph.nodes if graph.edges(node)]
    endpoint1, endpoint2 = edge_to_remove[:2]
    graph.remove_edge(endpoint1, endpoint2)
    new_apsp = prev_apsp # copy?
    new_score = 0.0
    for node1 in connected_nodes:
        for node2 in connected_nodes:
            if prev_apsp[node1][node2] == min(prev_apsp[node1][endpoint1] + edge_to_remove[2]['dist'] + prev_apsp[node2][endpoint2], prev_apsp[node1][endpoint2] + edge_to_remove[2]['dist'] + prev_apsp[node2][endpoint1]):
                try:
                    new_apsp[node1][node2] = nx.shortest_path_length(graph, node1, node2, weight='dist')
                except nx.exception.NetworkXNoPath:
                    new_apsp[node1][node2] = inf
            else:
                new_apsp[node1][node2] = prev_apsp[node1][node2]
            if node1 is not node2:
                new_score += score_calc(graph.nodes[node1]["population"], graph.nodes[node2]["population"], new_apsp[node1][node2])
    return graph, new_apsp, new_score

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
        color = 'rgb(0, 0, 255)',
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

def display_solution_continuous(graph):
    pass


def optimize_hyperparameters_grid(algo, cities, k, *args, score_func=lambda x: approx_evaluate_solution(x, [], len(x.nodes)), num_runs=1):
    args_arrays = product(*args)
    print(args)

    curr_best = None
    curr_best_val = -inf

    for i in args_arrays:
        print(i)
        new_vals = [score_func(algo(cities, k, *i)) for run in range(num_runs)]
        new_val = average(new_vals)
        if new_val > curr_best_val:
            curr_best = i
            curr_best_val = new_val
    
    return curr_best

    
