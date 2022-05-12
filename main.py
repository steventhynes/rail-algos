from cmath import inf
from dataclasses import dataclass
from math import ceil
import networkx as nx
from collections import deque
import heapq as hq
import plotly.graph_objects as go
import disjoint_set as ds
from random import shuffle

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
    return (pop1 * pop2) / (distance ** 2)

# given the solution edges in the graph, calculate the score (O(n^3) time)
def evaluate_solution(graph):
    apsp = all_pairs_shortest_paths(graph) # This WILL need to be optimized or approximated; takes way too long
    score = 0.0
    for i in apsp:
        for j in apsp[i]:
            if i is not j and apsp[i][j] < inf:
                score += score_calc(graph.nodes[i]["population"], graph.nodes[j]["population"], apsp[i][j])
    return apsp, score

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
    sorted_edges = deque(sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True))
    curr_sol = empty
    cost = 0.0
    while sorted_edges:
        new_edge = sorted_edges.popleft()
        curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
        cost += new_edge[2]['dist']
        if cost > k:
            curr_sol.remove_edge(new_edge[0], new_edge[1])
            return curr_sol
    return curr_sol

# use 0-1 knapsack algorithm to add highest-weight edges up to k distance. O(n^2 * k) pseudo-polynomial time.
def knapsack_buildup(cities, k):
    empty = empty_solution(cities)
    complete = complete_solution(cities)
    edges = list(complete.edges.data())
    table = [[0 for num in range(k+1)] for edge in range(len(edges)+1)] # table[i][j] is the maximum sum of edges[1...i] summing to at most j distance
    # fill out table
    for i in range(len(table))[1:]:
        for j in range(len(table[0]))[1:]:
            dist = ceil(edges[i-1][2]['dist'])
            score = score_calc(complete.nodes[edges[i-1][0]]['population'], complete.nodes[edges[i-1][1]]['population'], dist)
            if dist > j:
                table[i][j] = table[i-1][j]
            else:
                table[i][j] = max(table[i-1][j], table[i-1][j-dist] + score)
    # backtrack to get edge set
    i = len(edges)
    j = k
    to_return = empty
    while True:
        if i == 0:
            break
        if table[i][j] > table[i-1][j] and j >= ceil(edges[i-1][2]['dist']): # If this item was used
            to_return.add_weighted_edges_from([edges[i-1]])
            j -= ceil(edges[i-1][2]['dist'])
            i -= 1
        else:
            i -= 1 # if this item was not used
    return to_return

# Build a maximum weight spanning tree, then add edges until k distance is reached. If quit is True, exits once spanning tree is reached.
def max_weight_spanning_tree_buildup(cities, k, quit=False):
    curr_sol = empty_solution(cities)
    complete = complete_solution(cities)
    sorted_edges = deque(sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True))
    cost = 0.0
    leftover_edges = deque()
    disj_set = ds.DisjointSet()
    while sorted_edges:
        new_edge = sorted_edges.popleft()
        if disj_set.connected(new_edge[0], new_edge[1]):
            leftover_edges.append(new_edge)
        else:
            curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
            disj_set.union(new_edge[0], new_edge[1])
            cost += new_edge[2]['dist']
            if cost > k:
                curr_sol.remove_edge(new_edge[0], new_edge[1])
                return curr_sol
    if not quit:
        while leftover_edges:
            new_edge = leftover_edges.popleft()
            curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
            cost += new_edge[2]['dist']
            if cost > k:
                curr_sol.remove_edge(new_edge[0], new_edge[1])
                return curr_sol
    return curr_sol

# Build a minimum distance spanning tree, then add edges until k distance is reached. If quit is True, exits once spanning tree is reached.
def min_dist_spanning_tree_buildup(cities, k, quit=False):
    curr_sol = empty_solution(cities)
    complete = complete_solution(cities)
    sorted_edges = deque(sorted(complete.edges.data(), key=lambda x: x[2]['dist']))
    cost = 0.0
    leftover_edges = deque()
    disj_set = ds.DisjointSet()
    while sorted_edges:
        new_edge = sorted_edges.popleft()
        if disj_set.connected(new_edge[0], new_edge[1]):
            leftover_edges.append(new_edge)
        else:
            curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
            disj_set.union(new_edge[0], new_edge[1])
            cost += new_edge[2]['dist']
            if cost > k:
                curr_sol.remove_edge(new_edge[0], new_edge[1])
                return curr_sol
    if not quit:
        while leftover_edges:
            new_edge = leftover_edges.popleft()
            curr_sol.add_edge(new_edge[0], new_edge[1], dist=new_edge[2]['dist'])
            cost += new_edge[2]['dist']
            if cost > k:
                curr_sol.remove_edge(new_edge[0], new_edge[1])
                return curr_sol
    return curr_sol

# Find shortest path spanning trees (Dijkstra) for highest-weighted trees until k miles is reached.
def shortest_path_spanning_tree_buildup(cities, k):
    pass

@dataclass
class PartialSolution:
    edge_array: list
    total_weight: float
    shortest_path_lengths: dict
    score: float
    depth: int
    graph: int

    def __init__(self, edge_array=None, total_weight=None, shortest_path_lengths=None, score=None, depth=None, graph=None):
        self.edge_array = edge_array
        self.total_weight =  total_weight
        self.shortest_path_lengths = shortest_path_lengths
        self.score = score
        self.depth = depth
        self.graph = graph
    
    def __lt__(self, other):
        self.total_weight < other.total_weight

    def __le__(self, other):
        self.total_weight <= other.total_weight

def backtracking_iterative_deepening(cities, k, score_threshold, iter_depth=5, max_heap_graphs=2500):
    complete = complete_solution(cities)
    complete_edges = sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True)
    partial_solutions_heap = []
    initial_sol = PartialSolution([], 0, *evaluate_solution(empty_solution(cities)), 0, empty_solution(cities))
    hq.heappush(partial_solutions_heap, (0, initial_sol)) #priority queue
    curr_best = initial_sol
    heap_graphs_count = 0
    try:
        while partial_solutions_heap:
            # Pop partial solution off the priority queue and initialize the next run
            print("partial sol popped off.")
            partial_sol = hq.heappop(partial_solutions_heap)[1]
            base_depth = partial_sol.depth
            stack = deque()
            # Rebuild graph if it's not there
            if partial_sol.graph is None:
                partial_sol.graph = empty_solution(cities)
                edges_to_add = [complete_edges[idx][:2] for idx in range(len(partial_sol.edge_array)) if partial_sol.edge_array[idx]]
                partial_sol.graph.add_edges_from(edges_to_add)
            # If graph is there, decrease the count
            else:
                heap_graphs_count -= 1
            stack.append(partial_sol)
            while stack:
                curr_sol = stack.pop()
                print("popped off from stack")
                # print(f"{curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.depth=}, {curr_sol.edge_array}")
                print(f"{curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.depth=}")

                if curr_sol.total_weight > k: # This solution is prohibitive -- the cost is too high.
                    continue # don't add expanded solutions to the stack
                
                if curr_sol.score > curr_best.score:
                    curr_best = curr_sol

                if score_threshold is not None and curr_sol.score >= score_threshold:
                    return curr_sol.graph

                elif curr_sol.depth == base_depth + iter_depth: # This is a partial solution; we've reached max depth for this iteration
                    # score = curr_sol.score / curr_sol.total_weight if curr_sol.total_weight else 0
                    print("adding partial sol to heap")
                    if heap_graphs_count >= max_heap_graphs:
                        curr_sol.graph = None
                    else:
                        heap_graphs_count += 1
                    score = (.25*(curr_sol.depth / len(complete_edges)) + .75*(curr_sol.score / score_threshold)) - (curr_sol.total_weight / k)
                    hq.heappush(partial_solutions_heap, (-score, curr_sol)) # score is negative because hq only supports minheap

                else: # None of the above are satisfied; create new solutions and add them to the stack
                    no_edge_sol = PartialSolution()
                    no_edge_sol.edge_array = curr_sol.edge_array + [False] * (base_depth + iter_depth - curr_sol.depth + 1)
                    no_edge_sol.total_weight = curr_sol.total_weight
                    no_edge_sol.shortest_path_lengths = curr_sol.shortest_path_lengths
                    no_edge_sol.score = curr_sol.score
                    no_edge_sol.depth = curr_sol.depth + 1
                    no_edge_sol.graph = curr_sol.graph
                    stack.append(no_edge_sol)

                    print("adding to stack")

                    for edge_add_idx in range(curr_sol.depth, base_depth + iter_depth):
                        yes_edge_sol = PartialSolution()
                        yes_edge_sol.edge_array = curr_sol.edge_array + [False] * (edge_add_idx - curr_sol.depth) + [True]
                        yes_edge_sol.graph, yes_edge_sol.shortest_path_lengths, yes_edge_sol.score = add_edge_and_eval(curr_sol.graph.copy(), complete_edges[edge_add_idx], curr_sol.shortest_path_lengths)
                        yes_edge_sol.total_weight = yes_edge_sol.graph.size(weight='dist')
                        yes_edge_sol.depth = edge_add_idx+1
                        stack.append(yes_edge_sol)
                    
    except KeyboardInterrupt:
        print(curr_best.edge_array)
        return_graph = empty_solution(cities)
        edges_to_add = [complete_edges[idx][:2] for idx in range(len(curr_best.edge_array)) if curr_sol.edge_array[idx]]
        return_graph.add_edges_from(edges_to_add)
        return return_graph


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


#tests

def test_add_edge_and_eval():
    cities = cities_from_file('data/us-cities-top-1k.csv')
    complete = complete_solution(cities)
    greedy_sol = greedy_buildup(cities, 50)
    prev_apsp, prev_score = evaluate_solution(greedy_sol)
    la_ny = "Los Angeles, California", "New York, New York", complete.edges["Los Angeles, California", "New York, New York"]
    test_graph, test_apsp, test_score = add_edge_and_eval(greedy_sol, la_ny, prev_apsp)
    bench_apsp, bench_score = evaluate_solution(test_graph)
    print(prev_score, test_score, bench_score)
    # print(test_apsp["New York, New York"])
    # print(bench_apsp["New York, New York"])
    for city in test_apsp:
        try:
            assert test_apsp["New York, New York"][city] == bench_apsp["New York, New York"][city]
        except:
            print("%.20f, %.20f" % (test_apsp["New York, New York"][city], bench_apsp["New York, New York"][city]))
    for city in bench_apsp:
        try:
            assert test_apsp["New York, New York"][city] == bench_apsp["New York, New York"][city]
        except:
            print("%.20f, %.20f" % (test_apsp["New York, New York"][city], bench_apsp["New York, New York"][city]))
    assert bench_score == test_score

def test_remove_edge_and_eval():
    cities = cities_from_file('data/us-cities-top-1k.csv')
    complete = complete_solution(cities)
    greedy_sol = greedy_buildup(cities, 50)
    prev_apsp, prev_score = evaluate_solution(greedy_sol)
    nj_ny = "Jersey City, New Jersey", "New York, New York", complete.edges["Jersey City, New Jersey", "New York, New York"]
    test_graph, test_apsp, test_score = remove_edge_and_eval(greedy_sol, nj_ny, prev_apsp)
    bench_apsp, bench_score = evaluate_solution(test_graph)
    print(prev_score, test_score, bench_score)
    # print(test_apsp["New York, New York"])
    # print(bench_apsp["New York, New York"])
    for city in test_apsp:
        try:
            assert test_apsp["New York, New York"][city] == bench_apsp["New York, New York"][city]
        except:
            print("%.20f, %.20f" % (test_apsp["New York, New York"][city], bench_apsp["New York, New York"][city]))
    for city in bench_apsp:
        try:
            assert test_apsp["New York, New York"][city] == bench_apsp["New York, New York"][city]
        except:
            print("%.20f, %.20f" % (test_apsp["New York, New York"][city], bench_apsp["New York, New York"][city]))
    assert bench_score == test_score


# read in args and stuff
if __name__ == "__main__":
    pass 