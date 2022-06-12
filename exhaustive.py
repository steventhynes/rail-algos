from copy import copy
from random import randint
from util import *
from collections import deque
import heapq as hq
from dataclasses import dataclass

@dataclass
class PartialSolution:
    edge_array: list # used only in 
    edge_set: set # used only in BnB
    total_weight: float
    shortest_path_lengths: dict
    score: float
    depth: int
    graph: int

    def __init__(self, edge_array=None, edge_set=None, total_weight=None, shortest_path_lengths=None, score=None, depth=None, graph=None):
        self.edge_array = edge_array
        self.edge_set = edge_set
        self.total_weight =  total_weight
        self.shortest_path_lengths = shortest_path_lengths
        self.score = score
        self.depth = depth
        self.graph = graph
    
    def __lt__(self, other):
        self.total_weight < other.total_weight

    def __le__(self, other):
        self.total_weight <= other.total_weight

def max_potential(sol, max_weight, complete, *args, **kwargs):
        try:
            edge = max_potential.best_edge
        except AttributeError:
            max_potential.best_edge = max(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']))
            edge = max_potential.best_edge
        num_edges = ((max_weight - sol.total_weight) / edge[2]['dist'])
        max_pot = sol.score +  num_edges * score_calc(complete.nodes[edge[0]]['population'], complete.nodes[edge[1]]['population'], edge[2]['dist'])
        return max_pot
        

def heuristic(sol, max_weight, complete, score_threshold, *args, **kwargs):
    # return .25*(sol.depth / len(complete_edges)) + .75*(sol.score / score_threshold) - (sol.total_weight / max_weight)
    # return (sol.score / score_threshold)**2 - (sol.total_weight / max_weight)**2
    # return ((sol.score / score_threshold) - (sol.total_weight / max_weight)) * len([node for node in complete.nodes if complete.edges(node)])
    # connected_components = nx.connected_components(sol.graph)
    # return sum((len(comp)**2 for comp in connected_components))+sol.score-sol.total_weight
    # return sol.score * (max_weight - sol.total_weight)**2
    # return max_potential(sol, max_weight, complete)
    # try:
    #     edge = heuristic.average_edge
    # except AttributeError:
    #     heuristic.average_edge = sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']))[len(complete.edges)//2]
    #     edge = heuristic.average_edge
    # num_edges = ((max_weight - sol.total_weight) / edge[2]['dist'])
    # exp_pot = sol.score +  num_edges * score_calc(complete.nodes[edge[0]]['population'], complete.nodes[edge[1]]['population'], edge[2]['dist'])
    # return exp_pot
    try:
        return approx_evaluate_solution(sol.graph, heuristic.sorted_edges[:20], 180)
    except AttributeError:
        heuristic.sorted_edges = sorted(complete.nodes, key=lambda x: complete.nodes[x]['population'])
        return approx_evaluate_solution(sol.graph, heuristic.sorted_edges[:20], 180) / sol.total_weight

def branch_and_bound_iterative_deepening(cities, k, score_threshold, iter_depth=5, heap_max=2000, heap_after_cull=500):
    complete = complete_solution(cities)
    complete_edges = sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True)
    partial_solutions_heap = []
    initial_sol = PartialSolution([], None, 0, *evaluate_solution(empty_solution(cities)), 0, empty_solution(cities))
    hq.heappush(partial_solutions_heap, (0, initial_sol)) #priority queue
    curr_best = initial_sol
    heap_graphs_count = 0
    try:
        while partial_solutions_heap:
            if len(partial_solutions_heap) >= heap_max:
                print("trimming heap")
                new_heap = []
                while len(new_heap) < heap_after_cull:
                    hq.heappush(new_heap, hq.heappop(partial_solutions_heap))
                partial_solutions_heap = new_heap
            # Pop partial solution off the priority queue and initialize the next run
            print("partial sol popped off.")
            partial_sol = hq.heappop(partial_solutions_heap)[1]
            base_depth = partial_sol.depth
            stack = deque()
            # Rebuild graph if it's not there
            stack.append(partial_sol)
            while stack:
                curr_sol = stack.pop()
                print("popped off from stack")
                # print(f"{curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.depth=}, {curr_sol.edge_array}")
                print(f"{curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.depth=}")

                if curr_sol.total_weight > k: # This solution is prohibitive -- the cost is too high.
                    continue # don't add expanded solutions to the stack

                if max_potential(curr_sol, k, complete) < curr_best.score: # The upper bound on this solution's performance is less than what we already have
                    print("solution discarded - upper bound less than current best")
                    continue # don't add expanded solutions to the stack
                
                if curr_sol.score > curr_best.score:
                    curr_best = curr_sol

                if score_threshold is not None and curr_sol.score >= score_threshold:
                    return curr_sol.graph

                elif curr_sol.depth == base_depth + iter_depth: # This is a partial solution; we've reached max depth for this iteration
                    # score = curr_sol.score / curr_sol.total_weight if curr_sol.total_weight else 0
                    print("adding partial sol to heap")
                    heuristic_score = heuristic(curr_sol, k, complete, score_threshold)
                    hq.heappush(partial_solutions_heap, (-heuristic_score, curr_sol)) # score is negative because hq only supports minheap

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
def branch_and_bound_breadth(cities, k, score_threshold, expansion=1000, heap_max=2000, heap_after_cull=500):
    
    complete = complete_solution(cities)
    complete_edges = sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True)
    # complete_edges = list(complete.edges.data())
    partial_solutions_heap = []
    initial_sol = PartialSolution(None, set(), 0, *evaluate_solution(empty_solution(cities)), 0, empty_solution(cities))
    hq.heappush(partial_solutions_heap, (0, initial_sol)) #priority queue
    curr_best = initial_sol
    # heap_graphs_count = 0
    try:
        while partial_solutions_heap:
            if len(partial_solutions_heap) >= heap_max:
                print("trimming heap")
                new_heap = []
                while len(new_heap) < heap_after_cull:
                    hq.heappush(new_heap, hq.heappop(partial_solutions_heap))
                partial_solutions_heap = new_heap
            # Pop partial solution off the priority queue and initialize the next run
            print("partial sol popped off.")
            curr_sol = hq.heappop(partial_solutions_heap)[1]
            # Rebuild graph if it's not there
            # if curr_sol.graph is None:
            #     curr_sol.graph = empty_solution(cities)
            #     edges_to_add = [complete_edges[idx][:2] for idx in range(len(partial_sol.edge_array)) if partial_sol.edge_array[idx]]
            #     curr_sol.graph.add_edges_from(edges_to_add)
            # # If graph is there, decrease the count
            # else:
            #     heap_graphs_count -= 1

            print(f"{curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.depth=}")

            if curr_sol.total_weight > k: # This solution is prohibitive -- the cost is too high.
                print("solution discarded - cost too high")
                continue # don't add expanded solutions to the stack

            if max_potential(curr_sol, k, complete) < curr_best.score: # The upper bound on this solution's performance is less than what we already have
                print("solution discarded - upper bound less than current best")
                continue # don't add expanded solutions to the stack
            
            if curr_sol.score > curr_best.score:
                curr_best = curr_sol

            if score_threshold is not None and curr_sol.score >= score_threshold:
                return curr_sol.graph

            # elif curr_sol.depth == base_depth + iter_depth: # This is a partial solution; we've reached max depth for this iteration
            #     # score = curr_sol.score / curr_sol.total_weight if curr_sol.total_weight else 0
            #     print("adding partial sol to heap")
            #     if heap_graphs_count >= max_heap_graphs:
            #         curr_sol.graph = None
            #     else:
            #         heap_graphs_count += 1
            #     score = (.25*(curr_sol.depth / len(complete_edges)) + .75*(curr_sol.score / score_threshold)) - (curr_sol.total_weight / k)
            #     hq.heappush(partial_solutions_heap, (-score, curr_sol)) # score is negative because hq only supports minheap

            else: # None of the above are satisfied; create new solutions and add them to the stack
                # no_edge_sol = PartialSolution()
                # no_edge_sol.edge_array = curr_sol.edge_array + [False] * (base_depth + iter_depth - curr_sol.depth + 1)
                # no_edge_sol.total_weight = curr_sol.total_weight
                # no_edge_sol.shortest_path_lengths = curr_sol.shortest_path_lengths
                # no_edge_sol.score = curr_sol.score
                # no_edge_sol.depth = curr_sol.depth + 1
                # no_edge_sol.graph = curr_sol.graph
                # stack.append(no_edge_sol)

                print("adding new solutions to heap")

                edges_generated = 0
                while edges_generated < expansion:
                    edge_add_idx = edges_generated if edges_generated < expansion/2 else randint(0, len(complete_edges) - 1) 
                    if edge_add_idx not in curr_sol.edge_set:
                        new_edge_sol = PartialSolution()
                        new_edge_sol.edge_set = copy(curr_sol.edge_set)
                        new_edge_sol.edge_set.add(edge_add_idx)
                        new_edge_sol.graph, new_edge_sol.shortest_path_lengths, new_edge_sol.score = add_edge_and_eval(curr_sol.graph.copy(), complete_edges[edge_add_idx], curr_sol.shortest_path_lengths)
                        new_edge_sol.total_weight = new_edge_sol.graph.size(weight='dist')
                        new_edge_sol.depth = len(new_edge_sol.edge_set)
                        heuristic_score = heuristic(new_edge_sol, k, complete, score_threshold)
                        hq.heappush(partial_solutions_heap, (-heuristic_score, new_edge_sol))
                    edges_generated += 1
                print(len(partial_solutions_heap))
                
    except KeyboardInterrupt:
        print(curr_best.edge_set)
        return_graph = empty_solution(cities)
        edges_to_add = [complete_edges[idx][:2] for idx in curr_best.edge_set]
        return_graph.add_edges_from(edges_to_add)
        return return_graph