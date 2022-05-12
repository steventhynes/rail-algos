from util import *
from collections import deque
import heapq as hq
from dataclasses import dataclass

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