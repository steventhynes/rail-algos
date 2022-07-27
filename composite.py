from random import shuffle
import time
from util import cities_from_file, display_solution, empty_solution, complete_solution

def naive_merge(cities, k, solutions):
    complete = complete_solution(cities)
    edge_counts = {}
    for sol in solutions:
        for edge in sol.edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    edges = list(edge_counts.keys())
    shuffle(edges)
    sorted_edges = sorted(edges, key=lambda x: edge_counts[x], reverse=True)
    new_sol = empty_solution(cities)
    for edge in sorted_edges:
        if new_sol.size('dist') + complete.edges[edge]['dist'] < k:
            new_sol.add_edge(*edge, dist=complete.edges[edge]['dist'])
    return new_sol

def test():
    from naive import greedy_buildup, min_dist_spanning_tree_buildup, max_weight_spanning_tree_buildup
    from heuristic import ant_colony_optimization
    cities = cities_from_file('data/us-cities-top-1k.csv')
    # sol1 = greedy_buildup(cities, 1000)
    # sol2 = max_weight_spanning_tree_buildup(cities, 1000)
    # sol3 = ant_colony_optimization(cities, 1000)
    # final = naive_merge(cities, 1000, [sol1, sol2, sol3])
    sols = [ant_colony_optimization(cities, 1000) for i in range(6)]
    final = naive_merge(cities, 1000, sols)
    display_solution(final)
    return final

        

    
    