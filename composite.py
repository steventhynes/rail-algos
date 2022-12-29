from random import shuffle

from numpy import disp
from util import cities_from_file, display_solution

def naive_merge(empty, complete, k, solutions):
    edge_counts = {}
    for sol in solutions:
        for edge in sol.edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    edges = list(edge_counts.keys())
    shuffle(edges)
    sorted_edges = sorted(edges, key=lambda x: edge_counts[x], reverse=True)
    new_sol = empty.copy()
    for edge in sorted_edges:
        if new_sol.size('dist') + complete.edges[edge]['dist'] < k:
            new_sol.add_edge(*edge, dist=complete.edges[edge]['dist'])
    return new_sol

def test():
    from naive import greedy_buildup, min_dist_spanning_tree_buildup, max_weight_spanning_tree_buildup
    from heuristic import ant_colony_optimization, evolutionary_algorithm
    empty, complete = cities_from_file('data/us-cities-top-1k.csv')
    sol1 = greedy_buildup(empty, complete, 1000)
    display_solution(sol1)
    sol2 = min_dist_spanning_tree_buildup(empty, complete, 1000)
    display_solution(sol2)
    sol3 = max_weight_spanning_tree_buildup(empty, complete, 1000)
    display_solution(sol3)
    sol4 = ant_colony_optimization(empty, complete, 1000)
    display_solution(sol4)
    sol5 = evolutionary_algorithm(empty, complete, 1000)
    display_solution(sol5)
    sols = [sol1, sol2, sol3, sol4, sol5]
    # sols = [ant_colony_optimization(cities, 1000) for i in range(5)]
    final = naive_merge(empty, complete, 1000, sols)
    display_solution(final)
    return final

        

    
    