from util import *
import disjoint_set as ds
from collections import deque
from math import ceil

# continually add highest-weight edges until k miles is reached.
def greedy_buildup(empty, complete, k):
    sorted_edges = deque(sorted(complete.edges.data(), key=lambda x: score_calc(complete.nodes[x[0]]['population'], complete.nodes[x[1]]['population'], x[2]['dist']), reverse=True))
    curr_sol = empty.copy()
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
def knapsack_buildup(empty, complete, k):
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
    to_return = empty.copy()
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
def max_weight_spanning_tree_buildup(empty, complete, k, quit=False):
    curr_sol = empty.copy()
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
def min_dist_spanning_tree_buildup(empty, complete, k, quit=False):
    curr_sol = empty.copy()
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