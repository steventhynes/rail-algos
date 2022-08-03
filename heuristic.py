from copy import copy
from util import *
from random import random, choice
from naive import max_weight_spanning_tree_buildup, min_dist_spanning_tree_buildup
import time
from dataclasses import dataclass
from math import exp, ceil

@dataclass
class PossibleSolution:
    edge_dict: dict
    total_weight: float
    score: float
    heuristic_score: float
    graph: int

    # static vars
    complete_graph = None
    empty_graph = None
    weight_limit = None

    def __init__(self, edge_dict=None, total_weight=None, score=None, heuristic_score=None, graph=None):
        self.edge_dict = edge_dict
        self.total_weight = total_weight
        self.score = score
        self.heuristic_score = heuristic_score
        self.graph = graph
    
    def __lt__(self, other):
        self.total_weight < other.total_weight

    def __le__(self, other):
        self.total_weight <= other.total_weight

    def fill_from_graph(self):
        self.total_weight = self.graph.size(weight='dist')
        self.score = self._get_score()
        self.heuristic_score = self._get_heuristic_score()

    def graph_from_edges(self):
        self.graph = self.empty_graph.copy()
        for edge in self.edge_dict:
            if self.edge_dict[edge]:
                self.graph.add_edge(*edge, dist=self.complete_graph.edges[edge]['dist'])

    # A small deviation from the solution
    def tweak(self):
        print("tweaking")
        num_to_add = round(10*random()) if self.total_weight < self.weight_limit else 0
        num_to_remove = 0

        add_count = 0
        add_weight = 0
        remove_count = 0
        remove_weight = 0
        new_sol = PossibleSolution()
        new_sol.empty_graph = self.empty_graph
        new_sol.complete_graph = self.complete_graph
        new_sol.weight_limit = self.weight_limit
        new_sol.edge_dict = {edge:self.edge_dict[edge] for edge in self.edge_dict}

        def naive_add():
            nonlocal add_count
            nonlocal add_weight
            while add_count < num_to_add:
                selection = choice([edge for edge in self.edge_dict])
                if not new_sol.edge_dict[selection]:
                    new_sol.edge_dict[selection] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[selection]['dist']

        def add_2opt():
            while True:
                nonlocal add_count
                nonlocal add_weight
                edge1, edge2 = sample([edge for edge in self.edge_dict], 2)
                if random() < 0.5:
                    new1 = (edge1[0], edge2[0])
                    new2 = (edge1[1], edge2[1])
                else:
                    new1 = (edge1[0], edge2[1])
                    new2 = (edge1[1], edge2[0])
                if new1 not in self.edge_dict:
                    new1 = new1[1], new1[0]
                if new1 not in self.complete_graph.edges:
                    continue
                if new2 not in self.edge_dict:
                    new2 = new2[1], new2[0]
                if new2 not in self.complete_graph.edges:
                    continue
                print([edge1, edge2])
                print([new1, new2])
                if not self.edge_dict[new1]:
                    self.edge_dict[new1] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[new1]['dist']
                if not self.edge_dict[new2]:
                    self.edge_dict[new2] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[new2]['dist']
                self.edge_dict[edge1] = False
                self.edge_dict[edge2] = False
                break
        
        def add_shortest_path(connected_only=False):
            nonlocal add_count
            nonlocal add_weight
            node1, node2 = sample([node for node in self.graph.nodes if (not connected_only or self.graph.edges(node))], 2)
            try:
                # shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight='dist')
                shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight=lambda end1, end2, dict: 1 / score_calc(self.complete_graph.nodes[end1]['population'], self.complete_graph.nodes[end2]['population'], dict['dist']))
                # shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight=lambda end1, end2, dict: 1 / score_calc(self.complete_graph.nodes[end1]['population'], self.complete_graph.nodes[end2]['population'], 1))
            except nx.NetworkXNoPath:
                return
            for node_idx in range(len(shortest_path) - 1):
                new_edge = tuple(shortest_path[node_idx:node_idx+2])
                if new_edge not in self.edge_dict:
                    new_edge = new_edge[1], new_edge[0]
                if not self.edge_dict[new_edge]:
                    new_sol.edge_dict[new_edge] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[new_edge]['dist']

        def replace_shortest_path():
            nonlocal add_count
            nonlocal add_weight
            nonlocal remove_count
            nonlocal remove_weight
            node1 = choice([node for node in self.graph.nodes if self.graph.edges(node)])
            shortest_paths_from_node1 = nx.shortest_path(self.graph, source=node1, weight='dist')
            # print(shortest_paths_from_node1)
            node2 = choice(list(shortest_paths_from_node1.keys()))
            try:
                old_shortest_path = shortest_paths_from_node1[node2]
                new_shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight=lambda end1, end2, dict: 1 / score_calc(self.complete_graph.nodes[end1]['population'], self.complete_graph.nodes[end2]['population'], dict['dist']**2))
                # new_shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight='dist')
            except nx.NetworkXNoPath:
                print("no path found")
                return
            for node_idx in range(len(old_shortest_path) - 1):
                new_edge = tuple(old_shortest_path[node_idx:node_idx+2])
                if new_edge not in self.edge_dict:
                    new_edge = new_edge[1], new_edge[0]
                if self.edge_dict[new_edge]:
                    new_sol.edge_dict[new_edge] = False
                    remove_count += 1
                    remove_weight += self.complete_graph.edges[new_edge]['dist']
            for node_idx in range(len(new_shortest_path) - 1):
                new_edge = tuple(new_shortest_path[node_idx:node_idx+2])
                if new_edge not in self.edge_dict:
                    new_edge = new_edge[1], new_edge[0]
                if not self.edge_dict[new_edge]:
                    new_sol.edge_dict[new_edge] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[new_edge]['dist']


        add_shortest_path(connected_only=False)
        while self.total_weight + add_weight - remove_weight > self.weight_limit:
            # replace_shortest_path()
            selection = choice([edge for edge in self.graph.edges])
            if new_sol.edge_dict[selection]:
                new_sol.edge_dict[selection] = False
                remove_count += 1
                remove_weight += self.complete_graph.edges[selection]['dist']
        # print(f"tweak: {add_count=} of {(len(self.complete_graph.edges)-len(self.graph.edges))=}, {remove_count=} of {len(self.graph.edges)}")
        new_sol.graph_from_edges()
        new_sol.fill_from_graph()
        print(f"TWEAK: {new_sol.total_weight=}, {new_sol.score=}, {new_sol.heuristic_score=}")
        return new_sol

    # A large deviation from the solution, meant to break out of a local optimum
    def perturb(self):
        print("perturbing")
        # num_to_add = 10 * (1 - self.total_weight / self.weight_limit)
        num_to_add = random() * ceil(self.weight_limit / 100)
        # num_to_remove = random() * 1000

        add_count = 0
        add_weight = 0
        remove_count = 0
        remove_weight = 0
        new_sol = PossibleSolution()
        new_sol.edge_dict = {edge:self.edge_dict[edge] for edge in self.edge_dict}
        def naive_add():
            nonlocal add_count
            nonlocal add_weight
            while add_count < num_to_add:
                selection = choice([edge for edge in self.edge_dict])
                if not new_sol.edge_dict[selection]:
                    new_sol.edge_dict[selection] = True
                    add_count += 1
                    add_weight += self.complete_graph.edges[selection]['dist']
        def add_shortest_paths():
            nonlocal add_count
            nonlocal add_weight
            source = choice([node for node in self.graph.nodes])
            try:
                # shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight='dist')
                shortest_paths = nx.shortest_path(self.complete_graph, source=source, weight=lambda end1, end2, dict: 1 / score_calc(self.complete_graph.nodes[end1]['population'], self.complete_graph.nodes[end2]['population'], dict['dist']))
            except nx.NetworkXNoPath:
                print("no path found")
                return
            for target in shortest_paths:
                print(target)
                for node_idx in range(len(shortest_paths[target]) - 1):
                    new_edge = tuple(shortest_paths[target][node_idx:node_idx+2])
                    if new_edge not in new_sol.edge_dict:
                        new_edge = new_edge[1], new_edge[0]
                    if not new_sol.edge_dict[new_edge]:
                        new_sol.edge_dict[new_edge] = True
                        add_count += 1
                        add_weight += self.complete_graph.edges[new_edge]['dist']
                print("complete")

        naive_add()
        while self.total_weight + add_weight - remove_weight > self.weight_limit:
            selection = choice([edge for edge in self.graph.edges if new_sol.edge_dict[edge]])
            if new_sol.edge_dict[selection]:
                new_sol.edge_dict[selection] = False
                remove_count += 1
                remove_weight += self.complete_graph.edges[selection]['dist']
        print(self.total_weight + add_weight - remove_weight)
        print(f"perturb: {add_count=} of {(len(self.complete_graph.edges)-len(self.graph.edges))=}, {remove_count=} of {len(self.graph.edges)}")
        new_sol.graph_from_edges()
        new_sol.fill_from_graph()
        return new_sol

    # Decide whether to move the home base to the new local optimum
    def new_home_base(self, new_home):
        return self if self.heuristic_score > new_home.heuristic_score else new_home

    def _get_score(self):
        try:
            # print("try")
            # return approx_evaluate_solution(self.graph, self.sorted_nodes[:20], 0)
            return approx_evaluate_solution(self.graph, PossibleSolution.sorted_nodes[:500], 0)
            # return approx_evaluate_solution(self.graph, [], len(self.graph))
        except AttributeError:
            # print("except")
            PossibleSolution.sorted_nodes = sorted(self.complete_graph.nodes, key=lambda x: self.complete_graph.nodes[x]['population'])
            # return approx_evaluate_solution(self.graph, self.sorted_nodes[:20], 0)
            return approx_evaluate_solution(self.graph, PossibleSolution.sorted_nodes[:500], 0)
            # return approx_evaluate_solution(self.graph, [], len(self.graph))

    def _get_heuristic_score(self):
        return self.score - 1e15 * max(0, (self.total_weight - self.weight_limit))



def generate_solution(empty_graph, complete_graph, weight_limit):
    # edge_prob = 400/len(complete_graph.edges)

    new_sol = PossibleSolution()
    PossibleSolution.empty_graph = empty_graph
    PossibleSolution.complete_graph = complete_graph
    PossibleSolution.weight_limit = weight_limit
    # new_sol.graph = greedy_buildup(cities, weight_limit)
    # new_sol.graph = max_weight_spanning_tree_buildup(cities, weight_limit)
    new_sol.graph = min_dist_spanning_tree_buildup(empty_graph, complete_graph, weight_limit)
    new_sol.edge_dict = {edge:(edge in new_sol.graph.edges) for edge in complete_graph.edges}
    new_sol.fill_from_graph()

    return new_sol
    

# Local search with a perturbation when a local optimum is found. Perturbation is made to be different
# from a previous local minima.
def iterated_local_search(empty, complete, k, global_timeout=600, local_timeout=30):
    curr_sol = generate_solution(empty, complete, cities, k)
    curr_home = curr_sol
    curr_best = curr_sol
    global_start_time = time.time()
    while time.time() - global_start_time < global_timeout:
        local_start_time = time.time()
        while time.time() - local_start_time < local_timeout:
            print(f"BEST: {curr_best.total_weight=}, {curr_best.score=}, {curr_best.heuristic_score=}")
            print(f"HOME: {curr_home.total_weight=}, {curr_home.score=}, {curr_home.heuristic_score=}")
            print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.heuristic_score=}")
            new_sol = curr_sol.tweak()
            if new_sol.heuristic_score > curr_sol.heuristic_score:
                curr_sol = new_sol
        if curr_sol.heuristic_score > curr_best.heuristic_score:
            curr_best = curr_sol
        curr_home = curr_home.new_home_base(curr_sol)
        curr_sol = curr_home.perturb()
    return curr_best.graph
        
        
# Local search with a steadily decreasing chance to move to an inferior solution
def simulated_annealing(empty, complete, k, timeout=600, temp_mult=30):
    def temp(elapsed_time):
        return temp_mult * (1 - (elapsed_time / timeout))
    def switch_prob(temp, better_qual, worse_qual):
        try:
            print(exp((worse_qual - better_qual) / temp))
            return exp((worse_qual - better_qual) / temp)
        except OverflowError:
            return 0

    curr_sol = generate_solution(empty, complete, cities, k)
    curr_best = curr_sol
    start_time = time.time()
    while time.time() - start_time < timeout:
        print(f"BEST: {curr_best.total_weight=}, {curr_best.score=}, {curr_best.heuristic_score=}")
        print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.heuristic_score=}")
        new_sol = curr_sol.tweak()
        if new_sol.heuristic_score > curr_sol.heuristic_score or random() < switch_prob(temp(time.time() - start_time), curr_sol.heuristic_score, new_sol.heuristic_score):
            curr_sol = new_sol
        if curr_sol.heuristic_score > curr_best.heuristic_score:
            curr_best = curr_sol
    return curr_best.graph

def ant_colony_optimization(empty, complete, k, global_timeout=600, local_timeout=30, evaporation=0.9, popsize=10, initial_pheromone=1, sigma=1, epsilon=1):
    edges = complete.edges
    pheromones = {edge:initial_pheromone for edge in edges}
    
    def generate_ant_trail():
        desirabilities = [(edge, (pheromones[edge] ** sigma * complete.edges[edge]['dist'] ** -epsilon)) for edge in edges]
        cumulative_desirabilities = desirabilities
        for idx in range(1, len(desirabilities)):
            cumulative_desirabilities[idx] = (desirabilities[idx][0], cumulative_desirabilities[idx-1][1] + desirabilities[idx][1])
        new_sol = PossibleSolution()
        new_sol.empty_graph = empty
        new_sol.graph = empty.copy()
        new_sol.complete_graph = complete
        new_sol.weight_limit = k
        while new_sol.graph.size('dist') < k:
            random_num = random() * cumulative_desirabilities[-1][1]
            # binary search
            lower_limit = 0
            upper_limit = len(desirabilities)
            while True:
                selection_idx = (lower_limit + upper_limit) // 2
                if cumulative_desirabilities[selection_idx][1] > random_num:
                    if selection_idx == 0 or random_num > cumulative_desirabilities[selection_idx-1][1]:
                        edge = cumulative_desirabilities[selection_idx][0]
                        new_sol.graph.add_edge(*edge, dist=edges[edge]['dist'])
                        break
                    else:
                        upper_limit = selection_idx
                else:
                    lower_limit = selection_idx
        if new_sol.graph.size('dist') > k:
            new_sol.graph.remove_edge(*edge)
        new_sol.edge_dict = {edge:(edge in new_sol.graph.edges) for edge in complete.edges}
        new_sol.fill_from_graph()
        # print(new_sol.total_weight)
        return new_sol

    best = None
    start_time = time.time()
    while time.time() - start_time < global_timeout:
        trails = []
        while len(trails) < popsize:
            curr_sol = generate_ant_trail()
            local_start_time = time.time()
            while time.time() - local_start_time < local_timeout:
                new_sol = curr_sol.tweak()
                if new_sol.heuristic_score > curr_sol.heuristic_score:
                    curr_sol = new_sol
            if best is None or curr_sol.heuristic_score > best.heuristic_score:
                best = curr_sol
            print(f"BEST: {best.total_weight=}, {best.score=}, {best.heuristic_score=}")
            print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}, {curr_sol.heuristic_score=}")
            trails.append(curr_sol)
        for edge in pheromones:
            pheromones[edge] *= (1 - evaporation)
        for trail in trails:
            for edge in trail.graph.edges:
                pheromones[edge] += trail.heuristic_score
    return best.graph

            
def evolutionary_algorithm(empty, complete, k, timeout=1200, num_parents=10, popsize=20, parents_persist=True, genetic=False):
    edges = complete.edges
    edges_list = list(edges)
    curr_pop = []

    def new_solution():
        new_sol = PossibleSolution()
        new_sol.empty_graph = empty
        new_sol.graph = empty.copy()
        new_sol.complete_graph = complete
        new_sol.weight_limit = k
        while new_sol.graph.size('dist') < k:
            edge = choice(edges_list)
            # print(edge)
            new_sol.graph.add_edge(*edge, dist=edges[edge]['dist'])
        if new_sol.graph.size('dist') > k:
            new_sol.graph.remove_edge(*edge)
        new_sol.edge_dict = {edge:(edge in new_sol.graph.edges) for edge in complete.edges}
        new_sol.fill_from_graph()
        return new_sol

    def genetic_crossover(parent1, parent2, swap_prob=0.1):
        child1 = copy(parent1)
        child2 = copy(parent2)
        for edge in edges:
            if random() < swap_prob:
                child1.edge_dict[edge], child2.edge_dict[edge] = parent2.edge_dict[edge], parent1.edge_dict[edge]
        child1.graph_from_edges()
        child2.graph_from_edges()
        child1.fill_from_graph()
        child2.fill_from_graph()
        return child1, child2

    while len(curr_pop) < popsize:
        curr_pop.append(new_solution())
    best = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        for sol in curr_pop:
            if best is None or best.heuristic_score < sol.heuristic_score:
                best = sol
        sorted_pop = sorted(curr_pop, key=lambda x: x.heuristic_score, reverse=True)
        parents = sorted_pop[:num_parents]
        print(f"BEST: {best.total_weight=}, {best.score=}, {best.heuristic_score=}")
        curr_pop = parents[:] if parents_persist else []
        if genetic: # using genetic algorithm with crossover
            while len(curr_pop) < popsize:
                curr_parents = sample(parents, 2)
                children = genetic_crossover(*curr_parents)
                children = [child.tweak() for child in children]
                curr_pop.extend(children)
        else: # using regular evolutionary algorithm
            for parent in parents:
                for child_num in range(popsize // num_parents):
                    curr_pop.append(parent.tweak())
        print(f'{len(curr_pop)=}')
    return best.graph