from copy import copy
from util import *
from random import random, choice
import time
from naive import greedy_buildup, max_weight_spanning_tree_buildup, min_dist_spanning_tree_buildup
from dataclasses import dataclass
from math import exp, ceil

class PossibleSolution:
    # edge_dict: dict
    # total_weight: float
    # score: float
    # graph: int

    # static vars
    complete_graph = None
    empty_graph = None
    weight_limit = None

    def __init__(self, edge_dict=None, total_weight=None, score=None, graph=None): # regular constructor
        self.edge_dict = edge_dict
        self.total_weight = total_weight
        self.score = score
        self.graph = graph
    
    @classmethod
    def init_statics(cls, empty, complete, k):
        cls.empty_graph = empty.copy()
        cls.complete_graph = complete
        cls.weight_limit = k

    @classmethod
    def copy_from(cls, other): # copy "constructor"
        return cls(other.edge_dict, other.total_weight, other.score, other.graph.copy())
    
    def __lt__(self, other):
        self.total_weight < other.total_weight

    def __le__(self, other):
        self.total_weight <= other.total_weight

    def fill_from_graph(self):
        self.total_weight = self.graph.size(weight='dist')
        self.score = self._get_score()

    def graph_from_edges(self):
        self.graph = self.empty_graph.copy()
        # print(self.graph.size(weight='dist'))
        for edge in self.edge_dict:
            if self.edge_dict[edge] == True:
                self.graph.add_edge(*edge, dist=self.complete_graph.edges[edge]['dist'])

    # A small deviation from the solution
    def tweak(self):
        print("tweaking")
        add_count = 0
        add_weight = 0
        remove_count = 0
        remove_weight = 0
        # new_sol = PossibleSolution()
        # new_sol.empty_graph = self.empty_graph
        # new_sol.complete_graph = self.complete_graph
        # new_sol.weight_limit = self.weight_limit
        # new_sol.edge_dict = {edge:self.edge_dict[edge] for edge in self.edge_dict}
        
        def add_shortest_path():
            nonlocal add_count
            nonlocal add_weight
            nonlocal remove_count
            nonlocal remove_weight
            edge = choice([edge for edge in self.complete_graph.edges])
            try:
                if edge in self.graph.edges:
                    self.graph.remove_edge(*edge)
                    remove_count += 1
                    remove_weight += self.complete_graph.edges[edge]['dist']
                # shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight='dist')
                # edge_dist = self.complete_graph.edges[edge]['dist']
                # self.complete_graph.remove_edge(*edge)
                shortest_path = nx.shortest_path(self.complete_graph, *edge, weight=lambda end1, end2, dict: dict['dist']**2)
                # self.complete_graph.add_edge(*edge, dist=edge_dist)
                # shortest_path = nx.shortest_path(self.complete_graph, node1, node2, weight=lambda end1, end2, dict: 1 / score_calc(self.complete_graph.nodes[end1]['population'], self.complete_graph.nodes[end2]['population'], 1))
            except nx.NetworkXNoPath:
                return
            for node_idx in range(len(shortest_path) - 1):
                new_edge = tuple(shortest_path[node_idx:node_idx+2])
                if new_edge not in self.edge_dict:
                    new_edge = new_edge[1], new_edge[0]
                if not self.edge_dict[new_edge]:
                    self.edge_dict[new_edge] = True
                    new_dist = self.complete_graph.edges[new_edge]['dist']
                    self.graph.add_edge(*new_edge, dist=new_dist)
                    add_count += 1
                    add_weight += new_dist

        def optimize_subgraph(circle_radius=10):
            nonlocal add_count
            nonlocal add_weight
            nonlocal remove_count
            nonlocal remove_weight
            center_node = choice(list(self.complete_graph.nodes))
            nodes_in_circle = [node for node in self.complete_graph.nodes if (center_node, node) in self.complete_graph.edges and self.complete_graph.edges[(center_node, node)]['dist'] <= circle_radius or (node, center_node) in self.complete_graph.edges and self.complete_graph.edges[(node, center_node)]['dist'] <= circle_radius]
            nodes_in_circle.append(center_node)
            # complete_subgraph = self.complete_graph.subgraph(nodes_in_circle)
            # print(len(nodes_in_circle))
            complete_subgraph = nx.induced_subgraph(self.complete_graph, nodes_in_circle)
            # curr_subgraph = self.graph.subgraph(nodes_in_circle)
            curr_subgraph = nx.induced_subgraph(self.graph, nodes_in_circle)
            # print(f"{self.total_weight=}")
            # print(f"{self.graph.size(weight='dist')=}")
            weight = curr_subgraph.size(weight='dist')
            empty_subgraph = nx.create_empty_copy(curr_subgraph)
            optimized_subgraph = min_dist_spanning_tree_buildup(empty_subgraph, complete_subgraph, float('inf'), quit=True)
            for edge in curr_subgraph.edges:
                if edge not in self.edge_dict:
                    edge = edge[1], edge[0]
                if self.edge_dict[edge] == True:
                    self.edge_dict[edge] = False
                    self.graph.remove_edge(*edge)
                    remove_count += 1
                    remove_weight += self.complete_graph.edges[edge]['dist']
            for edge in optimized_subgraph.edges:
                if edge not in self.edge_dict:
                    edge = edge[1], edge[0]
                if self.edge_dict[edge] == False:
                    self.edge_dict[edge] = True
                    new_dist = self.complete_graph.edges[edge]['dist']
                    self.graph.add_edge(*edge, dist=new_dist)
                    add_count += 1
                    add_weight += new_dist
            # print(f"{sum((self.complete_graph.edges[edge]['dist'] for edge in new_sol.edge_dict if new_sol.edge_dict[edge] == True))=}")

        def random_tweak():
            rand = random() * 24
            if 3 < rand < 15:
                print('optimize subgraph with radius ' + str(rand))
                optimize_subgraph(rand)
            else:
                print('add shortest path')
                add_shortest_path()

        # add_shortest_path(connected_only=False)
        random_tweak()
        # print(f"{self.total_weight + add_weight - remove_weight=}")
        while self.total_weight + add_weight - remove_weight > self.weight_limit:
            # replace_shortest_path()
            selection = choice([edge for edge in self.graph.edges])
            if self.edge_dict[selection]:
                self.edge_dict[selection] = False
                self.graph.remove_edge(*selection)
                remove_count += 1
                remove_weight += self.complete_graph.edges[selection]['dist']
        # print(f"tweak: {add_count=} of {(len(self.complete_graph.edges)-len(self.graph.edges))=}, {remove_count=} of {len(self.graph.edges)}")
        print("time before fill: %f" % time.time())
        self.fill_from_graph()
        print("time after fill: %f" % time.time())
        # print(f"{new_sol.graph.size(weight='dist')=}")
        print(f"TWEAK: {self.total_weight=}, {self.score=}")

    # A large deviation from the solution, meant to break out of a local optimum
    def perturb(self):
        print("perturbing")
        # num_to_add = 10 * (1 - self.total_weight / self.weight_limit)
        num_to_add = 0.01 * len(self.graph.edges)
        # num_to_remove = random() * 1000

        add_count = 0
        add_weight = 0
        remove_count = 0
        remove_weight = 0
        def naive_add():
            nonlocal add_count
            nonlocal add_weight
            while add_count < num_to_add:
                selection = choice([edge for edge in self.edge_dict])
                if not self.edge_dict[selection]:
                    self.edge_dict[selection] = True
                    new_dist = self.complete_graph.edges[selection]['dist']
                    self.graph.add_edge(*selection, dist=new_dist)
                    add_count += 1
                    add_weight += new_dist

        naive_add()
        while self.total_weight + add_weight - remove_weight > self.weight_limit:
            # replace_shortest_path()
            selection = choice([edge for edge in self.graph.edges])
            if self.edge_dict[selection]:
                self.edge_dict[selection] = False
                self.graph.remove_edge(*selection)
                remove_count += 1
                remove_weight += self.complete_graph.edges[selection]['dist']
        # print(f"tweak: {add_count=} of {(len(self.complete_graph.edges)-len(self.graph.edges))=}, {remove_count=} of {len(self.graph.edges)}")
        self.fill_from_graph()
        # print(f"{new_sol.graph.size(weight='dist')=}")

    # Decide whether to move the home base to the new local optimum
    def new_home_base(self, new_home):
        return self if self.score > new_home.score else new_home

    def _get_score(self):
        while True:
            try:
                # print("try")
                # return approx_evaluate_solution(self.graph, self.sorted_nodes[:20], 0)
                # score = approx_evaluate_solution(self.graph, PossibleSolution.sorted_nodes[:len(self.graph.nodes)//30] + PossibleSolution.sorted_nodes[-len(self.graph.nodes)//30:], 0) - 1e15 * max(0, (self.total_weight - self.weight_limit))
                score = approx_evaluate_solution(self.graph, PossibleSolution.sorted_nodes[:len(self.graph.nodes)//50] + PossibleSolution.sorted_nodes[len(self.graph.nodes)//50::50], 0) - 1e15 * max(0, (self.total_weight - self.weight_limit))
                return score
            except AttributeError:
                # print("except")
                PossibleSolution.sorted_nodes = sorted(self.complete_graph.nodes, key=lambda x: self.complete_graph.nodes[x]['population'])

def generate_solution(empty_graph, complete_graph, weight_limit):
    # edge_prob = 400/len(complete_graph.edges)

    new_sol = PossibleSolution()
    # new_sol.graph = greedy_buildup(cities, weight_limit)
    # new_sol.graph = max_weight_spanning_tree_buildup(cities, weight_limit)
    new_sol.graph = min_dist_spanning_tree_buildup(empty_graph, complete_graph, weight_limit)
    # new_sol.graph = empty_graph.copy()
    new_sol.edge_dict = {edge:(edge in new_sol.graph.edges) for edge in complete_graph.edges}
    new_sol.fill_from_graph()

    return new_sol
    

# Local search with a perturbation when a local optimum is found. Perturbation is made to be different
# from a previous local minima.
def iterated_local_search(empty, complete, k, global_timeout=600, local_timeout=30):
    PossibleSolution.init_statics(empty, complete, k)
    curr_sol = generate_solution(empty, complete, k)
    curr_home = PossibleSolution.copy_from(curr_sol)
    curr_best = PossibleSolution.copy_from(curr_sol)
    global_start_time = time.time()
    while time.time() - global_start_time < global_timeout:
        local_start_time = time.time()
        while time.time() - local_start_time < local_timeout:
            print(f"BEST: {curr_best.total_weight=}, {curr_best.score=}")
            print(f"HOME: {curr_home.total_weight=}, {curr_home.score=}")
            print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}")
            print('time before copy: %f' % time.time())
            old_sol = PossibleSolution.copy_from(curr_sol)
            print('time after copy: %f' % time.time())
            curr_sol.tweak()
            print('time after tweak: %f' % time.time())
            if curr_sol.score < old_sol.score:
                curr_sol = old_sol
        if curr_sol.score > curr_best.score:
            curr_best = PossibleSolution.copy_from(curr_sol)
        curr_home = PossibleSolution.copy_from(curr_home.new_home_base(curr_sol))
        curr_sol = PossibleSolution.copy_from(curr_home)
        curr_sol.perturb()
    return curr_best.graph
        
        
# Local search with a steadily decreasing chance to move to an inferior solution
def simulated_annealing(empty, complete, k, timeout=400, post_timeout=200, temp_mult=1250):
    def temp(elapsed_time):
        return temp_mult * (1 - (elapsed_time / timeout))
    def switch_prob(temp, better_qual, worse_qual):
        try:
            return exp((worse_qual - better_qual) / temp)
        except OverflowError:
            return 0

    PossibleSolution.init_statics(empty, complete, k)
    curr_sol = generate_solution(empty, complete, k)
    curr_best = PossibleSolution.copy_from(curr_sol)
    start_time = time.time()
    while time.time() - start_time < timeout:
        print(f"BEST: {curr_best.total_weight=}, {curr_best.score=}")
        print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}")
        old_sol = PossibleSolution.copy_from(curr_sol)
        curr_sol.tweak()
        prob = switch_prob(temp(time.time() - start_time), old_sol.score, curr_sol.score)
        print(prob)
        if old_sol.score > curr_sol.score and random() > prob:
            curr_sol = old_sol
        if curr_sol.score > curr_best.score:
            curr_best = PossibleSolution.copy_from(curr_sol)
    post_start_time = time.time()
    curr_sol = PossibleSolution.copy_from(curr_best)
    while time.time() - post_start_time < post_timeout:
        old_sol = PossibleSolution.copy_from(curr_sol)
        curr_sol.tweak()
        if old_sol.score > curr_sol.score:
            curr_sol = old_sol
        if curr_sol.score > curr_best.score:
            curr_best = PossibleSolution.copy_from(curr_sol)
    return curr_best.graph

def ant_colony_optimization(empty, complete, k, global_timeout=600, local_timeout=30, evaporation=0.9, popsize=10, initial_pheromone=1, sigma=1, epsilon=0):
    PossibleSolution.init_statics(empty, complete, k)
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
                old_sol = PossibleSolution.copy_from(curr_sol)
                curr_sol.tweak()
                if old_sol.score > curr_sol.score:
                    curr_sol = old_sol
            if best is None or curr_sol.score > best.score:
                best = PossibleSolution.copy_from(curr_sol)
            print(f"BEST: {best.total_weight=}, {best.score=}")
            print(f"CURRENT: {curr_sol.total_weight=}, {curr_sol.score=}")
            trails.append(curr_sol)
        for edge in pheromones:
            pheromones[edge] *= (1 - evaporation)
        for trail in trails:
            for edge in trail.graph.edges:
                pheromones[edge] += trail.score
    return best.graph

            
def evolutionary_algorithm(empty, complete, k, timeout=600, num_parents=10, popsize=20, parents_persist=True, genetic=False):
    PossibleSolution.init_statics(empty, complete, k)
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
            if best is None or best.score < sol.score:
                best = sol
        sorted_pop = sorted(curr_pop, key=lambda x: x.score, reverse=True)
        parents = sorted_pop[:num_parents]
        print(f"BEST: {best.total_weight=}, {best.score=}")
        curr_pop = parents[:] if parents_persist else []
        if genetic: # using genetic algorithm with crossover
            while len(curr_pop) < popsize:
                curr_parents = sample(parents, 2)
                children = genetic_crossover(*curr_parents)
                for child in children:
                    child.tweak()
                curr_pop.extend(children)
        else: # using regular evolutionary algorithm
            for parent in parents:
                for child_num in range(popsize // num_parents):
                    tweaked_parent = PossibleSolution.copy_from(parent)
                    tweaked_parent.tweak()
                    curr_pop.append(tweaked_parent)
        print(f'{len(curr_pop)=}')
    return best.graph