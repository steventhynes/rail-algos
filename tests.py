import time
from naive import *

def test_add_edge_and_eval():
    cities = cities_from_file('data/us-cities-top-1k.csv')
    complete = complete_solution(cities)
    greedy_sol = greedy_buildup(cities, 1000)
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
    greedy_sol = greedy_buildup(cities, 1000)
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

def test_approx_evaluate_solution():
    cities = cities_from_file('data/us-cities-top-1k.csv')
    # complete = complete_solution(cities)
    greedy_sol = greedy_buildup(cities, 1000)
    sorted_nodes = sorted(greedy_sol.nodes, key=lambda x: greedy_sol.nodes.data()[x]['population'], reverse=True)

    # print(evaluate_solution(greedy_sol)[1])
    t = time.time()
    print(approx_evaluate_solution(greedy_sol, [], len(greedy_sol)))
    print(approx_evaluate_solution(greedy_sol, [], 200))
    print(approx_evaluate_solution(greedy_sol, sorted_nodes[:20], 180))
