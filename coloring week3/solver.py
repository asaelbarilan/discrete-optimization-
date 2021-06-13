#!/usr/bin/python
# -*- coding: utf-8 -*-
from cvxopt.modeling import op
# import cvxpy as cp
import numpy as np
import constraint


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    M = np.zeros((node_count, node_count), dtype=int)
    for i in edges:
        M[i[0]][i[1]] = 1
        M[i[1]][i[0]] = 1

    solution = [1] * (node_count)

    def node_degree(M):
        # from brooks theoram a graph cant more than max degree
        return (np.sum(M, axis=0))

    max_degree = max(node_degree(M))

    # 0.
    # build a trivial solution
    # # every node has its own color
    # def trivial_solution(node_count):
    #     solution = range(0, node_count)
    #     return solution

    # 1.
    # build a constraint programming solution
    # def cp_basic(node_count,edge_count,edges,M,solution):
    #     for k in range(node_count):
    #         if k==0:
    #             solution[0]=0
    #         else:
    #             j=list(M[k][:]).index(0)
    #             if k== j:
    #                 solution[k] = max(solution)+1
    #             else:
    #                 solution[k]=M[k][j]
    #     colors=list(set(solution))
    #     solution=list(np.array(solution, int))
    #     return solution,colors



    # 2.
    def cp_with_constraint(node_count, edge_count, edges, M, solution, max_degree):

        import constraint
        problem = constraint.Problem()

        for i in range(node_count):
            problem.addVariable(i, range(0, node_count))

        for edge in edges:
            problem.addConstraint(lambda a, b: a != b, (edge[0], edge[1]))

        solution = problem.getSolutions()[0]
        x = [solution[node] for node in range(node_count)]
        return x

    #
    # .3
    # working!
    def cp_basic_3(node_count, edge_count, edges, M, solution, max_degree):
        # i can minimize colors with max degree
        for k in range(node_count):
            j = M[k][:]
            if k > 0:
                if solution[k] in solution * j:
                    solution[k] = max(solution) + 1
                else:
                    continue
        return solution


    #

    # .4
    # def cp_basic_4(node_count,edge_count,edges,M,solution,max_degree):
    #     #i can minimize colors with max degree
    #     for v in range(node_count):
    #         for u in range(node_count):
    #             if (u, v) in edges and solution[u] == solution[v]:
    #                 solution[u] = solution[v] + 1
    #     return solution

    # .5
    # working!
    def cp_basic_5(node_count, edge_count, edges, M, solution, max_degree):
        import networkx as nx
        # G = nx.Graph()
        # G.add_nodes_from(range(node_count))
        G = nx.Graph(M)
        G.add_edges_from(edges)
        if node_count < 100:
            D = nx.coloring.greedy_color(G=G, strategy=nx.coloring.strategy_largest_first)
        else:
            D = nx.coloring.greedy_color(G=G, strategy=nx.coloring.strategy_independent_set)
        for i in range(node_count):
            solution[list(D.keys())[i]] = list(D.values())[i]
        return solution

    # TO  DO: need to learn how to work with!
    # 6.
    # def cp_with_cvxopt_glpk(node_count,edge_count,edges,M,solution,max_degree):
    #     from cvxopt import solvers
    #     from cvxopt.modeling import variable, op
    #     import cvxopt.glpk
    #     cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
    #
    #     colors=variable(node_count, 'colors')
    #     for e in edges:
    #         op.addconstraint(colors(e[0])!=colors(e[1]))
    #
    #     cp = op(min(sum(colors)), [constraint])
    #     cp.solve()
    #
    #     return colors.value

    # 7.
    # def cp_with_cvxpy(node_count,edge_count,edges,M,solution,max_degree):
    #     import cvxpy as cp
    #     import numpy as np
    #
    #     # Problem data.
    #     m = 30
    #     n = 20
    #     np.random.seed(1)
    #     A = np.random.randn(m, n)
    #     b = np.random.randn(m)
    #
    #     # Construct the problem.
    #     x = cp.Variable(n)
    #     objective = cp.Minimize(cp.sum_squares(A * x - b))
    #     constraints = [0 <= x, x <= 1]
    #     prob = cp.Problem(objective, constraints)
    #
    #     # The optimal objective value is returned by `prob.solve()`.
    #     result = prob.solve()
    #     # The optimal value for x is stored in `x.value`.
    #     print(x.value)
    #
    #     return solution.value

    # .8
    def cp_with_ortools(node_count, edge_count, edges, M, solution, max_degree):
        global colors
        from ortools.sat.python import cp_model
        max_degree=int(max_degree)
        # Creates the model.
        model = cp_model.CpModel()
        vars_array=[0] * (node_count)
        # Creates the variables.
        for i in range(node_count):
            #vars_array[i] = model.NewIntVar(0, int(max_degree), 'x[%i]' % i)
            vars_array[i] = model.NewIntVar(0, int(node_count), 'x[%i]' % i)

        # Adds  constraint
        for edge in edges:
            # model.Add(vars_array[edge[0]-1] != vars_array[edge[1]-1])
            model.Add(vars_array[edge[0]] != vars_array[edge[1]])


        # # symmetry breaking
        # for i in range(max_degree):
        #     model.Add(vars_array[i] <= i + 1)

        # Create the objective function
        # obj_var = model.NewIntVar(0, max_degree, 'makespan')
        obj_var = model.NewIntVar(0, node_count, 'makespan')
        model.AddMaxEquality(obj_var,vars_array)
        model.Minimize(obj_var)
        #model.Minimize(sum(solution))



        # Creates a solver and solves the model.
        solver = cp_model.CpSolver()

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            colors = [solver.Value(vars_array[i]) for i in range(node_count)]

        return colors

    colors = cp_with_ortools(node_count, edge_count, edges, M, solution, max_degree)
    # colors2 = cp_basic_5(node_count, edge_count, edges, M, solution, max_degree)
    # colors3 = cp_basic_3(node_count, edge_count, edges, M, solution, max_degree)
    #colors4 = cp_with_constraint(node_count, edge_count, edges, M, solution, max_degree)
    # colors = list(set(solution))
    # prepare the solution in the specified output format
    output_data = str(len(colors)) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, colors))

    return output_data


import sys

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

    file_location = './data/gc_20_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
