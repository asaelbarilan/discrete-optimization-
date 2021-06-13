# Copyright 2021 Hakan Kjellerstrand hakank@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

  Simple coloring problem using MIP in OR-tools CP-SAT Solver.

  Problem instance from GLPK:s model color.mod

  This is a port of my old OR-tools CP model.

  This model was created by Hakan Kjellerstrand (hakank@gmail.com)
  Also see my other OR-tools models: http://www.hakank.org/or_tools/
"""
import numpy as np
import sys

import math, sys


# from cp_sat_utils import *


def solve_it(input_data):


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
    def cp_with_ortools(node_count, edge_count, edges, M, solution, max_degree):
        global colors
        from ortools.sat.python import cp_model
        import sys
        from ortools.sat.python import cp_model as cp
        import math, sys

        model = cp.CpModel()

        # max number of colors
        # [we know that 4 suffices for normal, planar, maps]
        nc =int(max_degree)

        # number of nodes
        n = node_count#11
        # set of nodes
        V = range(n)

        num_edges = edge_count#20

        #
        # Neighbours
        #
        # This data correspond to the instance myciel3.col from:
        # http://mat.gsia.cmu.edu/COLOR/instances.html
        #
        # Note: 1-based (adjusted below)
        E =edges
        #
        # decision variables
        #
        x = [model.NewIntVar(1, nc, 'x[%i]' % i) for i in V]
        max_color = model.NewIntVar(0, nc, "max_color")

        # number of colors used, to minimize

        model.AddMaxEquality(max_color, x)

        #
        # constraints
        #

        # adjacent nodes cannot be assigned the same color
        # (and adjust to 0-based)
        for i in range(num_edges):
            model.Add(x[E[i][0] - 1] != x[E[i][1] - 1])

        # symmetry breaking
        # solver.Add(x[0] == 1);
        # solver.Add(x[1] <= 2);
        for i in range(nc):
            model.Add(x[i] <= i + 1)

        # objective (minimize the number of colors)
        model.Minimize(max_color)

        #
        # solution
        #
        solver = cp.CpSolver()
        status = solver.Solve(model)

        if status == cp.OPTIMAL:
            colors=[solver.Value(x[i]) for i in V]
            print("x:", [solver.Value(x[i]) for i in V])
            print("max_color:", solver.Value(max_color))
            print()

        print()
        # print("num_solutions:", num_solutions)
        print("NumConflicts:", solver.NumConflicts())
        print("NumBranches:", solver.NumBranches())
        print("WallTime:", solver.WallTime())
        return colors

    colors = cp_with_ortools(node_count, edge_count, edges, M, solution, max_degree)
    output_data = str(len(colors)) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, colors))

    return output_data

if __name__ == '__main__':

    #regular[1, 3, 1, 1, 1, 4, 1, 2, 1, 5, 1, 2, 1, 6, 7, 1, 8, 9, 10, 1]
    #or1   2 1 2 2 1 0 2 0 0 0 0 1 1 2 0 1 1 0 0 0
    #or1.1 0 2 1 1 1 0 1 0 0 0 0 2 1 2 0 1 2 0 0 0
    #or1.2 0 2 2 2 2 1 1 1 0 1 1 1 0 0 1 1 0 0 0 2
    #      0 2 2 2 2 1 1 1 0 1 1 1 0 0 1 1 0 0 0 2
    #or2   1 2 1 1 3 2 2 2 1 3 1 2 2 2 1 3 3 1 1 1
    file_location = './data/gc_20_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))