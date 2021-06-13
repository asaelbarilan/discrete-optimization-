# #!/usr/bin/python
# # -*- coding: utf-8 -*-
#
import math
from collections import namedtuple
import networkx as nx
import random
from functools import lru_cache
import math
import time
import random
import pdb
from datetime import datetime
from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import numba
from numba import jit
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
from math import sqrt
import threading
Point = namedtuple("Point", ['x', 'y'])

#@jit(nopython=True)
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
def cost(solution,points):
    total_distance=0
    for i in solution:
        if i>=(len(solution)-1):
            k=0
        else:
            k=i+1
        total_distance+=length(points[i],points[k])
    total_distance += length(points[solution[0]], points[solution[-1]])
    return total_distance

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    global POINTS
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))



    def random_graph(nodeCount,points):
        G=nx.DiGraph()
        radnom_list=[]
        for i in range(len(points)):
            G.add_node(i, posxy=(points[i][0], points[i][1]))
            if i<(len(points)-1):
                G.add_edge(i,i+1)
            else:
                G.add_edge(i, 0)
            radnom_list.append('p'+str(i))
        M=nx.adjacency_matrix(G).toarray()
        return G,M

    def random_solution(points):
        import random
        numLow=0
        numHigh=len(points)
        data = list(range(numLow, numHigh))
        random.shuffle(data)
        return data

    def neighbours(s, i):
        n1=0
        n2=0
        n_1, n_2=0,0
        if i > 0 and i < (len(s)-1):
            n1 = s[i - 1]
            n2 = s[i + 1]
            n_1= i - 1
            n_2= i + 1
        elif i == 0:
            n1 = s[-1]
            n2 = s[i + 1]
            n_1 = len(s)-1
            n_2 = i + 1
        elif i == s[-1]:
            # not totally correct
            n1 = s[i - 1]
            n2 = s[0]
            n_1 = i - 1
            n_2 = 0

        return n1, n2,n_1, n_2

    def best_neighbour(points,s, i):
        n1, n2,n_1, n_2 = neighbours(s, i)
        l1=length(points[i],points[n1])
        l2=length(points[i], points[n2])
        if l1<l2:
            return n1,n_1
        else:
            return n2,n_2

    def swap(points,s):
        i=random.randint(0,(len(s)-1))
        n,n_i=best_neighbour(points, s, i)
        s_new=s.copy()
        s_new[n_i]=s[i]
        s_new[i]=n
        return s_new


    def dist_matrix(nodeCount,points):


        D = np.zeros((nodeCount, nodeCount))
        data=[]


        start = time.time()
        for k in range(len(points)):
            data.append([int(points[k].x),int(points[k].y)])


        # start = time.time()
        # df = pd.DataFrame(data, columns=['xcord', 'ycord'], index=list(range(len(points))))
        # G=pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
        # D=G.values
        # end= time.time()
        # #print('pandas time',(end - start))

        #3
        D=euclidean_distances(data, data)


        #4
        # for i in range(len(points)):
        #     for j in range(len(points)):
        #         D[i][j] = (length(points[i], points[j]))


        return D


    # build a trivial solution
    # visit the nodes in the order they appear in the file
    def triv(nodeCount):
        return  range(0, nodeCount)

    #@jit(nopython=True)
    def cost_change(cost_mat, n1, n2, n3, n4):
        return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


    def two_opt(route, cost_mat):
        best = np.array(route)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
        return best

    # 1.simulated annealing


    #@jit(nopython=True)
    def simulated_annealing(nodeCount,points):
        def temperature(T):
            alpha=0.99
            T = alpha * T
            return T

        def P(E_1,E_2,T):
            p=np.exp(-(E_2 - E_1) / T)
            return p
        D=dist_matrix(nodeCount,points)
        # initialize random solution
        s= random_solution(points)
        s_best=s.copy()
        T=30

        # k_max=10000
        # for k in range(k_max):
        k=100
        if nodeCount<=100:
            k=10000
        elif 100<=nodeCount<=1000:
            k=1000
        elif nodeCount>1000:
            k=100
        for k_ in range(k):
            s_=two_opt(s,D)#random_solution(points) #for a  more but accurate solution- two_opt(s,D) can be used
            #keeps that s_ is different from s
            while np.any(s_!=s):
                break
            s_new=s_ #swap(points,s) #my version of two_opt
            c=cost(s,points)
            c_new=cost(s_new,points)
            if c_new < c:
                s, c = s_new, c_new

            elif P(c,c_new,T)>=random.uniform(0, 1):
                s, c = s_new, c_new
            T = temperature(T)
        return s

    # threads = []
    # for n in range(1, 4):
    #     t = threading.Thread(target=simulated_annealing, args=(nodeCount,points))
    #     t.start()
    #     threads.append(t)
    #
    # # wait for the threads to complete
    # for t in threads:
    #      t.join()




    #2.routing with or tools


    def create_data_model(D):
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = D # yapf: disable
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    def print_solution(manager, routing, solution):
        """#prints solution on console."""
        ##print('Objective: {} miles'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = []#Route for vehicle 0:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            #route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        #plan_output += ' {}\n'.format(manager.IndexToNode(index))
        ##print(plan_output)
        #plan_output += 'Route distance: {}miles\n'.format(route_distance)
        return plan_output

    #@jit(nopython=True)
    def or_main(nodeCount,points):
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp

        """Entry point of the program."""
        # Instantiate the data problem.
        global sol
        D=dist_matrix(nodeCount,points)
        data = create_data_model(D)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        k = 100

        if  nodeCount <= 1000:
            k = 1000
        elif nodeCount > 1000:
            k = 17000
        search_parameters.time_limit.seconds =k
        search_parameters.log_search = True


        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # #print solution on console.
        if solution:
            sol=print_solution(manager, routing, solution)
        return sol

    #3. py2opt solution
    def py2opt(nodeCount,points):
        from py2opt.routefinder import RouteFinder
        cities_names = list(range(nodeCount))
        dist_mat = dist_matrix(nodeCount,points)
        route_finder = RouteFinder(dist_mat, cities_names, iterations=5)
        best_distance, best_route = route_finder.solve()
        return best_route



    #tips:
    #visualisation, fast N search,do i need every edge
    ######################################################################
    start=time.time()
    solution=simulated_annealing(nodeCount,points)
    end = time.time()
    print('it took: '+str(end-start)+' sec')
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data




if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     #print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

    file_location = './data/tsp_5_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))


