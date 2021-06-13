#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
#from ortools.linear_solver import pywraplp
import cvxopt
import cvxopt.glpk
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
cvxopt.glpk.options['tm_lim'] = 3600 * 10 ** 3

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])
from sklearn.metrics.pairwise import euclidean_distances

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution- from here comment out the solution
    # pack the facilities one by one until all the customers are served
    def triv_sol(facilities, customers):
        solution = [-1]*len(customers)
        capacity_remaining = [f.capacity for f in facilities]
        facility_index = 0
        for customer in customers:
           if capacity_remaining[facility_index] >= customer.demand:
               solution[customer.index] = facility_index
               capacity_remaining[facility_index] -= customer.demand
           else:
               facility_index += 1
               assert capacity_remaining[facility_index] >= customer.demand
               solution[customer.index] = facility_index
               capacity_remaining[facility_index] -= customer.demand
        used = [0]*len(facilities)
        for facility_index in solution:
            used[facility_index] = 1

        # calculate the cost of the solution
        obj = sum([f.setup_cost*used[f.index] for f in facilities])
        for customer in customers:
            obj += length(customer.location, facilities[solution[customer.index]].location)
        return obj, solution

    # def ortools_solve_not_mine(facilities, customers, time_limit=None):
    #     import numpy
    #     import datetime
    #     print('Num facilities {}'.format(len(facilities)))
    #     print('Num customers {}'.format(len(customers)))
    #
    #     if time_limit is None:
    #         time_limit = 1000 * 60  # 1 minute
    #
    #     #solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    #     solver = pywraplp.Solver.CreateSolver('SCIP')
    #     # x_i = 1 iff facility i is chosen
    #     x = []  # 1xN
    #     # y_ij = 1 iff custome j is assigned to facility i
    #     y = [[] for x in range(len(facilities))]  # NxM
    #
    #     for i in range(len(facilities)):
    #         x.append(solver.BoolVar('x{}'.format(i)))
    #         for j in range(len(customers)):
    #             y[i].append(solver.BoolVar('y{},{}'.format(i, j)))
    #
    #     print('x variable with dim {}'.format(len(x)))
    #     print('y variable with dim {}x{}'.format(len(y), len(y[0])))
    #
    #     # total demand to 1 facility <= its capacity
    #     for i in range(len(facilities)):
    #         constraint = solver.Constraint(0.0, facilities[i].capacity)
    #         for j in range(len(customers)):
    #             constraint.SetCoefficient(y[i][j], customers[j].demand)
    #
    #     # exactly one facility per customer
    #     for j in range(len(customers)):
    #         constraint = solver.Constraint(1.0, 1.0)
    #         for i in range(len(facilities)):
    #             constraint.SetCoefficient(y[i][j], 1.0)
    #
    #     # y_ij can be 1 only if x_i is 1
    #     for i in range(len(facilities)):
    #         for j in range(len(customers)):
    #             constraint = solver.Constraint(-solver.infinity(), 0.0)
    #             constraint.SetCoefficient(y[i][j], 1.0)
    #             constraint.SetCoefficient(x[i], -1.0)
    #
    #     # objective
    #     objective = solver.Objective()
    #     objective.SetMinimization()
    #     for i in range(len(facilities)):
    #         objective.SetCoefficient(x[i], facilities[i].setup_cost)
    #         for j in range(len(customers)):
    #             objective.SetCoefficient(y[i][j], length(customers[j].location, facilities[i].location))
    #
    #     print('Number of variables =', solver.NumVariables())
    #     print('Number of constraints =', solver.NumConstraints())
    #
    #     solver.set_time_limit(time_limit)
    #     print('OR-Tools starts at {}'.format(datetime.datetime.now().time()))
    #     result_status = solver.Solve()
    #     print(result_status)
    #
    #     val = solver.Objective().Value()
    #     y_val = [[] for x in range(len(facilities))]  # NxM
    #     assignment = []
    #     for i in range(len(facilities)):
    #         for j in range(len(customers)):
    #             y_val[i].append(int(y[i][j].solution_value()))
    #     y_val = numpy.array(y_val)
    #     for j in range(len(customers)):
    #         assignment.append(numpy.where(y_val[:, j] == 1)[0][0])
    #
    #     return val, assignment

    # def dist_matrix(facilities,customers):
    #     fac_loc=[]
    #     cust_loc = []
    #     for i in range(len(facilities)):
    #         fac_loc.append([int(facilities[i].location.x),int(facilities[i].location.y)])
    #     for j in range(len(customers)):
    #         cust_loc.append([int(customers[j].location.x),int(customers[j].location.y)] )
    #
    #     D = euclidean_distances(cust_loc, fac_loc)
    #     return D


    # def or_tools_scip_mine(facilities, customers, time_limit=None):
    #
    #     import numpy
    #     import datetime
    #
    #     if time_limit is None:
    #         time_limit = 1000 * 60  # 1 minute
    #
    #
    #     solver = pywraplp.Solver.CreateSolver('SCIP')
    #
    #
    #     customer_count = range(len(customers))
    #     facility_count = range(len(facilities))
    #     x =[[] for _ in range(len(customers))]
    #     y = []
    #     facility_capacities=[facilities[i][2] for i in facility_count]
    #     facility_setup_costs = [facilities[i][1] for i in facility_count]
    #     demands=[customers[i][1] for i in customer_count]
    #     c=dist_matrix(facilities,customers)
    #
    #     for j in facility_count:
    #         y.append(solver.BoolVar("y(%s)" % j))
    #         for i in customer_count:
    #             x[i].append(solver.BoolVar("x(%s,%s)" % (i, j)))
    #
    #     for i in customer_count:
    #         solver.Add(solver.Sum(x[i][j] for j in facility_count) <= demands[i])#, "Demand(%s)" % i
    #     for j in facility_count:
    #         solver.Add(solver.Sum(x[i][j] for i in customer_count) <= facility_capacities[j] * y[j])#, "Capacity(%s)" % j)
    #     for j in facility_count:
    #         for i in customer_count:
    #             solver.Add(x[i][j] <= demands[i] * y[j])
    #     a=solver.Sum((facility_setup_costs[j] * y[j] for j in facility_count))
    #     b=solver.Sum((c[i, j] * x[i][j] for i in customer_count for j in facility_count))
    #     func_=solver.Sum([a,b])
    #     solver.Minimize(func_)
    #
    #     solver.set_time_limit(time_limit)
    #     result_status = solver.Solve()
    #     print(result_status)
    #     val = solver.Objective().Value()
    #
    #     x_val = [[] for _ in range(len(customers))]  # NxM
    #     solution = []
    #     for j in range(len(facilities)):
    #         for i in range(len(customers)):
    #             x_val[i].append(int(x[i][j].solution_value()))
    #     x_val = numpy.array(x_val)
    #     for j in range(len(customers)):
    #         solution.append(numpy.where(x_val[:, j] == 1)[0][0])
    #
    #     return val, solution

    def scip_solve(facilities, customers, time_limit=None):
        import datetime
        # import sys
        # sys.path.append('C:/Users/User/anaconda3/Lib/site-packages')
        import pyscipopt
        fac = len(facilities)
        cus = len(customers)

        model = pyscipopt.Model('FL')
        model.hideOutput()
        model.setMinimize()
        # model.setRealParam('limits/gap', 0.2)

        # x_i = 1 iff facility i is chosen
        x = []  # 1xN
        # y_ij = 1 iff customer j is assigned to facility i
        y = [[] for x in range(len(facilities))]  # NxM

        for i in range(fac):
            x.append(model.addVar(name='x{}'.format(i), vtype='B'))
            for j in range(cus):
                y[i].append(model.addVar(name='y{},{}'.format(i, j), vtype='B'))

        # total demand to 1 facility <= its capacity
        for i in range(fac):
            model.addCons(
                pyscipopt.quicksum(customers[j].demand * y[i][j] for j in range(cus)) <= facilities[i].capacity)

        # exactly 1 facility per customer
        for j in range(cus):
            model.addCons(
                pyscipopt.quicksum(y[i][j] for i in range(fac)) == 1)

        # y_ij can be 1 only if x_i is 1
        for i in range(fac):
            for j in range(cus):
                model.addCons(y[i][j] <= x[i])

        # objective
        model.setObjective(
            pyscipopt.quicksum(
                # distance facility -> customer
                length(customers[j].location, facilities[i].location) * y[i][j] for i in range(fac) for j in range(cus)
            ) +
            pyscipopt.quicksum(
                # setup cost
                facilities[i].setup_cost * x[i] for i in range(fac)
            ), 'minimize')

        if time_limit is not None:
            model.setRealParam('limits/time', time_limit)
        print('SCIP starts at {}'.format(datetime.datetime.now().time()))
        model.optimize()
        val = model.getObjVal()
        assignment = []
        for j in range(cus):
            for i in range(fac):
                sol = model.getVal(y[i][j])
                if sol == 1:
                    assignment.append(i)
                    break
        return val, assignment

    def mip(facilities, customers,time_limit=None):
        from cvxopt import solvers
        if time_limit is not None:
            solvers.options['glpk'] = {'tm_lim': time_limit}

        M = len(customers)
        N = len(facilities)
        c = []
        for j in range(N):
            c.append(facilities[j].setup_cost)
        for j in range(N):
            for i in range(M):
                c.append(length(facilities[j].location, customers[i].location))

        xA = []
        yA = []
        valA = []
        for i in range(M):
            for j in range(N):
                xA.append(i)
                yA.append(N + M * j + i)
                valA.append(1)

        b = np.ones(M)

        xG = []
        yG = []
        valG = []
        for i in range(N):
            for j in range(M):
                xG.append(M * i + j)
                yG.append(i)
                valG.append(-1)
                xG.append(M * i + j)
                yG.append(N + M * i + j)
                valG.append(1)

        for i in range(N):
            for j in range(M):
                xG.append(N * M + i)
                yG.append(N + M * i + j)
                valG.append(customers[j].demand)
        h = np.hstack([np.zeros(N * M),
                       np.array([fa.capacity for fa in facilities], dtype='d')])

        binVars = set()
        for var in range(N + M * N):
            binVars.add(var)

        status, isol = cvxopt.glpk.ilp(c=cvxopt.matrix(c),
                                       G=cvxopt.spmatrix(valG, xG, yG),
                                       h=cvxopt.matrix(h),
                                       A=cvxopt.spmatrix(valA, xA, yA),
                                       b=cvxopt.matrix(b),
                                       I=binVars,
                                       B=binVars)
        soln = []
        for i in range(M):
            for j in range(N):
                if isol[N + M * j + i] == 1:
                    soln.append(j)

        used = [0] * len(facilities)
        for facility_index in soln:
            used[facility_index] = 1

        obj = sum([f.setup_cost * used[f.index] for f in facilities])
        for customer in customers:
            obj += length(customer.location, facilities[soln[customer.index]].location)

        return obj, soln
    #
    # solution = mip(facilities, customers)
    # ######
    # used = [0] * len(facilities)
    # for facility_index in solution:
    #     used[facility_index] = 1
    #
    # obj = sum([f.setup_cost * used[f.index] for f in facilities])
    # for customer in customers:
    #     obj += length(customer.location, facilities[solution[customer.index]].location)





    ###############
    #solutions
    ###############
    other_model=False
    print('customers',len(customers))
    if len(customers) == 2000:  # instance 8

        obj, solution = triv_sol(facilities, customers)

    elif len(customers) < 201 or (len(customers) == 100 and len(facilities) == 100):  # instances 1,2,3
        time_limit = 180
        obj, solution = mip(facilities, customers,time_limit)
    elif len(customers) == 1000 and len(facilities) == 100:  # instance 4
        time_limit = 600
        obj, solution = mip(facilities, customers,time_limit)

    else:
        # test the or-tools from google
        # obj, solution = ortools_solve(facilities, customers)

        # test the scip suite

        if len(customers) == 800 and len(facilities) == 200:  # instance 5
            time_limit = 600

        else:  # instances 6,7
            time_limit = 1000

        obj, solution = scip_solve(facilities, customers, time_limit=time_limit)


    #obj, solution = scip_solve(facilities,customers)
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data




if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

    # file_location = './data/fl_200_1'
    # with open(file_location, 'r') as input_data_file:
    #     input_data = input_data_file.read()
    # print(solve_it(input_data))