from gurobipy import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



customer_num=15
dc_num=10
minxy=0
maxxy=10
M=maxxy**2
max_dist=3
service_level=0.7
covered_customers=math.ceil(customer_num*service_level)
n=0
customer = np.random.uniform(minxy,maxxy,[customer_num,2])


#Model 1 : Minimize number of warehouses

m = Model()

###Variable
dc={}
x={}
y={}
assign={}

for j in range(dc_num):
    dc[j] = m.addVar(lb=0,ub=1,vtype=GRB.BINARY, name="DC%d" % j)
    x[j]= m.addVar(lb=0, ub=maxxy, vtype=GRB.CONTINUOUS, name="x%d" % j)
    y[j] = m.addVar(lb=0, ub=maxxy, vtype=GRB.CONTINUOUS, name="y%d" % j)

for i in range(len(customer)):
    for j in range(len(dc)):
        assign[(i,j)] = m.addVar(lb=0,ub=1,vtype=GRB.BINARY, name="Cu%d from DC%d" % (i,j))

###Constraint
for i in range(len(customer)):
    for j in range(len(dc)):
        m.addConstr(((customer[i][0] - x[j])*(customer[i][0] - x[j]) +\
                              (customer[i][1] - y[j])*(customer[i][1] - \
                              y[j])) <= max_dist*max_dist + M*(1-assign[(i,j)]))

for i in range(len(customer)):
    m.addConstr(quicksum(assign[(i,j)] for j in range(len(dc))) <= 1)

for i in range(len(customer)):
    for j in range(len(dc)):
        m.addConstr(assign[(i, j)] <= dc[j])

for j in range(dc_num-1):
    m.addConstr(dc[j] >= dc[j+1])

m.addConstr(quicksum(assign[(i,j)] for i in range(len(customer)) for j in range(len(dc))) >= covered_customers)

#sum n
for j in dc:
    n=n+dc[j]

m.setObjective(n,GRB.MINIMIZE)

m.optimize()

print('\nOptimal Solution is: %g' % m.objVal)
for v in m.getVars():
    print('%s %g' % (v.varName, v.x))
#     # print(v)
