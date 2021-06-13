#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import lru_cache
import time
from scipy.sparse import csc_matrix
import numpy as np
from operator import attrgetter
from math import log10

import numpy as np
import cvxopt
import cvxopt.glpk
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'


Item = namedtuple("Item", ['index', 'value', 'weight'])

@lru_cache(maxsize = 128)
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    weights = []
    values=[]
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
        weights.append(Item(i-1, int(parts[0]), int(parts[1]))[2])
        values.append(Item(i-1, int(parts[0]), int(parts[1]))[1])



    # # 1. a trivial algorithm for filling the knapsack
    # # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)
    #
    # for item in items:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight

    # # 2.Dynamic programming
    # N=item_count
    # K=capacity
    # import numpy as np
    # B = np.zeros((N+1,K+1),dtype=np.int16)
    # #taken = (np.zeros((N+1,K+1,N),dtype=np.int16)) #csc_matrix((N+1,K+1,N),shape=(N+1,K+1,N),dtype = np.int8).toarray()#
    #
    # for n in range(N + 1):# rows- items
    #     weights_for_taken = [0] * len(items)
    #     for k in range(K + 1):#columns -capacity
    #         if n==0 or n == 0:
    #             B[n][k]=0
    #         elif weights[n - 1] <= k:
    #             B[n][k] = int(max((values[n-1] + B[n-1][k - weights[n-1]]), B[n-1][k]))
    #
    #             # if (values[n-1] + B[n-1][k - weights[n-1]])>B[n-1][k]:
    #             #     weights_for_taken[n-1]=1
    #             #     taken[n][k]=weights_for_taken+taken[n-1][k - weights[n-1]]
    #             # else:
    #             #     taken[n][k]=taken[n-1][k]
    #         else:
    #             B[n][k] = B[n][k - 1]
    #             #taken[n][k] = taken[n][k-1]

    # # prepare the solution in the specified output format
    # i.
    # import numpy as np
    # B = np.array(B)
    # indexes=np.where(B == np.amax(B))
    # Ni=indexes[0][-1]
    # Ki=indexes[1][-1]
    # value=B[Ni][Ki]
    #
    # ii.
    # taken_new=[0] * len(items)
    # totalWeight=Ki
    # for i in reversed(range(Ni)):
    #     if B[i][totalWeight] == B[i + 1][totalWeight]:
    #         continue
    #     else:
    #         taken_new[i] = 1
    #         totalWeight -= items[i].weight

    # import itertools
    #
    # iii.
    # result = [seq for i in range(len(values), 0, -1) for seq in itertools.combinations(values, i) if sum(seq) == value]
    # location = [0] * len(items)
    # if result:
    #     for j in range(len(result)):
    #         location = [0] * len(items)
    #         for i in range(np.shape(result[j])[0]):
    #             loc=np.where(np.array(values) == result[j][i])[0][0]
    #             location[loc]=1
    #
    #         if np.sum(np.array(weights)*[location]) <= K:
    #             #print('location is:', location)
    #             break
    #
    # else:
    #     loc=np.where(np.array(values) == value)[0]
    #     location[loc]=1
    #
    # taken_out= [int(i) for i in list(taken[Ni][Ki])]

    # 3. branch and bound

    # 4.mixed integer
    def mip(cap, items):
        item_count = len(items)
        values = np.zeros(item_count)
        weights = np.zeros([1, item_count])

        for i in range(item_count):
            values[i] = items[i].value
            weights[0][i] = items[i].weight

        binVars = set()
        for var in range(item_count):
            binVars.add(var)

        status, isol = cvxopt.glpk.ilp(c=cvxopt.matrix(-values, tc='d'),
                                       G=cvxopt.matrix(weights, tc='d'),
                                       h=cvxopt.matrix(cap, tc='d'),
                                       I=binVars,
                                       B=binVars)
        taken = [int(val) for val in isol]
        value = int(np.dot(values, np.array(taken)))
        return value, taken





    value, taken = mip(capacity, items)

    output_data = str(int(value)) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    #output_data += ' '.join(map(str, taken_out)) #dp
    #output_data += ' '.join(map(str, location)) #dp
    #output_data += ' '.join(map(str, taken)) #trivial
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

    # file_location = './data/ks_100_0'
    # with open(file_location, 'r') as input_data_file:
    #     input_data = input_data_file.read()
    # print(solve_it(input_data))
