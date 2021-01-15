# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np

global result_type

def load_data_file(dim, flag):

    data_type = ["_circ_prob_static.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy", "_circ_prob_static_small_time.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'

    #t_small = '_circ_small_time.npy'
    #b_small = '_circ_small_beta.npy'

    probability = np.load(str(dim) + data_type[flag])
    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)

    return probability, time, beta

def get_constrain_id(time, constrain_time):
    for i in range(len(time)-1):
        if (time[i] == constrain_time):
            return i
        elif(time[i] < constrain_time):
            if(constrain_time < time[i+1]):
                return i+1


def constrained_routine(dim, flag):

    #load all the necessary files
    probability, time, beta = load_data_file(dim,flag)

    #routines to get constrain time ID
    constrain_time = np.sqrt(dim)
    constrain_id = get_constrain_id(time, constrain_time)

    #routine to exclude values that are out of bound (cfr. constrain_time)
    for i in range(len(time)):
        for j in range(len(beta)):
            if(i<constrain_id):
                probability[j][i] = 10000
            else:
                probability[j][i] = float(time[i]/probability[j][i])

    #minimum of (T/p)
    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]

    #different types of results available
    if(result_type == 0): #return delta
        return min
    elif(result_type == 1): #return iter
        return min*(1/(time[min_index[1]]))
    elif(result_type == 2): #return time
        return time[min_index[1]]
    elif(result_type == 3): #return beta
        return beta[min_index[0]]


def all_s(dim):
    flag=[]
    ranges = [0,1,2,4]
    for i in ranges:
        flag.append(constrained_routine(dim,i))

    print(dim, flag[0], flag[1], flag[2], flag[3])




if __name__ == '__main__':

    min_dim = 3
    max_dim = 71
    dimensions = np.arange(min_dim, max_dim + 1, 2)

    result_type = int(sys.argv[1])

    print('# dim, static, lin, sqrt, cerf')
    for dim in dimensions:
        all_s(dim)
