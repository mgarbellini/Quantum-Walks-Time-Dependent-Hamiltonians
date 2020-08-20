# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping, shgo, dual_annealing
from scipy.integrate import odeint, solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import ray

global result_type
global dimension
def load_data_file(dim, flag):

    data_type = ["_circ_prob_static.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy", "_circ_prob_static_small_time.npy"]
    probability = np.load(str(dim) + data_type[flag])

    return probability

def routine(dim, flag, time_index, beta):

    prob = load_data_file(dim,flag)
    probability = prob[:,time_index]
    for i in range(len(beta)):
        probability[i] = float(time[time_index]/probability[i])

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index]

    if(result_type == 0): #return delta
        return min
    elif(result_type == 1): #return iter
        return min*(1/(time[time_index]))
    elif(result_type == 2): #return beta
        return beta[min_index]

if __name__ == '__main__':

    dimension = int(sys.argv[1])
    result_type = int(sys.argv[2])
    t = '_circ_time.npy'
    b = '_circ_beta.npy'
    time = np.load(str(dimension) + t)
    beta = np.load(str(dimension) + b)


    for i in range(len(time)):
        print(time[i], routine(dimension,0 , i, beta), routine(dimension,1 , i, beta), routine(dimension,2 , i, beta), routine(dimension,4 , i, beta))
