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

def load_data_file(dim, flag):

    data_type = ["_circ_prob_static.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy", "_circ_prob_static_small_time.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'

    t_small = '_circ_small_time.npy'
    b_small = '_circ_small_beta.npy'

    probability = np.load(str(dim) + data_type[flag])
    time = np.load(str(dim) + t_small)
    beta = np.load(str(dim) + b_small)

    return probability, time, beta

def delta_with_constrains(dim, flag):
    probability, time, beta = load_data_file(dim,flag)

    for i in range(len(time)):
        for j in range(len(beta)):
            if(i==0):
                probability[j][i] = 1000
            elif(i==1):
                probability[j][i] = 1000
            else:
                probability[j][i] = float(time[i]/probability[j][i])

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]
    return min*(1/(time[min_index[1]]))

def delta_no_costrains(dim, flag):
    probability, time, beta = load_data_file(dim,flag)

    for i in range(len(time)):
        for j in range(len(beta)):
            if(i==0):
                probability[j][i] = 1000
            else:
                probability[j][i] = float(time[i]/probability[j][i])

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]
    return min*(1/(time[min_index[1]]))

def all_s(dim):
    flag=[]
    for i in range(5):
        flag.append(delta_no_costrains(dim,i))

    print(dim, flag[0], flag[1], flag[2], flag[3], flag[4])



if __name__ == '__main__':

    min_dim = 3
    max_dim = 51
    dimensions = np.arange(min_dim, max_dim + 1, 2)



    for dim in dimensions:
        print(dim, delta_no_costrains(dim, 5))
