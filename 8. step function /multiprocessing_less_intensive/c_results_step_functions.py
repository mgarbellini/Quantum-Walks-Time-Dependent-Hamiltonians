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

    prob_sqrt = '_probability_sqrt.npy'
    t = '_time_array.npy'
    b = '_beta_array.npy'
    prob_pow2 = '_probability_cbrt.npy'

    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)

    reduced_time = int(len(time)/2)

    if(flag == 1):

        probability_sqrt = np.load(str(dim) + prob_sqrt)
        return probability_sqrt, time[0:reduced_time], beta

    elif(flag == 0):

        probability_pow2 = np.load(str(dim) + prob_pow2)

        return probability_pow2, time[0:reduced_time], beta

    else:

        print('Error: undefined file-loading flag')
        return 0

def dynamic_implementation(dim, flag):

    probability, time, beta = load_data_file(dim,flag)

    for i in range(len(time)):
        for j in range(len(beta)):
            if(time[i]>5):
                probability[j][i] = float(time[i]/probability[j][i])
            else:
                probability[j,i] = 10e4

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]
    iters_to_unity = float(1/((1/min)*time[min_index[1]]))
    if(flag==0):
        print("sqrt - step function")
    else:
        print("cbrt - step function")

    print(dim, round((1/min)*time[min_index[1]],2), round(time[min_index[1]],1), round(beta[min_index[0]],1), round(min,2), round(iters_to_unity, 1))


if __name__ == '__main__':

    dim = int(sys.argv[1])
    dynamic_implementation(dim, 0)
    dynamic_implementation(dim, 1)
