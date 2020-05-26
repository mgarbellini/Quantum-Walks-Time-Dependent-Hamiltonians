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

    prob = '_probability.npy'
    t = '_time_array.npy'
    b = '_beta_array.npy'
    adiab = '_adiabatic_check.npy'
    non = '_non_adiab'

    if(flag == 1):

        probability = np.load(str(dim) + prob)
        time = np.load(str(dim) + t)
        beta = np.load(str(dim) + b)
        adiabatic_flag = np.load(str(dim) + adiab)

        return probability, time, beta, adiabatic_flag

    elif(flag == 0):

        probability = np.load(str(dim) + non + prob)
        time = np.load(str(dim) + non + t)
        beta = np.load(str(dim) + non + b)

        return probability, time, beta

    else:

        print('Error: undefined file-loading flag')
        return 0

def adiabatic_delta(dim):

    probability, time, beta, adiabatic_flag = load_data_file(dim,1)

    for i in range(len(time)):
        for j in range(len(beta)):
            if(adiabatic_flag[j][i]==1):
                probability[j][i] = float(time[i]/probability[j][i])
            else:
                probability[j][i] = 10e4

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]
    print(min, (1/min)*time[min_index[1]], time[min_index[1]], beta[min_index[0]])

def adiabatic_no_constrains(dim):

    probability, time, beta, adiabatic_flag = load_data_file(dim,1)

    for i in range(len(time)):
        for j in range(len(beta)):
            if(time[i]>10):
                probability[j][i] = float(time[i]/probability[j][i])
            else:
                probability[j,i] = 10e4

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0],min_index[1]]
    a_flag = adiabatic_flag[min_index[0],min_index[1]]
    if(a_flag==0):
        print('Unphysical result!')
    print(min, (1/min)*time[min_index[1]], time[min_index[1]], beta[min_index[0]])

def non_adiabatic_delta(dim):

    probability, time, beta = load_data_file(dim,0)

    for i in range(len(time)):
        for j in range(len(beta)):
                if(time[i]>5):
                    probability[j][i] = float(time[i]/probability[j][i])
                else:
                    probability[j][i] = 10e4

    min_index = np.unravel_index(probability.argmin(), probability.shape)
    min = probability[min_index[0]][min_index[1]]
    print(min, (1/min)*time[min_index[1]], time[min_index[1]], beta[min_index[0]])

if __name__ == '__main__':

    adiabatic_delta(31)
    dimensions = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 41, 51, 61, 71]

    for dim in dimensions:
        adiabatic_delta(dim)
