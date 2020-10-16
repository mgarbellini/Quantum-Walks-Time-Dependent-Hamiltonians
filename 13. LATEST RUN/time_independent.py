# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import numpy as np
from scipy import linalg
import time
from scipy.optimize import minimize, basinhopping
import ray
#import constrNMPy as cNM
#from scipy.optimize import shgo

import matplotlib.pyplot as plt

# # # # # # # # # # # # # # #
# FUNCTIONS IMPLEMENTATION  #
# # # # # # # # # # # # # # #
#global definition
global dimension

#Implementation of generate_hamiltonian function
#this routine takes hilbert space dimension (n) as input
#and outputs a nxn matrix corrisponding to H = L + gamma
def generate_hamiltonian(dimension, gamma):

    #generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
    diag_matrix.fill(0)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                diag_matrix[i][j] = 2
            else:
                diag_matrix[i][j] = 0

    #generate loop adjacency matrix
    adj_matrix = np.empty([dimension, dimension])
    adj_matrix.fill(0)
    for i in range(dimension): #colonna
        for j in range(dimension): #riga
            if i == j:
                if i == 0 & j == 0:
                    adj_matrix[i][dimension-1] = 1
                    adj_matrix[i][j+1] = 1
                elif i == dimension-1 & j == dimension-1:
                    adj_matrix[i][j-1] = 1
                    adj_matrix[i][0] = 1
                else:
                    adj_matrix[i][j-1] = 1
                    adj_matrix[i][j+1] = 1

    #generate laplacian of loop
    hamiltonian = diag_matrix - adj_matrix

    #generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    index = int((dimension-1)/2)
    hamiltonian[index][index] += - gamma

    return hamiltonian

#Implementation of time evolution with time-independent hamiltonian
#and evalute 'crossing-to-oracle' probability for a given time t
def evaluate_probability(x):

    #generate_hamiltonian (really memory and time inefficient)
    hamiltonian = generate_hamiltonian(dimension, x[0])

    #Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1/np.sqrt(dimension))

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #set to zero variables
    probability = 0
    psi_t = np.empty([dimension, 1])
    psi_t.fill(0)

    #define time-evolution
    unitary_time_evolution = np.empty([dimension, dimension])
    unitary_time_evolution.fill(0)
    unitary_time_evolution = linalg.expm(-(1j)*hamiltonian*x[1])
    psi_t = np.dot(unitary_time_evolution, psi_0)
    probability = np.dot(oracle_site_state.transpose(), psi_t)

    #return 'crossing' probability
    return -np.abs(probability)**2


@ray.remote
def grid_eval(time_array, beta_array):

    probability = np.empty([len(beta_array),len(time_array) ])
    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            #evaluate probability
            probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]])

    return probability

def parallel_routine():

    #load beta and time arrays
    beta = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                    1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
                    2, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
                    3, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95]
    # initialize ray multiprocessing
    ray.init()
    tic = time.perf_counter()

    # useful definitions for multicore computation
    cpu_count = 3
    #time_sampling_points = 24
    sampling_per_cpu_count = len(beta)/3
    if(int(len(beta)/3) < (len(beta)/3)):
        print("ERROR CPU BETA COUNT")

    #beta_sampling_points = cpu_count * sampling_per_cpu_count
    #beta = np.linspace(lb_beta, ub_beta, beta_sampling_points)
    #time_array = np.linspace(lb_time, ub_time, time_sampling_points)
    time_0 = (np.pi/4)*np.sqrt(dimension)
    time_1 = np.sqrt(dimension)
    time_2 = (np.pi/2)*np.sqrt(dimension)
    time_3 = 2*np.sqrt(dimension)
    time_array = [time_0, time_1, time_2, time_3]
    process = []
    probability = []

    # parallel processes
    for i in range(cpu_count):
        process.append(grid_eval.remote(time_array, beta[int(
            sampling_per_cpu_count * i):int(sampling_per_cpu_count * (i + 1))]))

    # reassigning values to arrays
    for i in range(cpu_count):
        probability.append(ray.get(process[i]))

    # concatenate arrays to output array
    probability_array = np.concatenate(
        [probability[0], probability[1]], axis=0)
    for i in range(cpu_count - 2):
        probability_array = np.concatenate(
            [probability_array, probability[i + 2]])

    # shutting down ray
    ray.shutdown()


    toc = time.perf_counter() - tic

    print('Success: N ', dimension, ' in ', int(toc / 60), ' min')


    # preparing for export and export and miscellanea
    file_probability = str(dimension) + '_probability_static.npy'
    np.save(file_probability, probability_array)

    return 0

if __name__ == '__main__':

    step_function = 1
    rtolerance = 1e-6
    atolerance = 1e-6

    min_dim = 3
    max_dim = 71
    dims = np.arange(min_dim, max_dim + 1, 2)
    for dim in dims:
        dimension = dim
        parallel_routine()
