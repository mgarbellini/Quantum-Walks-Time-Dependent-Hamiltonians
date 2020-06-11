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
ray.init()
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
def grid_eval(time_lb, time_up, beta_array):

    time_sampling_points = 40
    #Define oracle site state

    #Define time, beta and probability and adiabatic_check array
    time_array = np.linspace(time_lb, time_up, time_sampling_points)
    probability = np.empty([len(beta_array), time_sampling_points])
    adiabatic_check  = np.empty([len(beta_array), time_sampling_points])

    for i in range(time_sampling_points):
        for j in range(len(beta_array)):
            #evaluate probability
            probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]])

    return probability, time_array, beta_array

def parallel_routine(lb_time, ub_time):

    tic = time.perf_counter()

    #beta arrays
    beta_1 = [0.1, 0.2, 0.3, 0.4]
    beta_2 = [0.5, 0.6, 0.7, 0.8]
    beta_3 = [0.9, 1.0, 1.1, 1.2]
    beta_4 = [1.3, 1.4, 1.5, 1.6]
    beta_5 = [1.7, 1.8, 1.9, 2.0]
    beta_6 = [2.1, 2.2, 2.3, 2.4]
    beta_7 = [2.5, 2.6, 2.7, 2.8]

    #parallel processes
    process_1 = grid_eval.remote(lb_time, ub_time, beta_1)
    process_2 = grid_eval.remote(lb_time, ub_time, beta_2)
    process_3 = grid_eval.remote(lb_time, ub_time, beta_3)
    process_4 = grid_eval.remote(lb_time, ub_time, beta_4)
    process_5 = grid_eval.remote(lb_time, ub_time, beta_5)
    process_6 = grid_eval.remote(lb_time, ub_time, beta_6)
    process_7 = grid_eval.remote(lb_time, ub_time, beta_7)

    #reassigning values to arrays
    probability_1, time_array_1, beta_array_1 = ray.get(process_1)
    probability_2, time_array_2, beta_array_2 = ray.get(process_2)
    probability_3, time_array_3, beta_array_3 = ray.get(process_3)
    probability_4, time_array_4, beta_array_4 = ray.get(process_4)
    probability_5, time_array_5, beta_array_5 = ray.get(process_5)
    probability_6, time_array_6, beta_array_6 = ray.get(process_6)
    probability_7, time_array_7, beta_array_7 = ray.get(process_7)

    #concatenate arrays to output array
    toc = time.perf_counter() - tic
    print(toc)

    probability = np.concatenate([probability_1, probability_2, probability_3, probability_4, probability_5, probability_6, probability_7], axis=0)
    beta_array = np.concatenate([beta_array_1, beta_array_2, beta_array_3, beta_array_4, beta_array_5, beta_array_6, beta_array_7])
    time_array = time_array_1

    #preparing for export
    file_probability = str(dimension) + '_non_adiab_probability.npy'
    file_time_array = str(dimension) + '_non_adiab_time_array.npy'
    file_beta_array = str(dimension) + '_non_adiab_beta_array.npy'

    np.save(file_probability, probability)
    np.save(file_time_array, time_array)
    np.save(file_beta_array, beta_array)

    return print('Success: N=',dimension,' in ',int(toc/10),'min')

if __name__ == '__main__':

    step_function = 1
    rtolerance = 1e-6
    atolerance = 1e-6

    dimension = int(sys.argv[1])
    parallel_routine(1,80)
