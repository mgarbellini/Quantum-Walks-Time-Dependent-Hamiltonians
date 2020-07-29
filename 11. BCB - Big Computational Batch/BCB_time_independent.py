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

def heatmap2d(probability, time, beta):

    time_array = time
    beta_array = beta
    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]), 1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i], 2)

    plt.imshow(probability, cmap='inferno_r', aspect=1., origin={'lower'})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Time-Independent Probability N=' + str(dimension)
    plt.title(title,  y=1.04, fontweight="bold",  ha='center')
    plt.colorbar()


    file_name = str(dimension) + '_cg_heatmap_static.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()


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
    time_text = str(dimension) + "_circ_time.npy"
    beta_text = str(dimension) + "_circ_beta.npy"
    beta = np.load(beta_text)
    time_array = np.load(time_text)
    # initialize ray multiprocessing
    ray.init()
    tic = time.perf_counter()

    # useful definitions for multicore computation
    cpu_count = 4
    sampling_per_cpu_count = 6
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

    # preparing for export and export and miscellanea
    npy = ".npy"

    file_probability = str(dimension) + '_circ_prob_static.npy'

    # export heatmap plot
    heatmap2d(probability_array, time_array, beta)

    np.save(file_probability, probability_array)
    toc = time.perf_counter() - tic

    print('Success: N ', dimension, ' in ', int(toc / 60), ' min')
    return 0

if __name__ == '__main__':

    step_function = 1
    rtolerance = 1e-6
    atolerance = 1e-6

    min_dim = 33
    max_dim = 51
    dims = np.arange(min_dim, max_dim + 1, 2)
    for dim in dims:
        dimension = dim
        parallel_routine()
