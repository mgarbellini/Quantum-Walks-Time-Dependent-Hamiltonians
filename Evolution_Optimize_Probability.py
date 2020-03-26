# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping
#import constrNMPy as cNM
#from scipy.optimize import shgo

import matplotlib.pyplot as plt

# # # # # # # # # # # # # # #
# FUNCTIONS IMPLEMENTATION  #
# # # # # # # # # # # # # # #

#global definition
dimension = int(sys.argv[1])

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

# # # # # # #
#   MAIN    #
# # # # # # #

if dimension%2 == 0 :
    exit('Error: even number of sites. Expected odd number!')

#Define lambda and time bounds
bnds = ([0, 3], [0, 10])
x = np.array([0.1, 1])
minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
result = basinhopping(evaluate_probability, x,  minimizer_kwargs=minimizer_kwargs,niter=1000)
print(dimension)
print(result.x)
print(-result.fun)
