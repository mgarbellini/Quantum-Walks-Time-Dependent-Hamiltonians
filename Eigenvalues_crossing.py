# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping
#from scipy.optimize import shgo

import matplotlib.pyplot as plt

# # # # # # # # # # # # # # #
# FUNCTIONS IMPLEMENTATION  #
# # # # # # # # # # # # # # #

def generate_loop_hamiltonian(dimension):

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

    return hamiltonian

def generate_oracle_hamiltonian(dimension):

    hamiltonian = np.empty([dimension, dimension])
    hamiltonian.fill(0)
    #generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    hamiltonian[(dimension-1)/2][(dimension-1)/2] =  1

    return hamiltonian

def time_dependent_hamiltonian(dimension, gamma, time, T):

    g = float (time)/T
    hamiltonian = (1-g)*generate_loop_hamiltonian(dimension) - g*gamma*generate_oracle_hamiltonian(dimension)

    return hamiltonian

# # # # # # #
#   MAIN    #
# # # # # # #

dimension = 5
gamma = 1.6
T = 10

hamiltonian = generate_loop_hamiltonian(dimension) - gamma*generate_oracle_hamiltonian(dimension)
t_hamiltonian = time_dependent_hamiltonian(dimension, gamma,3, T);
eigenvalues, eigenvectors = linalg.eig(t_hamiltonian)

eigenvalues_sorted = np.sort(eigenvalues, axis=None)
print(eigenvalues_sorted)
