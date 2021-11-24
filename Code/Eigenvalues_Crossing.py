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

def compute_eigenvalues(dimension, gamma, time, TIME):

    t_hamiltonian = time_dependent_hamiltonian(dimension, gamma, time, TIME)
    eigenvalues, eigenvectors = linalg.eig(t_hamiltonian)
    eigenvalues_sorted = np.sort(eigenvalues, axis=None)
    return eigenvalues_sorted.real


# # # # # # #
#   MAIN    #
# # # # # # #



"""
hamiltonian = generate_loop_hamiltonian(dimension) - gamma*generate_oracle_hamiltonian(dimension)
t_hamiltonian = time_dependent_hamiltonian(dimension, gamma,3, T);
eigenvalues, eigenvectors = linalg.eig(t_hamiltonian)
eigenvalues_sorted = np.sort(eigenvalues, axis=None)
"""

#eigenvalues = compute_eigenvalues(dimension, gamma, 5, T)
#eigenvalues1 = compute_eigenvalues(dimension, gamma, 5, T)

#np.insert(eigenvalues_sorted, [0], [1])

dimension = 15
gamma = 1.02212
T = 10.

#Eigenvalues plot
sampling = 100
time_step = float (T*T)/100
time = 0
eigenvalues = np.empty([sampling, dimension])
eigenvalues_distribution = np.empty([sampling, dimension])
for i in range(sampling):
    time += time_step
    eigenvalues[i] = compute_eigenvalues(dimension, gamma, time, T*T)

time = 0
for i in range(100):
    time += time_step
    eigenvalues_distribution = np.insert(eigenvalues,[i, 0], [7])

#print(eigenvalues)
np.savetxt('15_quadratic.txt', eigenvalues, fmt='%.3e')
