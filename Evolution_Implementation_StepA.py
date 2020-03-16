# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it


#ADIABATIC QUANTUM WALKS
#STEP A: Time evolution of circle graph witn n nodes


#NOTA:
#array[colonna][riga]

import numpy as np
from scipy import linalg

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS IMPLEMENTATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#Implementation of generate_hamiltonian function
#this routine takes hilbert space dimension (n) as input
#and outputs a nxn matrix corrisponding to H = L + gamma
def generate_loop_hamiltonian(dimension):

    #generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
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


#Implementation of problem hamiltonian. Takes oracle site and gamma value at input
def generate_problem_hamiltonian(problem_hamiltonian, oracle_site, gamma):

    problem_hamiltonian[oracle_site][oracle_site] += gamma
    return problem_hamiltonian


#Implementation of time evolution with time-independent hamiltonian
#and evalute 'crossing-to-oracle' probability for a given time t
def evalute_probability(hamiltonian, time, psi_0, oracle_site):

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[oracle_site][0] = 1

    #define time-evolution
    unitary_time_evolution = linalg.expm(-(1j)*hamiltonian*time)
    psi_t = np.dot(unitary_time_evolution, psi_0)
    probability = np.dot(oracle_site_state.transpose(), psi_t)

    #return 'crossing' probability
    return np.abs(probability)**2


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dimension = 3

#Generate so called 'flat-state'
flat_state = np.empty([dimension, 1])
flat_state.fill(1)


laplacian = generate_loop_hamiltonian(dimension)
problem_hamiltonian = generate_problem_hamiltonian(laplacian, 1, 11)

probability = evalute_probability(problem_hamiltonian, 1, flat_state, 1)
print(probability)
