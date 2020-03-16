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
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # #
# FUNCTIONS IMPLEMENTATION  #
# # # # # # # # # # # # # # #

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

    problem_hamiltonian[oracle_site][oracle_site] = problem_hamiltonian[oracle_site][oracle_site] + gamma
    return problem_hamiltonian


#Implementation of time evolution with time-independent hamiltonian
#and evalute 'crossing-to-oracle' probability for a given time t
def evaluate_probability(hamiltonian, time, psi_0, oracle_site):

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

#Implements (numerical) probability distribution given time-range and time-step
#Input: problem_hamiltonian, flat_state, oracle_site
#Ouput: array of probability for FIXED gamma value
def probability_distribution_time(laplacian, flat_state, oracle_site, time_range, time_step, gamma):

    #generate problem hamiltonian from laplacian
    problem_hamiltonian = generate_problem_hamiltonian(laplacian, oracle_site, -gamma)

    #defining time steps
    step = float (time_range) / time_step
    probability_distribution = np.empty([time_step+1, 1])

    #perform probability calculation
    for i in range(time_step+1):
        probability_distribution[i][0] = evaluate_probability(problem_hamiltonian, i*step, flat_state, oracle_site)
        #probability_distribution[i][1] = i*step
    return probability_distribution


#Implements (numerical) probability distribution given time-range and time-step
#Input: problem_hamiltonian, flat_state, oracle_site, time, gamma_range and gamma_steps
#Ouput: array of probability for FIXED time value
def probability_distribution_gamma(laplacian, flat_state, oracle_site, time, gamma_range, gamma_steps):

    #defining gamma steps
    step = float(gamma_range) / gamma_steps
    probability_distribution = np.empty([gamma_steps+1, 2])

    #perform probability calculation
    for i in range(gamma_steps+1):
        probability_distribution[i][0] = evaluate_probability(generate_problem_hamiltonian(laplacian, oracle_site,
                                                                    -gamma_steps*i), time, flat_state, oracle_site )
        probability_distribution[i][1] = step * i
    return probability_distribution

# # # # # # #
#   MAIN    #
# # # # # # #

#Define number of graph sites.
#This also represents the space dimension (odd number expected)
dimension = 3
oracle_site = (dimension-1)/2
if dimension%2 == 0 :
    exit('Error: even number of sites. Expected odd number!')

#Generate so called 'flat-state'
flat_state = np.empty([dimension, 1])
flat_state.fill(1/np.sqrt(dimension))


#Generate Hamiltonian of Loop graph
laplacian = generate_loop_hamiltonian(dimension)
print(laplacian)
problem_hamiltonian1 = generate_problem_hamiltonian(laplacian, oracle_site, 5)
print(problem_hamiltonian1)
problem_hamiltonian2 = generate_problem_hamiltonian(laplacian, oracle_site, 10)
print(problem_hamiltonian2)


#print(problem_hamiltonian1)
#print(problem_hamiltonian2)


test1 = evaluate_probability(problem_hamiltonian1, 5.5, flat_state, oracle_site)
test2 = evaluate_probability(problem_hamiltonian2, 5.5, flat_state, oracle_site)
print (test1)
print (test2)
#Probability Distribution with variable time and variable gamma
#gamma = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]
#for i in gamma:
    #print (evaluate_probability(problem_hamiltonian, i, flat_state, oracle_site))

#test1 = probability_distribution_time(laplacian, flat_state, oracle_site, 1, 100, 6 )
#test2 = probability_distribution_time(laplacian, flat_state, oracle_site, 1, 100, 6 )
#test = test1-test2
