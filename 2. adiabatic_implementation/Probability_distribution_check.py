# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

global dimension

def generate_hamiltonian(dimension, beta, time, TIME):

    #generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
    diag_matrix.fill(0)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                diag_matrix[i,j] = 2
            else:
                diag_matrix[i,j] = 0

    #generate loop adjacency matrix
    adj_matrix = np.empty([dimension, dimension])
    adj_matrix.fill(0)
    for i in range(dimension): #colonna
        for j in range(dimension): #riga
            if i == j:
                if i == 0 & j == 0:
                    adj_matrix[i,dimension-1] = 1
                    adj_matrix[i,j+1] = 1
                elif i == dimension-1 & j == dimension-1:
                    adj_matrix[i,j-1] = 1
                    adj_matrix[i,0] = 1
                else:
                    adj_matrix[i,j-1] = 1
                    adj_matrix[i,j+1] = 1

    #generate laplacian of loop
    hamiltonian = diag_matrix - adj_matrix



    #generate time dependet hamiltonian
    time_hamiltonian = (1 - float(time)/TIME)*hamiltonian

    #generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    index = int((dimension-1)/2)
    time_hamiltonian[index,index] += - (float(time)/TIME)*beta

    return time_hamiltonian

def schrodinger_equation(t, y, beta, TIME):

    H = generate_hamiltonian(dimension, beta, t, TIME)
    derivs = []
    psi = 0
    for i in range(dimension):
        for j in range(dimension):
            psi += H[i,j]*y[j]
        derivs.append(-1j*psi)
        psi = 0

    return derivs

def solve_schrodinger_equation(time, beta, meth):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1/(np.sqrt(dimension)))
    t_i = 0.
    t_step_max = 0.01
    t_f = time
    psoln_solve_ivp = solve_ivp(schrodinger_equation, [t_i, t_f], y0, method=meth, args=(beta,time))
    #for more precise results use method RK45 and max_step=t_step_max
    #for less precise results but faster computation use 'BDF'
    psi_t = np.empty(dimension,dtype=complex)
    for i in range(dimension):
        psi_t[i] = psoln_solve_ivp.y[i, len(psoln_solve_ivp.y[i])-1]

    return psi_t

def evaluate_probability(time, beta, method):

    #Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1/np.sqrt(dimension))

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #define time-evolution
    psi_t = solve_schrodinger_equation(time, beta, method)

    #check psi_t normalization
    normalization = np.dot(np.conj(psi_t), psi_t)

    probability = np.dot(oracle_site_state.transpose(), psi_t/(np.sqrt(normalization)))
    #print(np.abs(probability))

    #return 'crossing' probability
    #if(-np.abs(probability)**2 < -1):
        #print('Error: probability out of bounds: ', -np.abs(probability)**2)
    return np.abs(probability)**2


solver_method = 'RK45'
dimension = 27
points = 200
probability_distribution = np.empty([points, 9])
beta_points = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
time_points = np.linspace(0.1, 350, num=points, endpoint=False)
index = 0;

for i in time_points:
    probability_distribution[index, 0]  = i
    for j in range(8):
        probability_distribution[index, j+1] = evaluate_probability(i, beta_points[j], solver_method)
    index += 1

    #progress check
    prob = int((float (index)/points)*100)
    print('Progress: ', prob, '%')

np.savetxt('27_prob.txt', probability_distribution, fmt='%.3e')



#end
