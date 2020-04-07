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

dimension = 5

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
    time_scalar_function = 1 - float(time)/TIME
    time_hamiltonian = (time_scalar_function)*hamiltonian

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

def solve_schrodinger_equation(time, beta):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1/np.sqrt(dimension)+0*1j)
    t_i = 0.
    t_step_max = 0.01
    t_f = time
    psoln_solve_ivp = solve_ivp(schrodinger_equation, [t_i, t_f], y0, method='RK45', max_step=t_step_max, args=(beta,time))

    psi_t = np.empty(dimension,dtype=complex)
    for i in range(dimension):
        psi_t[i] = psoln_solve_ivp.y[i, len(psoln_solve_ivp.y[i])-1]

    return psi_t

def evaluate_probability(x):

    #Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1/np.sqrt(dimension))

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #define time-evolution
    psi_t = solve_schrodinger_equation(x[1], x[0])
    probability = np.dot(oracle_site_state.transpose(), psi_t)

    #return 'crossing' probability
    return -np.abs(probability)**2

#main solver
#x=[9, 3]
#probability = evaluate_probability(x)
#print(probability)


#Define lambda and time bounds
bnds = ([2, 4], [5, 15])
x = np.array([3, 11])
minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
tic = time.perf_counter()
result = basinhopping(evaluate_probability, x,  minimizer_kwargs=minimizer_kwargs,niter=10)
computation_time = time.perf_counter() - tic
print("Time for one computation of basinhopping optimization:")
print(computation_time)
print("Probability, beta and time results:")
print(result.x)
print(-result.fun)
