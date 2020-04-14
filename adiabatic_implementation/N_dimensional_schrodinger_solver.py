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

def evaluate_probability(x, method):

    #Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1/np.sqrt(dimension))

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #define time-evolution
    psi_t = solve_schrodinger_equation(x[1], x[0], method)

    #check psi_t normalization
    normalization = np.dot(np.conj(psi_t), psi_t)

    probability = np.dot(oracle_site_state.transpose(), psi_t/(np.sqrt(normalization)))
    #print(np.abs(probability))

    #return 'crossing' probability
    #if(-np.abs(probability)**2 < -1):
        #print('Error: probability out of bounds: ', -np.abs(probability)**2)
    return -np.abs(probability)**2

#print(computation_time)
#print(result.x)
#print(-result.fun)

#MINIMIZATION USING SINGLE ITERATIONS OPTIMIZE.MINIMIZE AND OPTIMIZE.BASINHOPPING
#using different methods.
#

par_bnds = ([0, 2], [0, 20])
x0 = np.array([1.,10])
solver_method = 'RK45'
minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=solver_method)



computation_results = np.empty([2,5])
computation_results.fill(0)

#BASINHOPPING

#DIM 3
dimension = 5
par_bnds = ([0, 5], [0,25])
minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=solver_method)

tic = time.perf_counter()
minimization = basinhopping(evaluate_probability, x0,  minimizer_kwargs=minimizer_kwargs,niter=100)
computation_results[0, 4] = time.perf_counter() - tic
computation_results[0, 0] = dimension
computation_results[0, 1] = -minimization.fun
computation_results[0, 2] = minimization.x[1]
computation_results[0, 3] = minimization.x[0]
print("Dimension ",dimension," done in ", int(computation_results[0,4]/60), " minuti")

#DIM 5
dimension = 29
par_bnds = ([0, 4], [0,350])
minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=solver_method)

tic = time.perf_counter()
minimization = basinhopping(evaluate_probability, x0,  minimizer_kwargs=minimizer_kwargs,niter=25)
computation_results[1, 4] = time.perf_counter() - tic
computation_results[1, 0] = dimension
computation_results[1, 1] = -minimization.fun
computation_results[1, 2] = minimization.x[1]
computation_results[1, 3] = minimization.x[0]
print("Dimension ",dimension," done in ", int(computation_results[1,4]/60), " minuti")



#OUTPUT
computation_results = np.around(computation_results, decimals=3)
print(computation_results)
np.savetxt('Adiabatic_Optimization_5_29.txt', computation_results, fmt='%.3e')

#PERFORM BENCHMARK FOR 'BDF' AND 'RK45' METHOD
#FOR SCHRODINGER SOLVER
"""
benchmark = np.empty([14, 2])
benchmark.fill(0)
dimension_array = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
bnds = ([0, 4], [5, 20])
x = np.array([2, 10])
method = 'BDF'

for i in range(14):
    #set current system dimension
    dimension = dimension_array[i]

    #first method
    method = 'BDF'
    tic = time.perf_counter()
    psi_t = solve_schrodinger_equation(10, 2, method)
    benchmark[i,0] = time.perf_counter()-tic

    #second method
    method = 'RK45'
    tic = time.perf_counter()
    psi_t = solve_schrodinger_equation(10, 2, method)
    benchmark[i,1] = time.perf_counter()-tic

    #print results for quick debugging
    print(dimension, benchmark[i,0], benchmark[i,1])
"""
