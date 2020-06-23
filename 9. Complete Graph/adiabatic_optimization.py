#!/usr/bin/env python
# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it




import sys
import time
import numpy as np
from scipy import linalg
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, basinhopping, shgo, dual_annealing
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import ray



#useful global variables, shouldn't be too inefficient
global dimension
global step_function
global rtolerance, atolerance #error tolerance (relative and absolute) for RK45 intergrator

#routine to generate loop hamiltonian + oracle state
def generate_hamiltonian(dimension, beta, time, T):

    #generate laplacian
    laplacian = np.empty([dimension, dimension])
    laplacian.fill(-1)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                laplacian[i,j] += dimension

    #generate time-stepping function g_T(t): let's consider three cases, ^1, ^1/2, 1^1/3
    #Note: if t=0 the function is automatically set to 'almost zero' (0.000001). This prevents warning within the ODE solver
    if(time==0 or T ==0):
        hamiltonian = laplacian
    else:
        if(step_function==1):
            g_T = float(time)/T
        elif(step_function==2):
            g_T = np.sqrt(float(time)/T)
        elif(step_function==3):
            g_T = np.cbrt(float(time)/T)
        else:
            print("Error: step_function value not defined")

        #generate time dependet hamiltonian
        hamiltonian = (1 - g_T)*laplacian

        #generate problem_hamiltonian (i.e. adding oracle to central site)
        hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += - g_T*beta

    return hamiltonian

#routine to implement schroedinger equation. returns d/dt(psi)
#for the ODE solver
def schrodinger_equation(t, y, beta, T):

    H = generate_hamiltonian(dimension, beta, t, T)
    derivs = []
    psi = 0
    for i in range(dimension):
        for j in range(dimension):
            psi += H[i,j]*y[j]
        derivs.append(-1j*psi)
        psi = 0

    return derivs
#manipolation of other local maxima found
def arrange_local_maxima(results):

    local_maxima = np.empty([len(results.funl), 3])

    for i in range(len(results.funl)):
        local_maxima[i][0] = -results.funl[i]
        local_maxima[i][1] = results.xl[i][0]
        local_maxima[i][2] = results.xl[i][1]

    return local_maxima

#schroedinger equation solver. returns psi(t)
def solve_schrodinger_equation(time, beta):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1/(np.sqrt(dimension)))

    sh_solved = solve_ivp(schrodinger_equation, [0., time], y0, method='RK45',atol = atolerance, rtol =rtolerance, args=(beta,time))
    #for more precise results use method RK45 and max_step=t_step_max
    #for less precise results but faster computation use 'BDF'
    psi_t = np.empty(dimension,dtype=complex)
    for i in range(dimension):
        psi_t[i] = sh_solved.y[i, len(sh_solved.y[i])-1]

    normalization = np.dot(np.conj(psi_t), psi_t)
    #print('Normalization:',normalization)
    return psi_t
    #return normalization.real

#routine to evaluate probability |<w|psi(t)>|^2
def evaluate_probability(x, oracle_site_state):

    #define time-evolution
    psi_t = solve_schrodinger_equation(x[1], x[0])

    #psi_t normalization
    #normalization should be 1. Values that differ from 1 are due to (expected) errors in the
    #integration. With max_step_size = 0.5 we can acheive error in the order of <10^-4
    normalization = np.dot(np.conj(psi_t), psi_t)


    #probability evaluation
    probability = np.dot(oracle_site_state.transpose(), psi_t/(np.sqrt(normalization)))
    if(np.abs(probability)**2 > 1):
        print('Error: probability out of bounds: ', np.abs(probability)**2)
    else:
        return -np.abs(probability)**2

def optimize_probability(par_bnds):

    #beginning of computation time flag
    tic = time.perf_counter()

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #maximize probability
    #sampling_method must be chosen between 'simplicial' and 'sobol'
    maximized = shgo(evaluate_probability, par_bnds, n=40, iters=1, args=(oracle_site_state,),sampling_method='sobol')

    #computation time in minutes (rounded)
    comp_time = int(time.perf_counter() - tic)/60

    #print other local maxima found
    #set 0 for probability sorting, 1 for time sorting
    local_results = arrange_local_maxima(maximized)

    #store results
    comp_results = [dimension, -maximized.fun, maximized.x[0], maximized.x[1], comp_time]
    print(dimension,-maximized.fun, maximized.x[0], maximized.x[1], int(comp_time))

    #return computational comp_results

    return 0



if __name__ == '__main__':

    step_function = 2
    rtolerance = 1e-6
    atolerance = 1e-6

    dimension = 15
    par_bnds = [[5,20], [0, dimension]]
    optimize_probability(par_bnds)
