# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping, shgo
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

#useful global variables, shouldn't be too inefficient
global dimension
global step_function
global optimization_method  #SHGO or BH (basinhopping)

#routine to generate loop hamiltonian + oracle state
def generate_hamiltonian(dimension, beta, time, T):

    #generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
    diag_matrix.fill(0)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                diag_matrix[i,j] = 2

    #generate loop adjacency matrix
    adj_matrix = np.empty([dimension, dimension])
    adj_matrix.fill(0)
    for i in range(dimension):
        for j in range(dimension):
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
    laplacian = diag_matrix - adj_matrix

    #generate time-stepping function g_T(t): let's consider three cases, ^1, ^1/2, 1^1/3
    #Note: if t=0 the function is automatically set to 'almost zero' (0.000001). This prevents warning within the ODE solver
    if(time==0):
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

#schroedinger equation solver. returns psi(t)
def solve_schrodinger_equation(time, beta):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1/(np.sqrt(dimension)))

    sh_solved = solve_ivp(schrodinger_equation, [0., time], y0, method='RK45', args=(beta,time))
    #for more precise results use method RK45 and max_step=t_step_max
    #for less precise results but faster computation use 'BDF'
    psi_t = np.empty(dimension,dtype=complex)
    for i in range(dimension):
        psi_t[i] = sh_solved.y[i, len(sh_solved.y[i])-1]

    return psi_t

#routine to evaluate probability |<w|psi(t)>|^2
def evaluate_probability(x, oracle_site_state):

    #define time-evolution
    psi_t = solve_schrodinger_equation(x[1], x[0])

    #psi_t normalization
    normalization = np.dot(np.conj(psi_t), psi_t)

    #probability evaluation
    probability = np.dot(oracle_site_state.transpose(), psi_t/(np.sqrt(normalization)))
    if(np.abs(probability)**2 > 1):
        print('Error: probability out of bounds: ', np.abs(probability)**2)


    return -np.abs(probability)**2




#parameters
par_bnds = [(0, 4), (1, 40)]
BH_iter = 50
dimension = 7
optimization_method = 'BH'
step_function = 1

#define oracle_site_state
oracle_site_state = np.empty([dimension, 1])
oracle_site_state.fill(0)
oracle_site_state[int((dimension-1)/2)][0] = 1

#Optimization methods. This prevents commenting of unused code snippets
if(optimization_method == 'SHGO'):

    #count time
    tic = time.perf_counter()

    #maximize probability
    maximized = shgo(evaluate_probability, par_bnds,n=100, iters=1, args=(oracle_site_state,),sampling_method='sobol')

    #computation time in minutes (rounded)
    comp_time = int((time.perf_counter() - tic)/60)

    #store results
    comp_results = [dimension, -maximized.fun, maximized.x[0], maximized.x[1], comp_time]

    #print computational comp_results
    print(comp_results)

elif(optimization_method == 'BH'):

        #initial values for minimization
        x0 = np.array([1.,10])

        #BH arguments for minimization
        #"L-BFGS-B" is the chosen minimization methods
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=oracle_site_state)

        #count time
        tic = time.perf_counter()

        #maximize probability
        maximized = basinhopping(evaluate_probability, x0,  minimizer_kwargs=minimizer_kwargs,niter=BH_iter)

        #computation time in minutes (rounded)
        comp_time = int(time.perf_counter() - tic)/60

        #store results
        comp_results = [dimension, -maximized.fun, maximized.x[0], maximized.x[1], comp_time]

        #print computational comp_results
        print(comp_results)
else:
        print("Error: minimization methods wrongly specified")

#np.savetxt('Adiabatic_Optimization_5.txt', computation_results, fmt='%.3e')
