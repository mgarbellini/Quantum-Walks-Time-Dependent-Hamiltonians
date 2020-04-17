# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping, shgo, dual_annealing
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

#useful global variables, shouldn't be too inefficient
global dimension
global step_function

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

    else:
        return -np.abs(probability)**2

#define callback functions. This allows to set precision to probability evaluation,
#e.g. stop process at 0.99 instead of 0.99875
def optimization_precision(x, probability, context):

    if(probability <= -0.99):
        return True
    else:
        return False


#routine to maximize probability
def optimization(par_bnds,optimization_method):

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    if(optimization_method == 'SHGO'):

        #count time
        tic = time.perf_counter()

        #maximize probability
        #sampling_method must be chosen between 'simplicial' and 'sobol'
        maximized = shgo(evaluate_probability, par_bnds,n=100, iters=5, args=(oracle_site_state,),sampling_method='simplicial')

        #computation time in minutes (rounded)
        comp_time = int(int(time.perf_counter() - tic)/60)

        #store results
        comp_results = [dimension, -maximized.fun, maximized.x[0], maximized.x[1], comp_time]

        #return computational comp_results
        return comp_results

    elif(optimization_method == 'BH'):

            #set basinhopping niter
            if(dimension > 15):
                BH_iter = 25
            else:
                BH_iter = 50

            #test basinhopping iter set to 1
            BH_iter = 1
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
            comp_time = int(int(time.perf_counter() - tic)/60)

            #store results
            comp_results = [dimension, float(-maximized.fun), maximized.x[0], maximized.x[1], comp_time]

            #print computational comp_results
            return comp_results
    elif(optimization_method == 'DUAL'):
        #count time
        tic = time.perf_counter()

        #maximize probability
        #sampling_method must be chosen between 'simplicial' and 'sobol'
        maximized = dual_annealing(evaluate_probability, par_bnds, args=(oracle_site_state,), maxiter=2)

        #computation time in minutes (rounded)
        comp_time = int(int(time.perf_counter() - tic)/60)

        #store results
        comp_results = [dimension, float(-maximized.fun), maximized.x[0], maximized.x[1], comp_time]

        #return computational comp_results
        return comp_results

    else:
        print('Error: invalid optimization method')

# # # # # # #
#   MAIN    #
# # # # # # #


#parameters
par_bnds = [(0, 4), (30,120)]
dimension = 17

step_function = 1
results = optimization(par_bnds, 'SHGO')
print(results)
step_function = 2
results = optimization(par_bnds, 'SHGO')
print(results)
step_function = 3
results = optimization(par_bnds, 'SHGO')
print(results)



#np.savetxt('Adiabatic_Optimization_5.txt', computation_results, fmt='%.3e')
