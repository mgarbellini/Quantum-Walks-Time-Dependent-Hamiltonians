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
        elif(step_function==0):
            nn = np.sqrt(-1+dimension)
            g_T = (nn*nn + nn*np.tan(2*nn*time*(1/dimension) - np.arctan(nn))*(1/(2*nn*nn)))
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
    if(np.abs(probability)**2 > 1.1):
        print('Error: probability out of bounds: ', np.abs(probability)**2)
    else:
        return -np.abs(probability)**2

def heatmap2d(probability, time, beta):

    time_array = time
    beta_array = beta
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i],2)

    plt.imshow(probability, cmap='inferno_r', aspect= 1., origin= {'lower'})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Dimension', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Dynamic Prob Complete Graph forcing T = sqrt(N) '
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    plt.colorbar()


    pdf = '.pdf'
    file_name = 'forced_T'+ pdf
    plt.savefig(file_name)
    plt.clf()
    plt.close()

@ray.remote
def grid_eval(time_array, beta_array):

    #Define oracle site state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #Define time, beta and probability and adiabatic_check array
    probability = np.empty([len(beta_array), len(time_array)])
    adiabatic_check  = np.empty([len(beta_array), len(time_array)])

    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]], oracle_site_state)

    return probability

def single_evaluation(beta):
    #Define oracle site state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    t = np.pi*0.5*np.sqrt(dimension)

    probability = -evaluate_probability([beta, t], oracle_site_state)

    return probability

def parallel_routine(lb_time, ub_time,lb_beta, ub_beta):

    #initialize ray multiprocessing
    ray.init()
    tic = time.perf_counter()


    #useful definitions for multicore computation
    cpu_count = 4
    time_sampling_points = 30
    sampling_per_cpu_count = 6
    beta_sampling_points = cpu_count*sampling_per_cpu_count
    beta = np.linspace(lb_beta, ub_beta, beta_sampling_points)
    time_array = np.linspace(lb_time, ub_time, time_sampling_points)
    process = []
    probability = []

    #parallel processes
    for i in range(cpu_count):
        process.append(grid_eval.remote(time_array, beta[int(sampling_per_cpu_count*i):int(sampling_per_cpu_count*(i+1))]))

    #reassigning values to arrays
    for i in range(cpu_count):
        probability.append(ray.get(process[i]))

    #concatenate arrays to output array
    probability_array = np.concatenate([probability[0], probability[1]], axis = 0)
    for i in range(cpu_count - 2):
        probability_array = np.concatenate([probability_array, probability[i+2]])

    #shutting down ray
    ray.shutdown()

    #preparing for export and export and miscellanea
    step_function_text = ["lin", "sqrt", "cbrt"]
    npy = ".npy"

    file_probability = str(dimension) + '_cg_probability_' + step_function_text[step_function-1]
    file_time = str(dimension) + '_cg_time_array_.npy'
    file_beta = str(dimension) + '_cg_beta_array_.npy'

    #export heatmap plot
    heatmap2d(probability_array, time_array, beta, step_function_text)

    np.save(file_probability, probability_array)
    np.save(file_time, time_array)
    np.save(file_beta, beta)
    toc = time.perf_counter() - tic


    print('Success: N ',dimension,' in ',int(toc/60),' min')
    return 0


if __name__ == '__main__':

    step_function = 1
    rtolerance = 1e-6
    atolerance = 1e-6

    dim = []
    for i in range(62):
        if (i%2 != 0):
            if(i != 1):
                dim.append(i)
    beta_array = np.linspace(6, 30, 10)
    probability = np.empty([len(beta_array), len(dim)])

    for i in range(len(dim)):
        tic = time.perf_counter()
        for j in range(len(beta_array)):
            dimension = dim[i]
            probability[j][i] = single_evaluation(beta_array[j])
        print(dimension, int((time.perf_counter()-tic)/60))

    np.save('probability.npy', probability)
    np.save('dimension.npy', dim)
    np.save('beta.npy', beta_array)
