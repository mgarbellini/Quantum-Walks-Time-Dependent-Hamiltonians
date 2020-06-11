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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import ray

ray.init()

#useful global variables, shouldn't be too inefficient
global dimension
global step_function
global rtolerance, atolerance #error tolerance (relative and absolute) for RK45 intergrator

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
            if (i == j):
                if (i == 0 & j == 0):
                    adj_matrix[i,dimension-1] = 1
                    adj_matrix[i,j+1] = 1
                elif (i == (dimension-1) & j == (dimension-1)):
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


#manipolation of other local maxima found
def arrange_local_maxima(results):

    local_maxima = np.empty([len(results.funl), 3])

    for i in range(len(results.funl)):
        local_maxima[i][0] = -results.funl[i]
        local_maxima[i][1] = results.xl[i][0]
        local_maxima[i][2] = results.xl[i][1]

    return local_maxima

#routine to generate loop hamiltonian + oracle state
def generate_hamiltonian_derivative(s, beta, derivative):

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
    if(derivative==0):
        if(step_function == 1):
            hamiltonian = (1-s)*laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -s*beta

        elif(step_function == 2):
            hamiltonian = (1-np.sqrt(s))*laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -np.sqrt(s)*beta

        elif(step_function == 3):
            hamiltonian = (1-np.cbrt(s))*laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -np.cbrt(s)*beta
        else:
            print('NameError: step_function not defined')

    elif(derivative==1):
        if(step_function == 1):
            hamiltonian = -laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -beta

        elif(step_function == 2):
            hamiltonian = -(1/(2*np.sqrt(s)))*laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -(1/(2*np.sqrt(s)))*beta

        elif(step_function == 3):
            hamiltonian = -(1/(3*np.cbrt(s)))*laplacian
            hamiltonian[int((dimension-1)/2),int((dimension-1)/2)] += -(1/(3*np.cbrt(s)))*beta
        else:
            print('NameError: step_function not defined')
    else:
        print('NameError: derivative flag unknown')

    return hamiltonian

def heatmap2d(arr: np.ndarray, time, beta, non_prob, non_prob_2, non_time, adiabatic_check):

    time_array = time
    beta_array = beta
    #for i in range(len(time_array)):
    #    time_array[i] = round((time_array[i]),1)
    #for i in range(len(beta_array)):
    #    beta_array[i] = round(beta_array[i],2)

    plt.imshow(arr, cmap='inferno_r', aspect= 1., origin= {'lower'})
    #plt.xticks(np.linspace(0, 40, 30, dtype=int), rotation='vertical')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Adiabatic Probability N=' + str(dimension) + '\nNon-Adiabatic (dashed): p = ' + str(non_prob) + ', T = ' + str(non_time)
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    #plt.suptitle(title, fontweight="bold", ha='center')
    plt.colorbar()
    levels = [0.9, 0.95, 0.99]
    non_adiabatic_levels = [non_prob, non_prob_2]
    ct = plt.contour(arr,levels, colors='white')
    cta = plt.contour(arr,non_adiabatic_levels, colors ='white', linestyles = 'dashed')
    plt.clabel(ct)

    #non physical results
    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            if(adiabatic_check[j][i] == 0):
                plt.gca().add_patch(Rectangle((-0.5+i, -0.5+j), 1, 1, fill=False, color = 'white', linewidth=0, hatch = '///////'))

    file_name = 'test.pdf'
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
    probability = np.empty([len(beta_array), int(len(time_array)/2)])



    for i in range(int(len(time_array)/2)):
        for j in range(len(beta_array)):
            #evaluate probability
            probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]], oracle_site_state)

    return probability, time_array, beta_array

def parallel_routine(time_array, beta_array):

    tic = time.perf_counter()

    #beta arrays

    beta_1 = [0.1, 0.2, 0.3]
    beta_2 = [0.4, 0.5, 0.6]
    beta_3 = [0.7, 0.8, 0.9]
    beta_4 = [1.0, 1.1, 1.2]
    beta_5 = [1.3, 1.4, 1.5]
    beta_6 = [1.6, 1.7, 1.8]
    beta_7 = [1.9, 2.0, 2.1]



    #parallel processes
    process_1 = grid_eval.remote(time_array, beta_1)
    process_2 = grid_eval.remote(time_array, beta_2)
    process_3 = grid_eval.remote(time_array, beta_3)
    process_4 = grid_eval.remote(time_array, beta_4)
    process_5 = grid_eval.remote(time_array, beta_5)
    process_6 = grid_eval.remote(time_array, beta_6)
    process_7 = grid_eval.remote(time_array, beta_7)

    #reassigning values to arrays
    probability_1, time_array_1, beta_array_1 = ray.get(process_1)
    probability_2, time_array_2, beta_array_2 = ray.get(process_2)
    probability_3, time_array_3, beta_array_3 = ray.get(process_3)
    probability_4, time_array_4, beta_array_4 = ray.get(process_4)
    probability_5, time_array_5, beta_array_5 = ray.get(process_5)
    probability_6, time_array_6, beta_array_6 = ray.get(process_6)
    probability_7, time_array_7, beta_array_7 = ray.get(process_7)

    #concatenate arrays to output array
    toc = time.perf_counter() - tic

    probability = np.concatenate([probability_1, probability_2, probability_3, probability_4, probability_5, probability_6, probability_7], axis=0)

    #preparing for export
    if(step_function == 1):
        file_probability = str(dimension) + '_probability.npy'
    elif(step_function == 2):
        file_probability = str(dimension) + '_probability_sqrt.npy'
    elif(step_function == 3):
        file_probability = str(dimension) + '_probability_cbrt.npy'
    else:
        print("Error: step function not defined")




    np.save(file_probability, probability)

    return print('Success: N=',dimension,' in ',int(toc/60),'min')

if __name__ == '__main__':

    rtolerance = 1e-6
    atolerance = 1e-6



    dimension = int(sys.argv[1])
    step_function = int(sys.argv[2])
    beta_array = np.load(str(dimension)+ "_beta_array.npy")
    time_array = np.load(str(dimension) + "_time_array.npy")

    parallel_routine(time_array, beta_array)
