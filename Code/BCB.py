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


# useful global variables, shouldn't be too inefficient
global dimension
global step_function
global topology
# error tolerance (relative and absolute) for RK45 intergrator
global rtolerance, atolerance


# routine to generate loop hamiltonian + oracle state
def generate_hamiltonian(dimension, beta, time, T):

    if(topology == "complete"):
        # generate laplacian
        laplacian = np.empty([dimension, dimension])
        laplacian.fill(-1)
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    laplacian[i, j] += dimension

    elif(topology== "circular"):
        # generate diagonal matrix
        diag_matrix = np.empty([dimension, dimension])
        diag_matrix.fill(0)
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    diag_matrix[i, j] = 2

        # generate loop adjacency matrix
        adj_matrix = np.empty([dimension, dimension])
        adj_matrix.fill(0)
        for i in range(dimension):
            for j in range(dimension):
                if (i == j):
                    if (i == 0 & j == 0):
                        adj_matrix[i, dimension - 1] = 1
                        adj_matrix[i, j + 1] = 1
                    elif (i == (dimension - 1) & j == (dimension - 1)):
                        adj_matrix[i, j - 1] = 1
                        adj_matrix[i, 0] = 1
                    else:
                        adj_matrix[i, j - 1] = 1
                        adj_matrix[i, j + 1] = 1

        # generate laplacian of loop
        laplacian = diag_matrix - adj_matrix
    else:
        print("Error: undefined topology")


    # generate time-stepping function g_T(t): let's consider three cases, ^1, ^1/2, 1^1/3
    # Note: if t=0 the function is automatically set to 'almost zero' (0.000001). This prevents warning within the ODE solver
    if(time == 0 or T == 0):
        hamiltonian = laplacian
    else:
        if(step_function == 'lin'):
            g_T = float(time) / T
        elif(step_function == 'sqrt'):
            g_T = np.sqrt(float(time) / T)
        elif(step_function == 'cbrt'):
            g_T = np.cbrt(float(time) / T)
        elif(step_function == 'cerf_3'):
            nn = np.sqrt(-1 + dimension)
            g_T = 0.5 * (1 + (2 * (float(time) / T) - 1)**3)
        elif(step_function == 'cerf_5'):
            g_T = 0.5 * (1 + (2 * (float(time) / T) - 1)**5)
        elif(step_function == 'cerf_7'):
            g_T = 0.5 * (1 + (2 * (float(time) / T) - 1)**7)

        else:
            print("Error: step_function value not defined")

        # generate time dependet hamiltonian
        hamiltonian = (1 - g_T) * laplacian

        # generate problem_hamiltonian (i.e. adding oracle to central site)
        hamiltonian[int((dimension - 1) / 2),
                    int((dimension - 1) / 2)] += - g_T * beta

    return hamiltonian

def generate_time_independent_hamiltonian(dimension, gamma):
    #generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
    diag_matrix.fill(0)
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

    #generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    index = int((dimension-1)/2)
    hamiltonian[index][index] += - gamma

    return hamiltonian

def evaluate_time_independent_probability(time, gamma, oracle_site_state):

    #generate_hamiltonian
    hamiltonian = generate_time_independent_hamiltonian(dimension, gamma)

    #Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1/np.sqrt(dimension))

    #set to zero variables
    probability = 0
    psi_t = np.empty([dimension, 1])
    psi_t.fill(0)

    #define time-evolution
    unitary_time_evolution = np.empty([dimension, dimension])
    unitary_time_evolution.fill(0)
    unitary_time_evolution = linalg.expm(-(1j)*hamiltonian*time)
    psi_t = np.dot(unitary_time_evolution, psi_0)
    probability = np.dot(oracle_site_state.transpose(), psi_t)

    #return 'crossing' probability
    return -np.abs(probability)**2

# routine to implement schroedinger equation. returns d/dt(psi)
# for the ODE solver
def schrodinger_equation(t, y, beta, T):

    H = generate_hamiltonian(dimension, beta, t, T)
    derivs = []
    psi = 0
    for i in range(dimension):
        for j in range(dimension):
            psi += H[i, j] * y[j]
        derivs.append(-1j * psi)
        psi = 0

    return derivs

# schroedinger equation solver. returns psi(t)
def solve_schrodinger_equation(time, beta):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1 / (np.sqrt(dimension)))

    sh_solved = solve_ivp(schrodinger_equation, [0., time], y0, method='RK45', atol=atolerance, rtol=rtolerance, args=(beta, time))
    # for more precise results use method RK45 and max_step=t_step_max
    # for less precise results but faster computation use 'BDF'
    psi_t = np.empty(dimension, dtype=complex)
    for i in range(dimension):
        psi_t[i] = sh_solved.y[i, len(sh_solved.y[i]) - 1]

    normalization = np.dot(np.conj(psi_t), psi_t)
    return psi_t*(1/np.sqrt(normalization))

# routine to evaluate probability |<w|psi(t)>|^2
def evaluate_probability(x, oracle_site_state):

    # find evolved state (already normalized)
    psi_t = solve_schrodinger_equation(x[1], x[0])

    # probability evaluation
    probability = np.dot(oracle_site_state.transpose(),psi_t)
    if(np.abs(probability)**2 > 1):
        print('Error: probability out of bounds: ', np.abs(probability)**2)
    else:
        return -np.abs(probability)**2

#routine for output heatmap plot
def heatmap2d(probability, time, beta, step_function):

    time_array = time
    beta_array = beta
    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]), 1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i], 2)

    plt.imshow(probability, cmap='inferno_r', aspect=1., origin={'lower'}, vmin=0, vmax=1)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Dynamic Prob Circular Graph N=' + \
        str(dimension) + ' (' + step_function + ')'
    plt.title(title,  y=1.04, fontweight="bold",  ha='center')
    plt.colorbar()

    levels = [0.9, 0.95, 0.99]
    ct = plt.contour(probability, levels, colors='white')
    plt.clabel(ct)


    file_name = str(dimension) + '_heatmap_' + \
        step_function + '.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()


@ray.remote
def grid_eval(time_array, beta_array):

    # Define oracle site state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension - 1) / 2)][0] = 1

    # Define time, beta and probability and adiabatic_check array
    probability = np.empty([len(beta_array), len(time_array)])

    if(type_of_qw == 'dependent'):
        for i in range(len(time_array)):
            for j in range(len(beta_array)):
                probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]], oracle_site_state)
    elif(type_of_qw == 'independent'):
        for i in range(len(time_array)):
            for j in range(len(beta_array)):
                probability[j][i] = -evaluate_time_independent_probability([beta_array[j], time_array[i]], oracle_site_state)

    return probability


def parallel_routine(lb_time, ub_time, lb_beta, ub_beta):

    # initialize ray multiprocessing
    ray.init()
    tic = time.perf_counter()

    # useful definitions for multicore computation
    cpu_count = 4
    time_sampling_points = 24
    sampling_per_cpu_count = 6
    beta_sampling_points = cpu_count * sampling_per_cpu_count
    beta = np.linspace(lb_beta, ub_beta, beta_sampling_points)
    time_array = np.linspace(lb_time, ub_time, time_sampling_points)
    process = []
    probability = []

    # parallel processes
    for i in range(cpu_count):
        process.append(grid_eval.remote(time_array, beta[int(
            sampling_per_cpu_count * i):int(sampling_per_cpu_count * (i + 1))]))

    # reassigning values to arrays
    for i in range(cpu_count):
        probability.append(ray.get(process[i]))

    # concatenate arrays to output array
    probability_array = np.concatenate(
        [probability[0], probability[1]], axis=0)
    for i in range(cpu_count - 2):
        probability_array = np.concatenate(
            [probability_array, probability[i + 2]])

    # shutting down ray
    ray.shutdown()

    # preparing for export and export and miscellanea
    if(type_of_qw == 'dependent'):
        file_probability = str(dimension) + '_' + topology + '_probability_' + step_function + '.npy'
    else:
        file_probability = str(dimension) + '_' + topology + '_probability_static.npy'

    file_time = str(dimension) + '_circ_time.npy'
    file_beta = str(dimension) + '_circ_beta.npy'

    # export heatmap plot
    #heatmap2d(probability_array, time_array, beta, step_function)

    np.save(file_probability, probability_array)
    np.save(file_time, time_array)
    np.save(file_beta, beta)
    toc = time.perf_counter() - tic

    print('Success: N ', dimension, ' in ', int(toc / 60), ' min')
    return 0


if __name__ == '__main__':


    print('\n \n \t*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *')
    print('\t*  Quantum Search on Graph with time-dependent Quantum Walks   *')
    print('\t*  M. Garbellini, Dept. of Physics, University of Milan        *')
    print('\t*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *\n')


    #type of quantum walk (time dependent or time independent)
    print('single/multiple (_step), dependent/independent, circular/complete, step_function,  beta, dim (extremes)')
    input = str.split(str(input()))


    type_of_run = input[0]
    type_of_qw = input[1]
    topology = input[2]
    step_function = input[3]
    b = input[4]
    beta = int(str(b))

    rtolerance = 1e-6
    atolerance = 1e-6
    step_functions = ['lin', 'sqrt', 'cbrt', 'cerf_3', 'cerf_5', 'cerf_7']



    if(input[0] == 'single'):

        dimension = int(input[5])
        parallel_routine(0.1, dimension, 0, beta)

    elif(input[0] == 'multiple'):

        min_dim = int(input[5])
        max_dim = int(input[6])
        dims = np.arange(min_dim, max_dim + 1, 2)

        for dim in dims:
            dimension = dim
            parallel_routine(0.1, dimension, 0, beta)

    elif(input[0] == 'single_step'):

        for step in step_functions:
            step_function = step
            dimension = int(input[5])
            parallel_routine(0.1, dimension, 0, beta)

    elif(input[0] == 'multiple_step'):

        min_dim = int(input[5])
        max_dim = int(input[6])
        dims = np.arange(min_dim, max_dim + 1, 2)

        for step in step_functions:
            step_function = step
            for dim in dims:
                dimension = dim
                parallel_routine(0.1, dimension, 0, beta)

    else:
        print('Error: undefined type_of_run')
