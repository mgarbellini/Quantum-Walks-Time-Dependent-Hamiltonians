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
global type_of_qw
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

    elif(topology == "circular"):
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
    # generate diagonal matrix
    diag_matrix = np.empty([dimension, dimension])
    diag_matrix.fill(0)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                diag_matrix[i][j] = 2
            else:
                diag_matrix[i][j] = 0

    # generate loop adjacency matrix
    adj_matrix = np.empty([dimension, dimension])
    adj_matrix.fill(0)
    for i in range(dimension):  # colonna
        for j in range(dimension):  # riga
            if i == j:
                if i == 0 & j == 0:
                    adj_matrix[i][dimension - 1] = 1
                    adj_matrix[i][j + 1] = 1
                elif i == dimension - 1 & j == dimension - 1:
                    adj_matrix[i][j - 1] = 1
                    adj_matrix[i][0] = 1
                else:
                    adj_matrix[i][j - 1] = 1
                    adj_matrix[i][j + 1] = 1

    # generate laplacian of loop
    hamiltonian = diag_matrix - adj_matrix

    # generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    index = int((dimension - 1) / 2)
    hamiltonian[index][index] += - gamma

    return hamiltonian


def evaluate_time_independent_probability(time, gamma, oracle_site_state):

    # generate_hamiltonian
    hamiltonian = generate_time_independent_hamiltonian(dimension, gamma)

    # Generate so called 'flat-state'
    psi_0 = np.empty([dimension, 1])
    psi_0.fill(1 / np.sqrt(dimension))

    # set to zero variables
    probability = 0
    psi_t = np.empty([dimension, 1])
    psi_t.fill(0)

    # define time-evolution
    unitary_time_evolution = np.empty([dimension, dimension])
    unitary_time_evolution.fill(0)
    unitary_time_evolution = linalg.expm(-(1j) * hamiltonian * time)
    psi_t = np.dot(unitary_time_evolution, psi_0)
    probability = np.dot(oracle_site_state.transpose(), psi_t)

    # return 'crossing' probability
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

    sh_solved = solve_ivp(schrodinger_equation, [
                          0., time], y0, method='RK45', atol=atolerance, rtol=rtolerance, args=(beta, time))
    # for more precise results use method RK45 and max_step=t_step_max
    # for less precise results but faster computation use 'BDF'
    psi_t = np.empty(dimension, dtype=complex)
    for i in range(dimension):
        psi_t[i] = sh_solved.y[i, len(sh_solved.y[i]) - 1]

    normalization = np.dot(np.conj(psi_t), psi_t)
    return psi_t * (1 / np.sqrt(normalization))

# routine to evaluate probability |<w|psi(t)>|^2


def evaluate_probability(x, oracle_site_state):

    # find evolved state (already normalized)
    psi_t = solve_schrodinger_equation(x[1], x[0])

    # probability evaluation
    probability = np.dot(oracle_site_state.transpose(), psi_t)
    if(np.abs(probability)**2 > 1):
        print('Error: probability out of bounds: ', np.abs(probability)**2)
    else:
        return -np.abs(probability)**2


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
                probability[j][i] = -evaluate_probability(
                    [beta_array[j], time_array[i]], oracle_site_state)
    elif(type_of_qw == 'independent'):
        for i in range(len(time_array)):
            for j in range(len(beta_array)):
                probability[j][i] = -evaluate_time_independent_probability(
                    [beta_array[j], time_array[i]], oracle_site_state)

    return probability


def parallel_routine():

    beta = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                    1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
                    2, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
                    3, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95]
    # initialize ray multiprocessing
    ray.init()
    tic = time.perf_counter()

    # useful definitions for multicore computation
    cpu_count = 3
    #time_sampling_points = 24
    sampling_per_cpu_count = len(beta)/3
    if(int(len(beta)/3) < (len(beta)/3)):
        print("ERROR CPU BETA COUNT")

    #beta_sampling_points = cpu_count * sampling_per_cpu_count
    #beta = np.linspace(lb_beta, ub_beta, beta_sampling_points)
    #time_array = np.linspace(lb_time, ub_time, time_sampling_points)
    time_0 = (np.pi/4)*np.sqrt(dimension)
    time_1 = np.sqrt(dimension)
    time_2 = (np.pi/2)*np.sqrt(dimension)
    time_3 = 2*np.sqrt(dimension)
    time_array = [time_0, time_1, time_2, time_3]
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

    file_probability = str(dimension) + '_probability_' + step_function + '.npy'
    file_time = str(dimension) + '_circ_time.npy'
    file_beta = str(dimension) + '_circ_beta.npy'

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



    rtolerance = 1e-6
    atolerance = 1e-6
    step_functions = ['lin', 'sqrt', 'cbrt', 'cerf_3']
    dims = np.arange(3, 71 + 1, 2)
    topology = 'circular'
    type_of_qw = 'dependent'

    for i in range(4):
        step_function = step_functions[i]
        for dim in dims:
            dimension = dim
            parallel_routine()
