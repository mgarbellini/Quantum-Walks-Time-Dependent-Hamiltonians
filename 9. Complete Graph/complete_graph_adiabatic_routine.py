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

def compute_eigenvalues_eigenvectors(s, beta, eigen_flag):

    t_hamiltonian = generate_hamiltonian_derivative(s,beta,0)
    eigenvalues, eigenstates = linalg.eig(t_hamiltonian)
    idx = eigenvalues.argsort()[::1]
    eigenvalues = eigenvalues[idx]
    eigenstates = eigenstates[:,idx]

    if(eigen_flag == 1):
        return eigenstates
    elif(eigen_flag == 0):
        return eigenvalues.real
    else:
        print('NameError: compute_eigen flag unknown!')
        return 0

def compute_gamma(s,beta):

    #find eigenstates
    #compute hamiltonian_derivative
    #return  | < phi1 | dH | phi0 > |

    eigenstates_array = compute_eigenvalues_eigenvectors(s, beta, 1)
    hamiltonian_derivative = generate_hamiltonian_derivative(s, beta, 1)

    phi0 = np.empty([dimension,1])
    phi1 = np.empty([dimension,1])

    for i in range(dimension):
        phi0[i] = eigenstates_array[i,0]
        phi1[i] = eigenstates_array[i,1]

    gamma = np.dot(np.transpose((np.conj(phi1))), np.dot(hamiltonian_derivative, phi0))
    return -np.abs(gamma)

def compute_energy_diff(s,beta):

    energy = compute_eigenvalues_eigenvectors(s, beta, 0)

    return (energy[1]-energy[0])

#check if adiabatic theorem with current parameters is applicable
#returns adiabatic_results which contains Adiabatic_Time, Max_Energy_Diff,
#Min_Energy_Diff, Crossing_Flag
def adiabatic_theorem_check(beta, time):


    #Performance counter
    #GAMMA MAXIMIZATION
    par_bnds = ([0, 1],)
    energy_min = 1

    minimization = shgo(compute_gamma, par_bnds,n=25, iters=1, args=(beta,),sampling_method='sobol')
    gamma_max = -minimization.fun

    #ENERGY MINIMUM

    minimization = shgo(compute_energy_diff, par_bnds,n=25, iters=1, args=(beta,),sampling_method='sobol')
    energy_min = minimization.fun

    #TIME BOUNDS FOR ADIABATIC THEOREM
    adiabatic_time = gamma_max/(energy_min**2)

    if(time < adiabatic_time):
        return 0
    else:
        return 1

def heatmap2d(arr: np.ndarray, time, beta):

    time_array = time
    beta_array = beta
    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]),1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i],2)

    plt.imshow(arr, cmap='inferno_r', aspect= 1., origin= {'lower'})
    #plt.xticks(np.linspace(0, 40, 30, dtype=int), rotation='vertical')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Dynamic Probability Complete Graph N=' + str(dimension)
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    plt.colorbar()

    levels = [0.9, 0.95, 0.99]
    ct = plt.contour(arr,levels, colors='white')
    plt.clabel(ct)


    file_name = str(dimension) + '_cg_heatmap.pdf'
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

def parallel_routine(lb_time, ub_time, lb_beta, ub_beta):

    tic = time.perf_counter()

    cpu_count = 5
    time_sampling_points = 40
    beta_sampling_points = cpu_count * 6
    beta = np.linspace(lb_beta, ub_beta, beta_sampling_points)
    time_array = np.linspace(lb_time, ub_time, time_sampling_points)
    process = []
    probability = []

    #parallel processes
    for i in range(cpu_count):
        process.append(grid_eval.remote(time_array, beta[int(6*i):int(6*(i+1))]))

    #reassigning values to arrays
    for i in range(cpu_count):
        probability.append(ray.get(process[i]))

    #concatenate arrays to output array
    probability_array = np.concatenate([probability[0], probability[1]], axis = 0)
    for i in range(cpu_count - 2):
        probability_array = np.concatenate([probability_array, probability[i+2]])


    #preparing for export and export and miscellanea
    file_probability = str(dimension) + '_cg_probability.npy'
    file_time = str(dimension) + '_cg_time_array.npy'
    file_beta = str(dimension) + '_cg_beta_array.npy'

    #export heatmap plot
    heatmap2d(probability_array, time_array, beta)

    np.save(file_probability, probability_array)
    np.save(file_time, time_array)
    np.save(file_beta, beta)
    toc = time.perf_counter() - tic

    return print('Success: N ',dimension,' in ',int(toc/60),' min')


if __name__ == '__main__':

    step_function = 2
    rtolerance = 1e-6
    atolerance = 1e-6



    #dimension = int(sys.argv[1])

    dimension = 15
    parallel_routine(0, dimension*2, 0, 15)
