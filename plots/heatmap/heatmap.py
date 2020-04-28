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

#define callback functions. This allows to set precision to probability evaluation,
#e.g. stop process at 0.99 instead of 0.99875
def optimization_precision(x, probability, context):

    if(probability <= -0.99):
        return True
    else:
        return False

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

    energy = compute_eigenvalues_eigenvectors(s, beta,0)

    return (energy[1]-energy[0])

#check if adiabatic theorem with current parameters is applicable
#returns adiabatic_results which contains Adiabatic_Time, Max_Energy_Diff,
#Min_Energy_Diff, Crossing_Flag
def adiabatic_theorem_check(current_par):


    #Performance counter
    tic = time.perf_counter()
    #GAMMA MAXIMIZATION
    par_bnds = ([0, 1],)

    minimization = shgo(compute_gamma, par_bnds,n=25, iters=1, args=(current_par[2],),sampling_method='sobol')
    gamma_max = -minimization.fun

    #ENERGY MINIMUM

    minimization = shgo(compute_energy_diff, par_bnds,n=25, iters=1, args=(current_par[2],),sampling_method='sobol')
    energy_min = minimization.fun

    #TIME BOUNDS FOR ADIABATIC THEOREM
    adiabatic_time = gamma_max/(energy_min**2)
    if(energy_min>0):
        adiabatic_flag = 'True'
    else:
        adiabatic_flag = 'False'

    adiabatic_results = [adiabatic_time, gamma_max, energy_min, adiabatic_flag, (tic - time.perf_counter())/60]

    return adiabatic_results

#routine to maximize probability
def optimization(par_bnds,optimization_method, its):

    #define oracle_site_state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    if(optimization_method == 'SHGO'):

        #count time
        tic = time.perf_counter()

        #minimizer options

        opt1 = {'f_min': -1, 'f_tol': 0.1}
        opt2 = {'f_min': -1}
        #maximize probability
        #sampling_method must be chosen between 'simplicial' and 'sobol'
        maximized = shgo(evaluate_probability, par_bnds,n=40, iters=1, args=(oracle_site_state,),sampling_method='sobol')

        #computation time in minutes (rounded)
        comp_time = int(time.perf_counter() - tic)/60

        #print other local maxima found
        #set 0 for probability sorting, 1 for time sorting
        local_results = arrange_local_maxima(maximized)

        #store results
        comp_results = [dimension, -maximized.fun, maximized.x[0], maximized.x[1], comp_time]

        #check adiabatic theorem for current results
        adiabatic_results = [0,0,0,0,0]
        #adiabatic_results = adiabatic_theorem_check(comp_results)

        #return computational comp_results
        return comp_results, local_results, adiabatic_results

    elif(optimization_method == 'BH'):


            #initial values for minimization
            x0 = np.array([1.,10])

            #BH arguments for minimization
            #"L-BFGS-B" is the chosen minimization methods
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=oracle_site_state)

            #count time
            tic = time.perf_counter()

            #maximize probability
            maximized = basinhopping(evaluate_probability, x0,  minimizer_kwargs=minimizer_kwargs,niter=its)

            #computation time in minutes (rounded)
            comp_time = int(time.perf_counter() - tic)/60

            #store results
            comp_results = [dimension, float(-maximized.fun), maximized.x[0], maximized.x[1], comp_time]

            #print computational comp_results
            return comp_results
    elif(optimization_method == 'DUAL'):
        #count time
        tic = time.perf_counter()

        #maximize probability
        #sampling_method must be chosen between 'simplicial' and 'sobol'
        maximized = dual_annealing(evaluate_probability, par_bnds, args=(oracle_site_state,), maxiter=its)

        #computation time in minutes (rounded)
        comp_time = int(time.perf_counter() - tic)/60

        #store results
        comp_results = [dimension, float(-maximized.fun), maximized.x[0], maximized.x[1], comp_time]

        #return computational comp_results
        return comp_results
    else:
        print('Error: invalid optimization method')

#routine for results exporting - formatting - etc
def export_results_routine(dimension, par_bnds, method, step_function, optimization_iterations):

    #bounds check: if lb > up, switch lb-up
    #prevents error from distractions
    #if()

    results, local_results, adiabatic_results = optimization(par_bnds, method, optimization_iterations)
    print()
    print('#  #  #  #  # COMPUTATIONAL RESULTS DIM ',dimension, ' #  #  #  #  #')
    print()
    print('Optimization algorithm: ', method)
    print('Par. bounds: ', par_bnds)
    print('Comp. time (min) ~', int(results[4]))
    print()
    print('Dimension: ', dimension)
    print('g_T = (t/T)^(1/', step_function,')')
    print('Probability: ', round(results[1], 4))
    print('T: ', round(results[3], 2))
    print('Beta: ', round(results[2], 3))
    print()
    print('Adiabatic theorem applies: ', adiabatic_results[3])
    print('Adiabatic time: ', round(adiabatic_results[0], 2))
    print('Gamma max: ', round(adiabatic_results[1],3))
    print('Energy min: ', round(adiabatic_results[2],3))
    print('Computation time (min) ~', int(adiabatic_results[4]))
    print()
    print('Additional local maxima')
    print('Prob - Beta - T')
    for i in range(len(local_results)):
        print(round(local_results[i][0], 4), round(local_results[i][1], 3), round(local_results[i][2],2))
    print()
    return ('Simulation success!')

def heatmap2d(arr: np.ndarray, time_array, beta_array, non_prob, non_time):

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

    title = 'Adiabatic Probability N=' + str(dimension) + '\nNon-Adiabatic (dashed): p = ' + str(non_prob) + ', T = ' + str(non_time)
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    #plt.suptitle(title, fontweight="bold", ha='center')
    plt.colorbar()
    levels = [0.9, 0.95, 0.99]
    non_adiabatic_levels = [non_prob]
    ct = plt.contour(arr,levels, colors='white')
    cta = plt.contour(arr,non_adiabatic_levels, colors ='white', linestyles = 'dashed')
    plt.clabel(ct)
    #plt.clabel(cta)
    #plt.tight_layout()
    file_name = str(dimension) + '_probability_heatmap.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()
    #plt.show()

def grid_probability_evaluation(dimension, time_lb, time_up, time_sampling_points, beta_lb, beta_up, beta_sampling_points, non_time, non_prob):

    #Define oracle site state
    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    #Define time, beta and probability array
    time_array = np.linspace(time_lb, time_up, time_sampling_points)
    beta_array = np.linspace(beta_lb, beta_up, beta_sampling_points)
    probability = np.empty([beta_sampling_points, time_sampling_points])

    for i in range(time_sampling_points):
        for j in range(beta_sampling_points):
            #evaluate probability
            probability[j][i] = -evaluate_probability([beta_array[j], time_array[i]], oracle_site_state)

    heatmap2d(probability, time_array, beta_array, non_time, non_prob)
    return probability

def single_evaluation_benchmark(x):

    oracle_site_state = np.empty([dimension, 1])
    oracle_site_state.fill(0)
    oracle_site_state[int((dimension-1)/2)][0] = 1

    tic = time.perf_counter()
    prob = evaluate_probability(x, oracle_site_state)
    toc = time.perf_counter() - tic
    print(dimension, ": ",int(toc))
    print()
    return 1

# # # # # # #
#   MAIN    #
# # # # # # #

step_function = 1
rtolerance = 1e-6
atolerance = 1e-6



dimension = 27
beta = [0, 2]
time = [1, 320]
non_prob = 0.22
non_time = 17
time_sampling = 40
beta_sampling = 30
grid_probability_evaluation(dimension, time[0],time[1], time_sampling, beta[0], beta[1], beta_sampling, non_prob, non_time)

dimension = 29
beta = [0, 2]
time = [1, 350]
non_prob = 0.19
non_time = 14
time_sampling = 40
beta_sampling = 30
grid_probability_evaluation(dimension, time[0],time[1], time_sampling, beta[0], beta[1], beta_sampling, non_prob, non_time)



"""
print()
print('N: ', dimension)
print('T: 350')
print('Beta: 0.4')
print()
atolerance = 1e-8
rtolerance = 1e-6
print('Relative accuracy:',rtolerance)
print('Absolute accuracy:',atolerance)
tic = time.perf_counter()
normalization = solve_schrodinger_equation(350, 0.5)
toc = time.perf_counter() - tic
print('RK45 Error: ', 1-normalization)
print('Comp time: ', toc)
"""



"""
grid_probability_evaluation(dimension, 0,250,10, 0, 3, 10)

#define oracle_site_state
oracle_site_state = np.empty([dimension, 1])
oracle_site_state.fill(0)
oracle_site_state[int((dimension-1)/2)][0] = 1

tic = time.perf_counter()
x = [0.6, 350]
prob = evaluate_probability(x, oracle_site_state)
toc = time.perf_counter() - tic
print(toc)
print(-prob)
"""
