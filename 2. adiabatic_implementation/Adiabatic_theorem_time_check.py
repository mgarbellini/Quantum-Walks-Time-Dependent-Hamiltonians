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

def generate_hamiltonian(beta, s, derivative):

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
    time_hamiltonian = (1 - s)*hamiltonian

    #generate problem_hamiltonian (i.e. adding oracle tipical energy to center site)
    index = int((dimension-1)/2)
    time_hamiltonian[index,index] += - (s)*beta

    hamiltonian_derivative = -1*hamiltonian
    hamiltonian_derivative[index,index] += -beta

    if(derivative == 1):
        return hamiltonian_derivative
    else:
        return time_hamiltonian

def compute_eigenvalues_eigenvectors(beta, s, eigen_s):

    t_hamiltonian = generate_hamiltonian(beta, s, 0)
    eigenvalues, eigenstates = linalg.eig(t_hamiltonian)
    idx = eigenvalues.argsort()[::1]
    eigenvalues = eigenvalues[idx]
    eigenstates = eigenstates[:,idx]

    if(eigen_s == 1):
        return eigenstates
    else:
        return eigenvalues.real

def compute_gamma(s,beta):

    #find eigenstates
    #compute hamiltonian_derivative
    #return  | < phi1 | dH | phi0 > |

    eigenstates_array = compute_eigenvalues_eigenvectors(beta, s, 1)
    hamiltonian_derivative = generate_hamiltonian(beta, s, 1)

    phi0 = np.empty([dimension,1])
    phi1 = np.empty([dimension,1])

    for i in range(dimension):
        phi0[i] = eigenstates_array[i,0]
        phi1[i] = eigenstates_array[i,1]

    gamma = np.dot(np.transpose((np.conj(phi1))), np.dot(hamiltonian_derivative, phi0))
    return -np.abs(gamma)

def compute_energy_diff(s, beta):

    energy = compute_eigenvalues_eigenvectors(beta,s,0)

    return (energy[1]-energy[0])

def compute_adiabatic_time(beta):

    #SET DIMENSION
    #GAMMA MAXIMIZATION
    par_bnds = ([0, 1],)
    x = 0.5
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=(beta))
    minimization = basinhopping(compute_gamma, x,  minimizer_kwargs=minimizer_kwargs, niter=25)
    gamma_max = -minimization.fun

    #ENERGY MINIMUM
    par_bnds = ([0, 1],)
    x = 0.5
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=(beta))
    minimization = basinhopping(compute_energy_diff, x,  minimizer_kwargs=minimizer_kwargs, niter=25)
    energy_min = minimization.fun

    #TIME BOUNDS FOR ADIABATIC THEOREM
    adiabatic_time = gamma_max/(energy_min**2)

    return adiabatic_time

def compute_predicted_time_min(beta):
    #ENERGY MINIMUM
    par_bnds = ([0, 1],)
    x = 0.5
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=(beta,0))
    minimization = basinhopping(compute_energy_diff, x,  minimizer_kwargs=minimizer_kwargs, niter=50)
    energy_min = minimization.fun

    return energy_min

def compute_predicted_time_max(beta):
    #ENERGY MAXIMUM
    par_bnds = ([0, 1],)
    x = 0.5
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=(beta,1))
    minimization = basinhopping(compute_energy_diff, x,  minimizer_kwargs=minimizer_kwargs, niter=50)
    energy_max = minimization.fun

    return energy_max

#Need to compute instaneaus eigenvalues and eigenstates of complete time
#dependent hamiltonian (interested in only ground and 1st excited states)
"""
beta_range = np.linspace(0.01, 4, num=100, endpoint=False)
adiabatic_time_distribution = np.empty([100, 15])
dimension_array  = [3,5,7,9,11,13,15,17,19,21,23,25,27,29]
index = 0
sum_toc = 0
for i in beta_range:
    tic = time.perf_counter()
    adiabatic_time_distribution[index,0] = i
    for j in range(14):
        dimension = dimension_array[j]
        adiabatic_time_distribution[index,j+1] = compute_adiabatic_time(i)
    index += 1
    toc = time.perf_counter() - tic
    sum_toc += toc
    time_remaining = int(((100 - index)*(sum_toc/index))/60)
    print('Progress: ', index,'% # Est. Time Remaining: ',time_remaining, ' min')

np.savetxt('adiabatic_time_distribution.txt', adiabatic_time_distribution, fmt='%.3e')
"""

#Check eigenvalues no-crossing
"""
dimension_array  = [3,5,7,9,11,13,15,17,19,21,23,25,27,29]
eigenvalues_crossing = np.empty([14, 5])

dimension = 27
energy_max = compute_predicted_time_max(0.604)
energy_min = compute_predicted_time_min(0.604)

print(-energy_max, energy_min)
print(-float(1)/energy_max,float(1)/energy_min )


for i in range(14):
    dimension = dimension_array[i]
    eigenvalues_crossing[i, 0] = dimension
"""



"""
#Adiabatic Time given beta

dimension_array  = [3,5,7,9,11,13,15,17,19,21,23,25,27,29]
beta_array = [2.985, 4.44, 3.899, 3.033, 2.054, 2.215, 1.47, 1.511, 1.507,0.918, 0.860, 0.707, 0.604, 0.618]

for i in range(14):
    dimension = dimension_array[i]
    adiabatic_time = compute_adiabatic_time(beta_array[i])
    print(dimension, adiabatic_time)
"""





















#end
