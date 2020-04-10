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


#Need to compute instaneaus eigenvalues and eigenstates of complete time
#dependent hamiltonian (interested in only ground and 1st excited states)

dimension = 27
beta = 1.7

#GAMMA MAXIMIZATION
par_bnds = ([0, 1],)
x = 0.5
minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=beta)
minimization = basinhopping(compute_gamma, x,  minimizer_kwargs=minimizer_kwargs, niter=50)
gamma_max = -minimization.fun
print('Gamma max: ', gamma_max)

#ENERGY MINIMUM
par_bnds = ([0, 1],)
x = 0.5
minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=beta)
minimization = basinhopping(compute_energy_diff, x,  minimizer_kwargs=minimizer_kwargs, niter=100)
energy_min = minimization.fun
print('Energy min: ',energy_min)

#TIME BOUNDS FOR ADIABATIC THEOREM
adiabatic_time = gamma_max/(energy_min**2)

print('Adiabatic time: ',adiabatic_time)
