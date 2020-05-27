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

def generate_hamiltonian(beta, s):

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

    return time_hamiltonian

def compute_eigenvalues_eigenvectors(beta, s, eigen_s):

    t_hamiltonian = generate_hamiltonian(beta, s)
    eigenvalues, eigenstates = linalg.eig(t_hamiltonian)
    idx = eigenvalues.argsort()[::1]
    eigenvalues = eigenvalues[idx]
    eigenstates = eigenstates[:,idx]

    if(eigen_s == 1):
        return eigenstates
    else:
        return eigenvalues.real

def compute_energy_difference(s, beta):

    energy = compute_eigenvalues_eigenvectors(beta,s,0)

    return (energy[1]-energy[0])

def check_eigenvalues_crossing(beta):

    #ENERGY MINIMUM
    par_bnds = ([0, 1],)
    x = 0.5
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=par_bnds, args=(beta))
    minimization = basinhopping(compute_energy_difference, x,  minimizer_kwargs=minimizer_kwargs, niter=25)
    energy_min = minimization.fun
    if(energy_min == 0):
        crossing_flag = 0
    elif(energy_min>0):
        crossing_flag = 1
    else:
        print('Error: negative energy difference value')

    return print(round(beta,2), round(energy_min,3), crossing_flag)


if __name__ == '__main__':

    dimension = 101
    beta = np.linspace(0.1, 4, 40, True)

    for b in beta:

        check_eigenvalues_crossing(b)
