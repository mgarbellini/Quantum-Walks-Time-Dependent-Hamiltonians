# M. Garbellini
# Dept. of Physics
# Università degli Studi di Milano
# matteo.garbellini@studenti.unimi.it


#ADIABATIC QUANTUM WALKS
#STEP A: Time evolution of circle graph witn n nodes

import numpy as np
from scipy import linalg
flat_state = np.matrix([1, 1, 1])
center_cycle = np.matrix([0,1,0])
gamma = np.matrix([0, 0, 0]; 0, 1, 0; 0, 0, 0])
hamiltonian = np.matrix([2, -1, -1; -1, 2, -1; -1, -1, 2])
hamiltonian = hamiltonian + gamma
unitary = linalg.expm(-(1j)*hamiltonian)

flat_state_evolved = np.dot(unitary,flat_state.transpose())
probability = np.dot(center_cycle, flat_state_evolved)
probability_square = np.abs(probability)**2

#Create Diagonal Matrix D

"""
class Hamiltonian:

    def __init__(self, dimension, oracle_site, gamma):
        self.dimension = dimension
        self.oracle_site = oracle_site
        self.gamma = gamma
"""
print(probability_square)
