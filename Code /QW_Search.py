"""
M. Garbellini
Dept. of Physics
Universita degli Studi di Milano
matteo.garbellini@studenti.unimi.it

Quantum Walks with time-dependent Hamiltonians:
application to the search problem on graphs

This code is a rewritten version of the code used for my bachelor thesis
at the University of Milan. The old version can be found at my github page
github.com/mgarbellini.

Major features and enhancement compare to
the previous version.

- QW Spatial search on graph with time-independent Hamiltonian (see Fahri & Gutmann)
- QW Spatial search with time-dependent Hamiltonian
- Different graph topology implemented: circle, complete, random, star, etc..


- Robustness of the approach
- Localization and Multiple iteration search
"""

import sys
import time
import numpy as np
from scipy import linalg
from scipy.integrate import odeint, solve_ivp

import multiprocessing as mp

from numba import njit
from numba import int32, float64
from numba import types, typed, typeof, deferred_type
from numba.experimental import jitclass


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: HAMILTONIAN  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hamiltonian_types = [
    ('dim', int32),
    ('type', int32),
    ('topology', int32),
    ('step_func', int32),
    ('target', int32),
    ('laplacian', float64[:,:]),
    ('hamiltonian', float64[:,:]),
]

@jitclass(hamiltonian_types)
class Hamiltonian:
    """Hamiltonian class"""
    def __init__(self, dim, type, topology, target, step_func):

        self.dim = dim
        self.type = type
        self.topology = topology
        self.target = target
        self.step_func = step_func


    def build_laplacian(self):
        lap = np.empty((self.dim, self.dim))
        lap.fill(0)

        """Complete graph (0) """
        if self.topology == 0:
            lap -= 1
            for i in range(self.dim):
                lap[i,i] += self.dim

        """Cycle graph (1) """
        if self.topology == 1:
            for i in range(self.dim):
                lap[i,i] = 2
                lap[i,(i+1)%self.dim] -= 1
                lap[i,(i-1)%self.dim] -= 1

        """Random graph (2)"""
        """Star graph (3)"""

        self.laplacian = lap

    def build_hamiltonian(self):



if __name__ == '__main__':

    #Test hamiltonian class (dim, type, topology, target, step_func)

    HC = Hamiltonian(5, 0, 0, 1, 1)
    HC.build_laplacian()
    HCy = Hamiltonian(5, 0, 1, 1, 1)
    HCy.build_laplacian()
