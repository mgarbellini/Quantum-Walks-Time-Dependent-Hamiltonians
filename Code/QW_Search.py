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

from numba import njit, cfunc
from numba import int32, float64, complex64, complex128
from numba import types, typed, typeof, deferred_type
from numba.experimental import jitclass

from NumbaLSODA import lsoda_sig, lsoda


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: HAMILTONIAN  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hamiltonian_types = [
    ('dim', int32),
    ('type', int32),
    ('topology', int32),
    ('step_func', int32),
    ('target', int32),
    ('gamma', float64),
    ('L', float64[:,:]),
    ('H', float64[:,:]),
]
@jitclass(hamiltonian_types)
class Hamiltonian:
    """Hamiltonian class"""
    def __init__(self, dim, type, topology, target, step_func, gamma):

        self.dim = dim
        self.type = type
        self.topology = topology
        self.target = target
        self.step_func = step_func
        self.gamma = gamma

    def build_laplacian(self):
        lap = np.zeros((self.dim, self.dim))

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

        self.L = lap

    def build_hamiltonian(self):

        """Time-independent Hamiltonian"""
        self.H = self.L
        self.H[self.target, self.target] -= self.gamma

        """Time-dependent Hamiltonian:
        in this scenario at t=0 -> H = L. The routine update_hamiltonian
        updates it for t>0"""
        self.H = self.L

    def update_hamiltonian(self, t_time, T_time):

        tau = float(t_time)/T_time #normalized time [0,1]
        g_T = 0 #step function

        if self.step_func == 0:
            g_T = 0.5*(1 + (2*tau - 1)**3)
        elif self.step_func == 1:
            g_T = tau
        elif self.step_func == 2:
            g_T = np.sqrt(tau)

        #build Hamiltonian in the form (1-s)L -s*gamma*target
        self.H = (1-g_T)*self.L
        self.H[self.target, self.target] += -g_T*self.gamma

    def exp_H(self): return np.exp(self.H)

state_types = [
    ("dim", int32),
    ("ket", complex128[:]),
    ("H", Hamiltonian.class_type.instance_type),
]
@jitclass(state_types)
class State:
    """Quantum state class"""
    def __init__(self, dim):
        self.dim = dim

    def set_initial(self, type = 0):
        if type == 0:
            self.ket = np.ones((self.dim), dtype = complex128)/np.sqrt(self.dim)
            #self.ket = self.ket/np.sqrt(self.dim)
        else:
            print("Initial type missing or not yet implemented")

    def set_hamiltonian(self, hamiltonian):
        self.H = hamiltonian

    def bra(self): return np.transpose(np.conjugate(self.ket))

    def evolve(self, time):

        if self.H.type == 0: #Time-independent evolution
            self.ket = np.dot(np.exp(self.H.H*(-1j)*time),np.transpose(self.ket))
        elif self.H.type == 1: #Time-dependent evolution
            print("_runge_kutta()")
        else:
            print("Error, unknown evolution type")

"""
@njit()
def _runge_kutta(state, psi, time):
    Evolves state using 4th order Runge-Kutta integrator
    step_size = 1E-3
    n_iterations = int(time/step_size)

    for i in range(1, n_iterations + 1):
        Runge-Kutta coefficients
        k1 =
        k2 =
        k3 =
        k4 =

        psi =
"""


if __name__ == '__main__':

    #Test hamiltonian class (dim, type, topology, target, step_func, gamma)

    HC = Hamiltonian(5, 0, 0, 1, 0, 10)
    HC.build_laplacian()
    HC.build_hamiltonian()

    psi = State(5)
    psi.set_initial(0)
    psi.set_hamiltonian(HC)
    print(psi.ket)
    psi.evolve(10)
    print(psi.ket)
