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
import ray

from numba import njit
from numba import int32, float64
from numba import types, typed, typeof, deferred_type
from numba.experimental import jitclass


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" CLASS: HAMILTONIAN  """""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hamiltonian_types = [
    ('dimension', int32),
    ('type', int32),
    ('topology', ??),  #char or numerical id?
    ('step_function', int32),
    ('target', int32),
    ('laplacian', float64[:]),
    ('hamiltonian', float64[:]),
]

@jitclass(hamiltonian_types):
class Hamiltonian:
    """Hamiltonian class"""
    def __init__(self, dimension, type, topology, target, step_function):
