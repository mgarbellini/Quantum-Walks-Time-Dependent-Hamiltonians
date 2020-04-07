# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping




#implementing ODE solver using scipy OdeInt
#to solve Schrodinger Equation, namely d/dt|psi> = H|psi>

"""
equations need to be in the form

    y1_dot = f1(t,y1,...,yn)
    y2_dot = f2(t,y1,...,yn)
    .
    .
    yn_dot = fn(t,y1,...,yn)

with n initial conditions for each of the yi. In our case

    y1(0) = 1/sqrt(n)
    y2(0) = 1/sqrt(n)
    .
    .
    yn(0) = 1/sqrt(n)

Odeint is implemented in this manner:

    odeint(func, y0, args=(,))

where func is a function that returns a list of all the values f1,...,fn

How to deal with lists?
create empty list (let's call it lista) and append object (let's call it oggetto)

    lista = []
    lista.append(oggetto)

    
"""
