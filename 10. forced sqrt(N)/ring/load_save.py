import sys
import time
import numpy as np
from scipy import linalg
from scipy.integrate import odeint, solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt


probability = np.load('probability.npy')
beta = np.load('beta.npy')
dim= np.load('dimension.npy')


for j in range(len(beta)):
    string = str(round(beta[j],1))
    for i in range(len(dim)):
        string = string + ' ' + str(round(probability[j,i],3))
    print(string)

print(len(dim))
