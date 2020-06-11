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
from matplotlib.patches import Rectangle
import matplotlib.colors as colors

#useful global variables, shouldn't be too inefficient
global dimension
global step_function
global rtolerance, atolerance #error tolerance (relative and absolute) for RK45 intergrator

def heatmap2d(arr: np.ndarray, time_array, beta_array):

    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]),1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i],2)


    #plt.imshow(arr,cmap='RdBu_r', aspect= 1., origin= {'lower'}, vmin=-0.5, vmax=0.5)
    plt.imshow(arr,aspect= 1., origin= {'lower'}, norm=colors.LogNorm(vmin=arr.min(), vmax=arr.max()),cmap='PuBu_r' )
    #cmap='inferno_r'
    # norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),cmap='PuBu_r'
    #plt.xticks(np.linspace(0, 40, 30, dtype=int), rotation='vertical')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Dynamic Implementation - Probability N =' + str(dimension)
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    #plt.suptitle(title, fontweight="bold", ha='center')
    plt.colorbar()
    levels = [0.9, 0.95]

    ct = plt.contour(arr,levels, colors='white')

    plt.clabel(ct)


    file_name = str(dimension) + '_heatmap_sqrt.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()


#MAIN
dimension = int(sys.argv[1])

prob = str(dimension) + '_probability_sqrt.npy'
prob1 = str(dimension) + '_probability.npy'
beta = str(dimension) + '_beta_array.npy'
time = str(dimension) + '_time_array.npy'
probability_1 = np.load(prob1)
probability = np.load(prob)
beta_array = np.load(beta)
time_array = np.load(time)

p = probability_1 - probability
heatmap2d(p, time_array, beta_array)
