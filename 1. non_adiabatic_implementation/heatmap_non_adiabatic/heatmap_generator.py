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

#useful global variables, shouldn't be too inefficient
global dimension
global step_function
global rtolerance, atolerance #error tolerance (relative and absolute) for RK45 intergrator

def heatmap2d(arr: np.ndarray, time_array, beta_array):

    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]),1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i],2)

    plt.imshow(arr, cmap='inferno_r', aspect= 1., origin= {'lower'})
    #plt.xticks(np.linspace(0, 40, 30, dtype=int), rotation='vertical')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")

    title = 'Non-Adiabatic Probability N=' + str(dimension)
    plt.title(title,  y=1.05,fontweight="bold",  ha = 'center')
    plt.text(20,len(beta_array)+0.5, 'White boxes represent p>0.22', fontsize=8, horizontalalignment='center')

    plt.colorbar()

    #text results
    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            if(arr[j][i]>0.22):
                plt.gca().add_patch(Rectangle((-0.5+i, -0.5+j), 1, 1, fill=True, color = 'white', linewidth=0.5))



    file_name = str(dimension) + '_probability_heatmap_non_adiabatic.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()


#MAIN
dimension = int(sys.argv[1])


prob = str(dimension) + '_non_adiab_probability.npy'
beta = str(dimension) + '_non_adiab_beta_array.npy'
time = str(dimension) + '_non_adiab_time_array.npy'

probability = np.load(prob)
beta_array = np.load(beta)
time_array = np.load(time)


heatmap2d(probability, time_array, beta_array)
