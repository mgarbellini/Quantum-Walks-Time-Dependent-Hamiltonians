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

def heatmap2d(arr: np.ndarray, time_array, beta_array, non_prob,non_prob_2, non_time, adiabatic_check):

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

    title = 'Adiabatic Probability N=' + str(dimension) + '\nNon-Adiabatic (dashed): p = ' + str(non_prob) + ', T = ' + str(non_time)
    plt.title(title,  y=1.04,fontweight="bold",  ha = 'center')
    #plt.suptitle(title, fontweight="bold", ha='center')
    plt.colorbar()
    levels = [0.9, 0.95, 0.99]
    non_adiabatic_levels = [non_prob,non_prob_2]
    ct = plt.contour(arr,levels, colors='white')
    cta = plt.contour(arr,non_adiabatic_levels, colors ='black', linestyles = 'dashed')
    plt.clabel(ct)

    #non physical results
    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            if(adiabatic_check[j][i] == 0):
                plt.gca().add_patch(Rectangle((-0.5+i, -0.5+j), 1, 1, fill=False, color = 'white', linewidth=0, hatch = '///////'))

    file_name = str(dimension) + '_probability_heatmap_2nd_version.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()


#MAIN
dimension = 29

probability = np.load('29_probability.npy')
beta_array = np.load('29_beta_array.npy')
time_array = np.load('29_time_array.npy')
adiabatic_check = np.load('29_adiabatic_check.npy')

heatmap2d(probability, time_array, beta_array,0.19,0.38,14 , adiabatic_check)
