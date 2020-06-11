# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#useful global variables, shouldn't be too inefficient
global dimension

def heatmap2d(prob, time_array, beta_array):

    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]),1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i],2)

    plt.imshow(prob, cmap='inferno_r', aspect= 1., origin= {'lower'},)
    #plt.xticks(np.linspace(0, 40, 30, dtype=int), rotation='vertical')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Beta', fontweight="bold")


    title = 'Adiabatic Probability N=' + str(dimension)
    plt.title(title,  y=1.08,fontweight="bold",  ha = 'center')
    #plt.suptitle(title, fontweight="bold", ha='center')
    plt.colorbar()

    file_name = str(dimension) + '_probability_heatmap.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()

def load_n_plot():
    prob = str(dimension) + '_probability.npy'
    beta = str(dimension) + '_beta_array.npy'
    time = str(dimension) + '_time_array.npy'

    probability = np.load(prob)
    beta_array = np.load(beta)
    time_array = np.load(time)

    heatmap2d(probability, time_array, beta_array)


#MAIN

dim = [31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71 ]

for i in range(len(dim)):
    dimension = dim[i]
    load_n_plot()
