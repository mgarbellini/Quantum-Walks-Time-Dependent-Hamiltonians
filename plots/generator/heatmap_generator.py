# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it


import sys
import time
import numpy as np
from scipy import linalg
from scipy.integrate import odeint, solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import ray

def heatmap2d(probability, time, beta, type, dimension):

    time_array = time
    beta_array = beta
    for i in range(len(time_array)):
        time_array[i] = round((time_array[i]), 1)
    for i in range(len(beta_array)):
        beta_array[i] = round(beta_array[i], 2)

    plt.imshow(probability, cmap='inferno_r', aspect=1., origin={'lower'}, vmin=0, vmax=0.4)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(range(len(time_array)), time_array, rotation='vertical')
    plt.yticks(range(len(beta_array)), beta_array)
    plt.xlabel('Time', fontweight="bold")
    plt.ylabel('Gamma', fontweight="bold")
    plt.colorbar()

    file_name = str(dimension) + '_heatmap_time_dependendent_' + \
        type + '.pdf'
    plt.savefig(file_name)
    plt.clf()
    plt.close()

def sampled_min_t_p(probability, time):

    for i in range(len(time)):
        print(time[i], float (time[i]/probability[5,i]), float (time[i]/probability[10,i]), float (time[i]/probability[15,i]), float (time[i]/probability[20,i]))

def sampled_t_over_p(probability, time):

    for i in range(len(time)):
        print(time[i],  probability[5,i],  probability[10,i],  probability[15,i],  probability[20,i])


if __name__ == '__main__':

    dimension = 47
    type = ['static', 'lin', 'sqrt', 'cbrt', 'cerf']
    type = str(sys.argv[1])
    probability = np.load(str(dimension) + '_circ_prob_' + type + '.npy')
    time = np.load(str(dimension) + '_circ_time.npy' )
    beta =np.load(str(dimension) + '_circ_beta.npy')

    #heatmap2d(probability, time, beta, type, dimension)

    #sampled_min_t_p(probability, time)
    sampled_t_over_p(probability, time)
