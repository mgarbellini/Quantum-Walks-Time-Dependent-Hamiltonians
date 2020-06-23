# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np
from scipy import linalg

def max_probability_by_threshold(threshold, time_array, beta_array, probability_array):

    flag = 0
    time = 0
    for i in range(len(time_array)):
        for j in range(len(beta_array)):
            if(probability_array[j,i]>=threshold):
                time = time_array[i]
                flag = 1
                break

        if(flag==1):
            break

    return time

def load_data_file(dim, type):

    prob = '_cg_probability'
    t = '_cg_time_array_.npy'
    b = '_cg_beta_array_.npy'
    flag = ['_lin.npy', '_sqrt.npy', '_cbrt.npy']

    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)
    probability = np.load(str(dim) + prob + flag[type-1])

    return probability, time, beta

def max_routine(dim, type):

    probability, time, beta = load_data_file(dim, type)
    print(dim, max_probability_by_threshold(0.9, time, beta, probability), max_probability_by_threshold(0.95, time, beta, probability), max_probability_by_threshold(0.99, time, beta, probability))

if __name__ == '__main__':

    dimensions = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,41,45, 51, 61]

    for dim in dimensions:
        max_routine(dim, 2)
