# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np


def load_data_file(dim, flag):

    data_type = ["_probability_static.npy","_probability_lin.npy","_probability_sqrt.npy", "_probability_cbrt.npy", "_probability_cerf_3.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'



    probability = np.load(str(dim) + data_type[flag])
    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)

    return probability, time, beta


def routine(dim, flag):

    #load all the necessary files
    probability, time, beta = load_data_file(dim,flag)

    #evaluate (T/p)
    for i in range(len(time)):
        for j in range(len(beta)):
            probability[j,i] = float (time[i]/probability[j,i])

    min_per_type = []
    iter_per_type = []
    #find min
    for i in range(len(time)):
        min = 1000
        for j in range(len(beta)):
            if(probability[j,i] < min):
                min = probability[j,i]
        min_per_type.append(min)

    for i in range(len(min_per_type)):
        min_per_type[i] = round(min_per_type[i], 2)
        iter_per_type.append(round(float(min_per_type[i]/time[i]), 2))

    #print routine
    #print(dim, min_per_type[2])
    print(dim, iter_per_type[2])




if __name__ == '__main__':

    min_dim = 3
    max_dim = 71
    dimensions = np.arange(min_dim, max_dim + 1, 2)

    flag = int(sys.argv[1])

    print("a b")
    for dim in dimensions:
        routine(dim, flag)
