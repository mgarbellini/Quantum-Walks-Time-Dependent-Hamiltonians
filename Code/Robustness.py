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

def robustness(probability, beta_index, time_index, variation):

    R_pos = probability[beta_index, time_index] - probability[beta_index+variation, time_index]
    R_neg = probability[beta_index, time_index] - probability[beta_index-variation, time_index]
    R = round(float((R_pos + R_neg)/2),4)

    return R


def routine(dim, flag):

    #load all the necessary files
    probability, time, beta = load_data_file(dim,flag)

    max_index_array = []
    #find min index:
    max = 0
    for j in range(len(beta)):
        if(probability[j,2] > max):
            max_index = j
            max = probability[j,2]


    robustness_array = []
    for i in range(4):
        robustness_array.append(robustness(probability, max_index, 2, 1+i))

    print(dim, robustness(probability, max_index, 2, 2), robustness(probability, max_index, 2,4))



if __name__ == '__main__':

    min_dim = 7
    max_dim = 71
    dimensions = np.arange(min_dim, max_dim + 1, 2)

    flag = int(sys.argv[1])


    for dim in dimensions:
        routine(dim, flag)
