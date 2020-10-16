# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np


def load_data_file(dim, flag):

    data_type = ["_circ_prob_static.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'



    probability = np.load(str(dim) + data_type[flag])
    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)

    return probability, time, beta


def routine(dim, flag):

    #load all the necessary files
    probability, time, beta = load_data_file(dim,flag)

    for i in range(len(time)):
        for j in range(len(beta)):
            print(time[i], beta[j], probability[j,i])
        if(i<len(time)+1):
            print(" ")




if __name__ == '__main__':

    dim = int(sys.argv[1])
    flag = int(sys.argv[2])
    routine(dim, flag)
