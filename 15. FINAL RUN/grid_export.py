# M. Garbellini
# Dept. of Physics
# Universita degli Studi di Milano
# matteo.garbellini@studenti.unimi.it

import sys
import time
import numpy as np


def load_data_file(dim,flag):

    data_type = ["_circ_prob_static.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'



    probability = np.load(str(dim) + data_type[flag])
    time = np.load(str(dim) + t)
    beta = np.load(str(dim) + b)

    return probability, time, beta

def load_fig6():
    data_type = ["_probability_lin.npy","_probability_cerf_3.npy"]
    t = '_time.npy'




    probability = np.load('FIG6_' + str(dim) + data_type[flag])
    time = np.load('FIG6_' + str(dim) + t)

    return probability, time

def load_fig6():
    data_type = ["_probability_lin.npy","_probability_cerf_3.npy"]
    t = 'circ_time.npy'
    b = 'circ_beta'

    probability = np.load('FIG14_' + str(dim) + data_type[flag])
    time = np.load('FIG14_' + str(dim) + t)
    beta= np.load('FIG14_' + str(dim) + b)


    return probability, time, beta

def load_fig8(dim, flag):
    data_type = ["51_circ_prob_lin.npy","51_circ_cerf.npy"]
    t = '51_circ_time.npy'



    probability = np.load(data_type[flag])
    time = np.load(t)

    return probability, time

def load_fig7():
    data_type = ["51_circ_prob_lin.npy","_circ_prob_lin.npy","_circ_prob_sqrt.npy", "_circ_prob_cbrt.npy", "_circ_prob_cerf.npy"]
    t = '_circ_time.npy'
    b = '_circ_beta.npy'



    probability = np.load('FIG7_51_circ_prob_lin.npy')
    time = np.load('FIG7_51_circ_time.npy')
    beta = np.load('FIG7_51_circ_beta.npy')

    return probability, time, beta

def fig2_routine(index):
    probability = np.load('51_circ_prob_static.npy')
    time = np.load('51_time.npy')


    for i in range(len(time)):
        print(time[i],probability[index,i])

def grid_routine():

    #load all the necessary files
    probability, time, beta = load_data_file(51,0)

    beta_array = [0.02, 0.06, 0.10, 0.14,  0.18, 0.22 , 0.26 , 0.3 , 0.34 , 0.38 , 0.42 , 0.46 , 0.50 , 0.54 , 0.58 , 0.62 , 0.66 , 0.70 , 0.74 , 0.78 , 0.82 , 0.86 , 0.90 , 0.94]
    print("a b c")
    for i in range(len(time)):
        for j in range(len(beta)):
            print(time[i], beta[j], probability[j,i])
        if(i<len(time)):
            print(" ")

def fig6_routine(dim, flag):

    #load all the necessary files
    probability, time = load_fig6(dim,flag)

    for i in range(len(time)):
        print(round(time[i],0), round(probability[0,i],4))

def fig8_routine(dim, flag, index):

    probability, time = load_fig8(dim, flag)

    for i in range(len(time)):
        print(time[i], float(time[i]/probability[index, i]))

def fig5_routine(flag, dim):

    probability, time, beta = load_data_file(dim, flag)

    print("a b c")
    for i in range(len(time)):
        for j in range(len(beta)):
            print(time[i], beta[j], probability[j,i])
        if(i<len(time)):
            print(" ")

if __name__ == '__main__':

    #dim = int(sys.argv[1])
    flag = int(sys.argv[1])
    #index = int(sys.argv[2])
    #fig2_routine(index)

    fig5_routine(flag,51)
