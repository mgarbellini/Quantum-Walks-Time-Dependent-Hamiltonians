import sys
import numpy as np


def check_probability(dim1, dim2):

    prob1 = np.load(str(dim1) + '_non_adiab_probability.npy')
    prob2 = np.load(str(dim2) + '_non_adiab_probability.npy')

    prob1 = prob1.flatten()
    prob2 = prob2.flatten()

    for i in range(len(prob1)):
        print(prob1[i]-prob2[i])



if __name__ == '__main__':

    check_probability(int(sys.argv[1]), int(sys.argv[2]))
