# Numerical Analysis Kincaid Cheney
# Q26 Pg 169(I think this code had a small bug)

import numpy as np

def vec_norm_1(x):
    return sum(abs(x))

def vec_norm_inf(x):
    return max(abs(x))

def matrix_norm(A):
    return vec_norm_inf(np.apply_along_axis(vec_norm_1,0,A))
    # return max(np.sum(abs(A),axis=0))

A1 = np.array([[4,-3,2],[-1,0,5],[2,6,-2]])
print('The infinity norm of our matrix \n{0}\n is {1}'.format(A1,matrix_norm(A1)))
