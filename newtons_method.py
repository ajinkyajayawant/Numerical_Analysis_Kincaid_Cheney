# Numerical Analysis Kincaid Cheney
# Algorithm pg 64

# Newton's method
# Could update functions to take the derivative
import math, sys
import numpy as np
from sympy import Symbol
from mpmath import *

# f1: Derivative of f.
def newt_alg(x0,M,delt,epsil,f,f1):
    v = f(x0)
    if abs(v)<epsil:
        return
    for k in range(M):
        x1 = x0 - v/f1(x0)
        v = f(x1)        
        # output k, x1, v
        if abs(x1-x0)<delt or abs(v)<epsil:
            return x0        
        x0 = x1
    return x0

x0 = 5.0
M = 10
delt = 1e-10
epsil = sys.float_info.epsilon

def matrix_der(f1,f2,f3,xx,yy,zz,x0,y0,z0):
    # This function takes in the functions and returns the corresponding matrix evaluated at a vector
    # xx, yy, zz are teh symbolic variables over which the functions are defined
    # x0,y0,z0 are the values over which you want to evaluate
    mat = np.empty([3,3])    
    mat[0,0] = f1.diff(xx).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[0,1] = f1.diff(yy).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[0,2] = f1.diff(zz).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[1,0] = f2.diff(xx).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[1,1] = f2.diff(yy).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[1,2] = f2.diff(zz).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[2,0] = f3.diff(xx).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[2,1] = f3.diff(yy).subs([(xx,x0),(yy,y0),(zz,z0)])
    mat[2,2] = f3.diff(zz).subs([(xx,x0),(yy,y0),(zz,z0)])
    return mat

def multi_dimen_newt(f1,f2,f3,xx,yy,zz,x0,x1,x2):
    # This function does the multi dimensional newton's method on a system of equations given by f1,f2,f3,...
    # Not going into the details of the epsilong deltas in this question don't have that much time.
    f_vec = np.zeros([3,1])
    x_vec = np.empty([3,1])    
    x_vec[0] = x0
    x_vec[1] = x1
    x_vec[2] = x2

    for k in range(40):        
        f_vec[0] = f1.subs([(xx,x0),(yy,x1),(zz,x2)])
        f_vec[1] = f2.subs([(xx,x0),(yy,x1),(zz,x2)])
        f_vec[2] = f3.subs([(xx,x0),(yy,x1),(zz,x2)])
        x_vec = x_vec - np.matmul(np.linalg.inv(matrix_der(f1,f2,f3,xx,yy,zz,x0,x1,x2)),f_vec)
        x0 = x_vec[0]
        x1 = x_vec[1]
        x2 = x_vec[2]
    return np.concatenate((x_vec,f_vec),axis=1)
    

# Q10 from the book
fx = lambda x: x**3 - 5*x**2 + 3*x -7
fx1 = lambda x: 3*x**2 - 10*x + 3

r = newt_alg(x0,M,delt,epsil,fx,fx1)
print('The root of the function is {0}'.format(r))
# Testing
print('And value of the function at {0} is {1}'.format(r,fx(r)))

# Q36 from the book
# Check the functions
xx = Symbol('x')
yy = Symbol('y')
zz = Symbol('z')
f1 = xx*yy-zz**2-1
f2 = xx*yy*zz-xx**2+yy**2-2
f3 = mp.e**xx-mp.e**yy+zz-3

print('The functions are as follows')
print('f1 ={0}'.format(f1))
print('f2 ={0}'.format(f2))
print('f3 ={0}'.format(f3))

x0 = 1.0
x1 = 1.0
x2 = 1.0
rf = multi_dimen_newt(f1,f2,f3,xx,yy,zz,x0,x1,x2)
print('The root of the function is {0}'.format(rf[:,0]))
# Testing
print('And value of the functions at {0} is {1}'.format(rf[:,0],rf[:,1]))
