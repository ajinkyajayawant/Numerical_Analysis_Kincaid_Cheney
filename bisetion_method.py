# Numerical Analysis Kincaid Cheney
# Algorithm pg 59

# The main function for the question
import numpy, math, sys

delt = 1e-10
epsil = sys.float_info.epsilon
M = 40 # Number of iterations

# The function for bisection
def bisect(a,b,M,delt,epsil,f):
    # input a, b, M, \delta, \epsilon
    u = f(a)
    v = f(b)
    e  = b - a
    # print('a = {0},b = {1},u = {2},v = {3}'.format(a,b,u,v))
    if numpy.sign(u) == numpy.sign(v):
        return
    for k in range(M):
        e = e/2.0
        c = a + e
        w = f(c)
        # print k, c, w, e
        if abs(e)<delt or abs(w)<epsil:            
            return c
        if numpy.sign(w) != numpy.sign(u):
            b = c
            v = w
            # print('a = {0},b = {1},u = {2},v = {3}'.format(a,b,u,v))
        else:
            a = c
            u = w
            # print('a = {0},b = {1},u = {2},v = {3}'.format(a,b,u,v))
    return a+e

# First question
a1 = 1e-6
b1 = 1
f1 = lambda x: x**(-1) - 2**x
r1 = bisect(a1,b1,M,delt,epsil,f1)
print('The root of the function is {0}'.format(r1))
print('The value of the function at the root is {0}'.format(f1(r1)))

# Second question
a2 = 1
b2 = 3
f2 = lambda x: 2**(-x) + math.exp(x) + 2*math.cos(x) - 6
r2 = bisect(a2,b2,M,delt,epsil,f2)
print('The root of the function is {0}'.format(r2))
print('The value of the function at the root is {0}'.format(f2(r2)))

# Tests done:
# Test the passing of functions - Tested
# Test with known function - Tested
# Test it on the Hw question that I solved yesterday manually
# Visual value comparing
# a0 = math.pi/2
# b0 = 2
# f0 = lambda x: x - 2* math.sin(x)
# r0 = bisect(a0,b0,M,delt,epsil,f0))
# print('The root of the function is {0}'.format(r0)
# print('The value of the function at the root is {0}'.format(f0(r0)))
# Test the two functions using wolfram alpha - Tested first function
