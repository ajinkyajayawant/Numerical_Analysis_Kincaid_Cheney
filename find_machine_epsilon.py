# Numerical Analysis Kincaid Cheney
# Algorithm pg 36

import sys

s = 1.0
for k in range(100):
    # print('s is changing from {0} to {1}'.format(s,0.5*s)) # Debug
    s = 0.5*s
    # print('t gets the value of {0}'.format(s+1.0)) # Debug
    t = s + 1.0
    if t <= 1.0:
        s = 2.0*s
        print('s = {0}  \nk-1 = {1}'.format(s,k-1))
        print('Datatype of s is {0}'.format(type(s)))
        print('Cross check after calling the system value {0}'.format(sys.float_info.epsilon))
        break
