# Numerical Analysis Kincaid Cheney
# Algorithm pg 174

import numpy as np
from scipy.linalg import toeplitz
import warnings
from hw5_code_ajinkya import palu,sol_ph

def test_iter_solve_one():
    # Check whether exact solution gives no updates
    # Test 1
    eps1 = 1e-10
    n1 = 3
    A1 = np.identity(n1)
    b1 = np.array([1,2,3])
    x0 = np.array([1,2,3])
    r0,e0,x1 = iter_solve_one(A1,b1,x0)
    print('x1 is {0}'.format(x1))
    print('r0 is {0}'.format(r0))
    if np.linalg.norm(x1-x0) > eps1:
        warnings.warn("The iteration should have resulted in almost the same value")
    # Test 2
    return

def iter_solve_one(A,b,x0):
    # This function does one iteration of the iterative refinement procedure 
    # Test the sizes of the inputs    
    n = A.shape[0]
    
    r0 = b - np.dot(A,x0)
    A1,p = palu(A,n)    
    e0 = sol_ph(A1,r0,p,n) #Gaussian elimination
    # print('A is {0}'.format(A))
    # print('e0 is {0}'.format(e0))
    x1 = x0 + e0
    return (r0,e0,x1)

def test_rich_iter():
    # This subroutine tests the richardson iteration
    n1 = 3
    n_iter = 20
    eps1 = 1e-10
    # Just some random matrix and array
    A1 = toeplitz([1,3,5],[1,-1,5])
    x0 = np.array([-1,2,-4])
    b = np.dot(A1,x0)
    x1,r = richardson_iter(A1,b,x0,n1,n_iter)
    if np.linalg.norm(x1-x0)>eps1:
        warnings.warn("The iteration should have resulted in almost the same value")
    return

def richardson_iter(A,b,x0,n,M):
    # This function implements the richardson method on pg 184
    x = np.copy(x0)
    r = np.zeros(n)
    for k in range(M):
        for i in range(n):
            r[i] = b[i] - np.dot(A[i,:],x)
        for i in range(n):
            x[i] = x[i] + r[i]        
    return x,r

def test_jacob_iter():
    # This subroutine tests the richardson iteration
    n1 = 3
    n_iter = 20
    eps1 = 1e-10
    # Just some random matrix and array
    A1 = toeplitz([1.0,3.0,5.0],[1.0,-1.0,5.0])
    x0 = np.array([-1.0,2.0,-4.0])
    b = np.dot(A1,x0)
    x1 = jacobi_iter(A1,b,x0,n1,n_iter)
    if np.linalg.norm(x1-x0)>eps1:
        warnings.warn("The iteration should have resulted in almost the same value")
    return

def jacobi_iter(A,b,x0,n,M):
    u = np.zeros(n)
    x = np.copy(x0)   

    for k in range(M):        
        for i in range(n):
            u[i] = (b[i] - np.dot(A[i,:],x)+A[i,i]*x[i])/A[i,i]            
        for i in range(n):
            x[i] = u[i]                    
    return x

def main():
    # Problem 33
    A1 = np.array([[60.0,30.0,20.0],[30.0,20.0,15],[20.0,15.0,12.0]])    
    b = np.array([110.0,65.0,47.0])
    x0 = np.array([0.0,0.0,0.0])
    n_it = 3
    
    # test_iter_solve_one()
    
    for i in range(n_it):
        (r0,e0,x1) = iter_solve_one(A1,b,x0)
        x0 = x1
    print('The estimate of x after 3 iterations is {0}'.format(x0))
    print('The estimate of b from our x after 3 iterations is {0}'.format(np.dot(A1,x0)))

    # Richardson method
    # test_rich_iter()
    # Jacobi method
    # test_jacob_iter()

    A2 = np.array([[1.0,1.0/2,1.0/3],[1.0/3,1.0,1.0/2],[1.0/2,1.0/3,1.0]])
    b2 = 11.0*np.ones(3)/18
    xi = np.array([1.0,-1.0,1.0])
    ni = 3
    n_it = 20
    xj = jacobi_iter(A2,b2,xi,ni,n_it)
    xr,r = richardson_iter(A2,b2,xi,ni,n_it)    
    print('The solution via jacobi iteration is {0}'.format(xj))
    print('The solution via richardson iteration is {0}'.format(xr))

if __name__ == '__main__':
    main()
