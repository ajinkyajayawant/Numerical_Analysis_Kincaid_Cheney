# Numerical Analysis Kincaid Cheney
# Algorithm pg 190, 201

import numpy as np
from scipy.linalg import toeplitz
import warnings

def gauss_seidel(A,b,x0,n,M):
    # This algorithm implements the Gauss Seidel method on the pg 190 of the Kincaid book
    x = np.copy(x0)
    for k in range(M):
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i,:],x) + A[i,i]*x[i])/A[i,i]

    return x

def test_gauss_seidel():
    n1 = 3
    n_iter = 20
    eps1 = 1e-10
    # Just some random matrix and array
    A1 = toeplitz([1,3,5],[1,-1,5])
    x0 = np.array([-1,2,-4])
    b = np.dot(A1,x0)
    x1 = gauss_seidel(A1,b,x0,n1,n_iter)
    if np.linalg.norm(x1-x0)>eps1:
        warnings.warn("The iteration should have resulted in almost the same value")
    return

def cheby_acc(G,n,c,u,a,b,M,delt):
    # This code implements the Chebyshev acceleration from Kincaid Pg 201
    gam = 2.0/(2-b-a)
    alph = ((b-a)/(2*(2-b-a)))**2
    # Do we need to give those outputs
    v = extrap(gam,n,G,c,u)    
    rho = 1.0/(1-2*alph)
    u = cheby(rho,gam,n,G,c,u,v)    
    k = 3
    stop = False
    while k<M and not stop:
        rho = 1.0 - rho*alph
        v = cheby(rho,gam,n,G,c,v,u)       
        rho = 1.0/(1 - rho*alph)
        u = cheby(rho,gam,n,G,c,u,v)        
        if max(abs(u-v)) < delt:
            stop = True        
        k = k+1
    return u


def extrap(gam,n,G,c,u):
    v = gam*c + (1-gam)*u
    v = gam*np.dot(G,u) + v
    return v

def cheby(rho,gam,n,G,c,u0,v):
    # Cheby method on pg 201 of kincaid
    u = np.copy(u0)
    u = rho*gam*c+rho*(1-gam)*v+(1-rho)*u
    u = rho*gam*np.dot(G,v) + u
    return u

    
def main():
    test_gauss_seidel()
    A1 = np.array([[3.0,1.0,1.0],[1.0,3.0,-1.0],[3.0,1.0,-5.0]])
    b1 = np.array([5.0,3.0,-1.0])
    x10 = np.array([1.0,0.0,0.0])
    n1 = 3
    M1 = 10
    x1 = gauss_seidel(A1,b1,x10,n1,M1)
    print("The solution to the first equation \n{0}x={1}\n using Gauss seidel method is {2}\n".format(A1,b1,x1))

    A2 = np.array([[3.0,1,1],[3,1,-5],[1,3,-1]])
    b2 = np.array([5.0,-1,3])
    x20 = np.array([1.0,0,0])
    n2 = 3
    M2 = 10
    x2 = gauss_seidel(A1,b1,x20,n1,M1)
    print("The solution to the second equation \n{0}x={1}\n using Gauss seidel method is {2}\n".format(A2,b2,x2))

    A3 = np.array([[4.0,-1.0,-1.0,0.0],[-1.0,4.0,0.0,-1.0],[-1.0,0,4.0,-1.0],[0.0,-1.0,-1.0,4.0]])
    b3 = np.array([-4.0,0.0,4.0,-4.0])
    x30 = np.array([0.0,0.0,0.0,0.0])
    n3 = 4
    M3 = 10
    delt = 1e-9
    Qinv = np.linalg.inv(np.tril(A3))    
    c3 = np.dot(Qinv,b3)    
    G3 = np.eye(n3) - np.dot(Qinv,A3)    
    lambdas = np.linalg.eig(G3)[0]
    a = min(lambdas)
    b = max(lambdas)
    x1 = cheby_acc(G3,n3,c3,x30,a,b,M3,delt)
    print("The solution to the equation \n{0}x={1}\n using Chebyshev acceleration is {2}\n".format(A3,b3,x1))

if __name__ == '__main__':
    main()
