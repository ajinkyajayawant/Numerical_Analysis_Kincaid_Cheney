# Numerical Analysis Kincaid Cheney
# Algorithm pg 148

import numpy as np
import scipy as sp
import scipy.linalg

# PA = LU factorization algorithm
# returns a tuple
def palu(A1,n):
    A = np.copy(A1)
    p = np.zeros(n,dtype=int)
    s = np.zeros(n)    
    for i in range(n):
        p[i] = i
        s[i] = max(abs(A[i,:]))
    for k in range(n-1):
        # Find the maximum element after dividing
        #print('Shape of s[k:] is {0}'.format(s[k:].shape))
        # print('Shape of A[k:,k] is {0}'.format(A[k:,k].shape))
        j = k + np.argmax(abs(A[k:,k])/s[k:])
        p[j],p[k] = p[k],p[j]
        for i in range(k+1,n):            
            z = A[p[i],k]/A[p[k],k]
            A[p[i],k] = z
            # print('z is')
            # print(z)
            for j in range(k+1,n):
                A[p[i],j] = A[p[i],j] - z*A[p[k],j]
    return A,p
    
def sol_ph(A,b1,p,n):
    x = np.zeros(n)
    b = np.copy(b1)
    for k in range(n-1):
        for i in range(k+1,n):
            b[p[i]] = b[p[i]] - A[p[i],k]*b[p[k]]
    for i in range(n-1,-1,-1):        
        x[i] = (b[p[i]] - np.dot(A[p[i],i+1:],x[i+1:]))/A[p[i],i]
    return x
    
def sol_tridiag(A,b1,n):
    b = np.copy(b1)
    x = np.zeros(n)
    # this solves the linear system where the matrix A is a tridiagonal matrix
    if not np.array_equal(np.triu(A,2)+np.tril(A,-2),np.zeros([n,n])):
        print('A is not tridiagonal')
        print(A)        
    a = np.diag(A,-1)
    d = np.copy(np.diag(A))
    c = np.diag(A,1)
    
    for i in range(1,n):
        d[i] = d[i] - a[i-1]*c[i-1]/d[i-1]
        b[i] = b[i] - a[i-1]*b[i-1]/d[i-1]
    x[n-1] = b[n-1]/d[n-1]
    for i in range(n-2,-1,-1):
        x[i] = (b[i] - c[i]*x[i+1])/d[i]
        
    return x
    

def test_palu(A,n):
    print('Input A is')
    print(A)
    A1, p1 = palu(A,n)
    print('My output A is')
    print(A1)
    print('My output permutation is')
    print(p1)
    # Testing with scipy LU code
    p2,l2,u2 = sp.linalg.lu(A)
    print('The scipy permutation is')
    print(p2)
    print('The scipy L is')
    print(l2)
    print('The scipy U is')
    print(u2)
    # Constructing L and U from my A and testing
    lr = np.diag(np.ones(n)) + np.tril(A1[p1.tolist(),:],-1)
    ur = np.triu(A1[p1.tolist(),:])
    A2 = np.dot(lr,ur)    
    print('The reconstructed A is')
    print(A2)#[p1.tolist(),:]
    print('The permuted A is')     
    print(A[p1.tolist(),:])#[p1.tolist(),:]
    return
    
def test_sol(A,n):
    # Check whether A is singular or nor
    if np.linalg.matrix_rank(A)<n:
        print('A should not be singular')
    x = np.random.rand(n)
    b = np.dot(A,x)
    
    # Do the decomposition    
    A1,p1 = palu(A,n)
    x1 = sol_ph(A1,b,p1,n)
    
    # See outputs
    print('Original x was')
    print(x)
    print('Solution x is')
    print(x1)
    
def test_tridiag_sol(A,n):
    # Check whether A is singular or nor
    if np.linalg.matrix_rank(A)<n:
        print('A should not be singular')
    x = np.random.rand(n)
    b = np.dot(A,x)
    
    # Do the decomposition    
    x1 = sol_tridiag(A,b,n)
    
    # See outputs
    print('Original x for tridiag system was')
    print(x)
    print('Solution x for tridiag system is')
    print(x1)    
    
def main():
    # Test case 1
    A = np.array([[2.0,1.0],[0.5,3.0]])
    n = 2
    test_palu(A,n)
    print('------')

    # Test case 2
    # Can't verify this because the scipy permutation is different from mine
    n2 = 4
    A2 = np.random.rand(n2,n2)
    test_palu(A2,n2)
    test_sol(A2,n2) 
    print('------')
    # Test the tridigonal solution
    n3 = 5
    A3 = np.diag(np.random.rand(n3)) + np.diag(np.random.rand(n3-1),-1) + np.diag(np.random.rand(n3-1),1)
    test_tridiag_sol(A3,n3) 
    print('------')

if __name__ == '__main__':
    main()
