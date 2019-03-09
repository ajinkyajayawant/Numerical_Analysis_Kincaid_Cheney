# Numerical Analysis Kincaid Cheney
# Algorithm pg 127, 131

# import math, sys

# Check the indexing to match with the range idiosyncracies and python 0 indexing
import numpy as np


# Forward substitution
def frwd_subs(n,A,b):
    # Check lower triangular
    # Check diagonal non zero
    if (not np.allclose(A,np.tril(A))) or (not all(np.diag(A) != 0)):
        print('Matrix is not lower triangular or diagonals non-zero')

    x = np.empty([n,1])

    for i in range(n):
        x[i] = (b[i] - np.matmul(A[i,0:i],x[0:i])/A[i,i]

    return x

# Backward substitution
def bck_subs(n,A,b):
    # Check upper triangular
    # Check diagonal non zero
    if (not np.allclose(A,np.triu(A))) or (not all(np.diag(A) != 0)):
        print('Matrix is not upper triangular or diagonals non-zero')

    x = np.empty([n,1])
    
    for i in range(n-1,-1,-1):
        x[i] = (b[i]- np.matmul(A[i,i:n],x[i:n])/A[i,i]

    return x

# Dolittle factorization
def dolil_lu(n,A):
    L1 = np.empty([n,n])
    U1 = np.empty([n,n])
    for k in range(n):
        L1[k,k]  = 1
        for j in range(k,n,1):
            # u_kj = a_kj - sum l_ks u_sj
            U1[k,j] = A[k,j] - np.matmul(L1[k,:k],U1[:k,j])
        for i in range(k+1,n,1):
            L1[i,k] = (A[i,k] - np.matmul(L1[i,:k],U1[:k,k]))/U1[k,k]
    return np.concatenate((L1,U1),axis = 1)

# Solve system using LU
def solve_lin(n,A,b):
    # Check whether sizes A, b same
    if A.shape[0] != b.shape[0]:
        print('Number of rows in A and b should be same')

    # Get LU decomposition
    LU = dolil_lu(n,A)
    L = LU[:,0:n]
    U = LU[:,n:2*n+1]

    # forward substitution
    x1 = frwd_subs(n,L,b)

    # back substitution
    x = bck_subs(n,U,x1)

    # Check whether length of x is n
    if x.shape[0] != n:
        print('Length of x should be n')

    return x

# Main function
A1 = np.array([[1,2],[2,3]])
b1 = np.array([[3],[-1]])
n1 = 2

A1 = np.array([[4,3,0,0],[8,1,2,0],[0,5,3,6],[0,0,-5,7]])
b1 = np.array([[2],[3],[0],[5]])
n2 = 4
# Test for forward substitution

# Test for backward substitution

# Test for dolittle factorization

# How to do LU factorization?

# Test for LU factorization
x1 = solve_lin(n1,A1,b1)
print('{0}'.format(x1))
x2 = solve_lin(n2,A2,b2)
