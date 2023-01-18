# Imports
import numpy as np
import scipy as sp
import scipy.sparse as spsp

def matrix_A(n): 
  diag = 4 * np.ones(n)
  hors_diag = -2 * np.ones(n-1)
  A = np.diag(diag) + np.diag(hors_diag, k = -1) + np.diag(hors_diag, k = 1)
  return A
  
def vector_b(n):
  b = np.ones(n)
  return b

# Version creuse CSR.

#Multiplication d'une matrice par un vecteur, ici A par b.

#La fonction multiplication:
def matvect_multiply(A, b):
  y = np.zeros(np.size(b)) 
  for i in range(np.size(b)): 
    for j in range(A.indptr[i], A.indptr[i + 1]): 
      c = A.indices[j]
      y[i] = y[i] + A.data[j] * b[c]
  return y


# Gradient à pas variable : 

def grad_variable_CSR(n, x_0):
  it_max = 1000
  it = 0

  A = spsp.diags([[4.]*n, [-2]*(n-1), [-2] *(n-1)], [0,1,-1])
  A = spsp.csr_matrix(A) 

  x_last = x_0
  r = matvect_multiply(A, x_last) - vector_b(n) 
  d = r
  rho = - (np.dot(r, d) / np.dot(matvect_multiply(A, d) , d) )
  x_now = x_last + rho * d
  r = matvect_multiply(A, x_now) - vector_b(n)

  while((np.linalg.norm(x_now - x_last) > 1e-12) and (it < it_max)):

    if(np.linalg.norm(r) < 1e-12):
      break

    x_last = x_now
    alpha = - np.dot( matvect_multiply(A, r), d)/np.dot(matvect_multiply(A, d), d) 
    d = r + alpha * d

    rho = - np.dot(r, d)/np.dot(matvect_multiply(A, d), d)
    x_now = x_last + rho * d
    r = matvect_multiply(A, x_now) - vector_b(n)

    it += 1

  return x_now