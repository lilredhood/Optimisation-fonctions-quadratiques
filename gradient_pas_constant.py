# Imports
import numpy as np

def matrix_A(n): 
  diag = 4 * np.ones(n)
  hors_diag = -2 * np.ones(n-1)
  A = np.diag(diag) + np.diag(hors_diag, k = -1) + np.diag(hors_diag, k = 1)
  return A
  
def vector_b(n):
  b = np.ones(n)
  return b

# Gradient à pas constant : 

def grad_const(n, rho, x_0):
  it_max = 100000
  it = 0

  x_last = x_0
  d = - (matrix_A(n) @ x_last - vector_b(n)) 
  x_now = x_last + rho * d

  x = [x_0] # liste des itérations x^k.
  err = [] # liste d'erreurs.

  while ((np.linalg.norm(x_now - x_last) > 1e-12) and (it < it_max)):

    it += 1
    x.append(x_now)
    err.append(np.linalg.norm(x_now - x_last))

    x_last = x_now
    d = - (matrix_A(n) @ x_last - vector_b(n)) 
    x_now = x_last + rho * d
  
    x.append(x_now)
    err.append(np.linalg.norm(x_now - x_last))

  return x, err, it