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

# Le gradient à pas optimale :

def grad_variable(n, x_0):
  it_max = 100000
  it = 0

  x_last = x_0
  d = matrix_A(n) @ x_last - vector_b(n) 
  rho = np.dot(d, d) / np.dot(matrix_A(n)@d , d) # formule du pas optimale.
  x_now = x_last - rho * d


  x = [x_0] # Liste des itérations x^k
  err = [] # Liste des erreurs.

  while ((np.linalg.norm(x_now - x_last) > 1e-12) and (it < it_max)): 
    
    it += 1
    err.append(np.linalg.norm(x_now - x_last))
    x.append(x_now)

    # L'algo du gradient :
    x_last = x_now
    d = matrix_A(n) @ x_last - vector_b(n) 
    rho = np.dot(d, d) / np.dot(matrix_A(n)@d , d)
    x_now = x_last - rho * d

  return x, err, it 