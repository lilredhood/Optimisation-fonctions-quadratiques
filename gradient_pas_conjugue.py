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

# Gradient à direction conjugué : 

def grad_conjugue(n, x_0):
  it_max = n
  it = 0

  x_last = x_0
  r = matrix_A(n) @ x_last - vector_b(n) 
  d = r
  rho = - (np.dot(r, d) / np.dot(matrix_A(n)@d , d) )
  x_now = x_last + rho * d
  r = matrix_A(n) @ x_now - vector_b(n)

  x = [x_0] # liste des itérations x^k.
  err = [] # liste d'erreurs.

  while((np.linalg.norm(x_now - x_last) > 1e-12) and (it < it_max)):
    
    it += 1
    x.append(x_now)
    err.append(np.linalg.norm(r))

    if(np.linalg.norm(r) < 1e-12):
      break

    x_last = x_now
    alpha = - np.dot( matrix_A(n) @ r, d)/np.dot(matrix_A(n) @ d, d) 
    d = r + alpha * d

    rho = - np.dot(r, d)/np.dot(matrix_A(n) @ d, d)
    x_now = x_last + rho * d
    r = matrix_A(n) @ x_now - vector_b(n)

  return x, err, it # Cette fois-ci, on renvoie les itérations x^k, l'erreur e^k et le nombre d'itérations.