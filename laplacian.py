import numpy as np
from solver import eig

start = -1
end = 1

n = 10

h = (end - start) / (n + 2)

A = np.zeros((n, n))

for i in range(0, n):
    if i > 0:
        A[i, i-1] = 1

    if i < n-1:
        A[i, i+1] = 1

    A[i, i] = -2

A = 1/(h**2) * A

eigenvalues, _ = np.linalg.eig(A)
print("Eigenvalues numpy:", eigenvalues)

eigenvalues = eig(A)
print("Eigenvalues C:", eigenvalues)