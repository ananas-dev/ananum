import numpy as np
from utils import save_matrix

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

save_matrix("laplacian.bin", A)
