import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ilupp

# Example sparse matrix
A = sp.csr_matrix([
    [4, -1, 0, 0],
    [-1, 4, -1, 0],
    [0, -1, 4, -1],
    [0, 0, -1, 3]
], dtype=float)

res = ilupp.ilu0(A)
print(res[0].todense())
print(res[1].todense())
