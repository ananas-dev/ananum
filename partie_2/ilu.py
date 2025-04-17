import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Example sparse matrix
A = sp.csr_matrix([
    [4, -1, 0, 0],
    [-1, 4, -1, 0],
    [0, -1, 4, -1],
    [0, 0, -1, 3]
], dtype=float)

# Incomplete LU factorization
ilu = spla.spilu(A)

# You can get L and U explicitly
L = ilu.L      # Lower triangular sparse matrix
U = ilu.U      # Upper triangular sparse matrix

print(L)
print(U)