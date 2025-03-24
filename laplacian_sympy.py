import numpy as np
import sympy as sp
from solver import eig
import matplotlib.pyplot as plt

def create_tridiagonal_matrix(n):
    # Définir le paramètre h
    h_val = sp.Rational(2, n + 2)  # Forme symbolique fractionnaire exacte
    
    # Initialiser une matrice symbolique de zéros
    A = sp.zeros(n, n)
    
    # Remplir la matrice selon le modèle donné
    for i in range(n):
        # Diagonale principale: -2/(h²)
        A[i, i] = -2 / (h_val**2)
        
        # Sous-diagonale: 1/(h²)
        if i > 0:
            A[i, i-1] = 1 / (h_val**2)
            
        # Sur-diagonale: 1/(h²)
        if i < n-1:
            A[i, i+1] = 1 / (h_val**2)
    
    return A

x = []
y = []

for n in range(5, 500):
    A = create_tridiagonal_matrix(n)
    x.append(n)

    data_sym = []
    for eigval, multiplicity in A.eigenvals().items():
        real_part, imag_part = eigval.as_real_imag()
        eigval_numeric = float(real_part)  # Extract only the real part
        data_sym.extend([eigval_numeric] * multiplicity)  # Repeat based on multiplicity

    A_np = np.array(A.tolist(), dtype=float)
    data_solver = eig(A_np)

    data_sym.sort()
    data_solver.sort()

    data_sym = np.array(data_sym[:5])
    data_solver = np.array(data_solver[:5])


    mean = np.mean(np.abs(data_sym - data_solver))

    print(n)

    y.append(mean)


plt.plot(x, y)

np.savetxt("kek.txt", np.array(y))

plt.xlabel("n")
plt.ylabel("Erreur de convergence moyenne")

plt.show()



