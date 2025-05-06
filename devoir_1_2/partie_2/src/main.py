import ctypes
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse import diags # Plus nécessaire si on utilise pyamg ou mmread
from scipy.io import mmread # Pour charger depuis un fichier .mtx
import os
import platform
import time
import pyamg # Nécessite pip install pyamg

# --- Chargement de la bibliothèque C ---
lib_path = None
lib_name = ""

if platform.system() == "Windows":
    lib_name = "devoir2.dll"
elif platform.system() == "Darwin": # macOS
    lib_name = "libdevoir2.so"
else: # Linux
    lib_name = "libdevoir2.so"

# Cherche la bibliothèque dans le dossier courant ou un sous-dossier 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
# Ajustez ces chemins si la bibliothèque est ailleurs par rapport au script
possible_paths = [
    os.path.join(script_dir, lib_name),
    os.path.join(script_dir, "src", lib_name),
    os.path.join(script_dir, "..", "partie_2", "src", lib_name),
    os.path.join(script_dir, "..", "src", lib_name)
]


for path in possible_paths:
    # Utiliser le chemin absolu pour éviter les ambiguïtés
    abs_path = os.path.abspath(path)
    print(f"Tentative de chargement depuis : {abs_path}")
    if os.path.exists(abs_path):
        lib_path = abs_path
        break

if lib_path is None:
    print(f"Erreur: Impossible de trouver la bibliothèque partagée '{lib_name}' dans les chemins:")
    for p in possible_paths:
        print(f" - {os.path.abspath(p)}")
    print("Veuillez la compiler et la placer dans un des chemins ci-dessus ou ajuster 'possible_paths'.")
    print(f"Répertoire courant: {os.getcwd()}")
    exit(1)


try:
    # Charger la bibliothèque
    if platform.system() == "Windows":
        c_lib = ctypes.WinDLL(lib_path)
    else:
        c_lib = ctypes.CDLL(lib_path)
    print(f"Bibliothèque C chargée avec succès depuis: {lib_path}")
except OSError as e:
    print(f"Erreur lors du chargement de la bibliothèque C depuis '{lib_path}': {e}")
    exit(1)
except Exception as e:
    print(f"Une erreur inattendue s'est produite lors du chargement: {e}")
    exit(1)


# --- Définition des types C pour ctypes ---
c_int_p = ctypes.POINTER(ctypes.c_int)
c_double_p = ctypes.POINTER(ctypes.c_double)

# --- Définition des signatures des fonctions C ---
# (Identique à la version précédente, on les suppose correctes)
# void Matvec(...)
try:
    c_lib.Matvec.argtypes = [
        ctypes.c_int, ctypes.c_int, c_int_p, c_int_p,
        c_double_p, c_double_p, c_double_p
    ]
    c_lib.Matvec.restype = None
except AttributeError:
    print("Erreur: Fonction 'Matvec' non trouvée.")
    exit(1)
# int CG(...)
try:
    c_lib.CG.argtypes = [
        ctypes.c_int, ctypes.c_int, c_int_p, c_int_p,
        c_double_p, c_double_p, c_double_p, ctypes.c_double
    ]
    c_lib.CG.restype = ctypes.c_int
except AttributeError:
    print("Avertissement: Fonction 'CG' non trouvée.")
# void ILU(...)
try:
    c_lib.ILU.argtypes = [
        ctypes.c_int, ctypes.c_int, c_int_p, c_int_p,
        c_double_p, c_double_p
    ]
    c_lib.ILU.restype = None
except AttributeError:
    print("Avertissement: Fonction 'ILU' non trouvée.")
# int PCG(...)
try:
    c_lib.PCG.argtypes = [
        ctypes.c_int, ctypes.c_int, c_int_p, c_int_p,
        c_double_p, c_double_p, c_double_p, ctypes.c_double
    ]
    c_lib.PCG.restype = ctypes.c_int
except AttributeError:
    print("Avertissement: Fonction 'PCG' non trouvée.")
# int csr_sym(...)
try:
    c_lib.csr_sym.argtypes = []
    c_lib.csr_sym.restype = ctypes.c_int
    use_symmetric_storage = (c_lib.csr_sym() == 1)
    print(f"Stockage CSR symétrique utilisé (selon csr_sym()): {use_symmetric_storage}")
except AttributeError:
    print("Avertissement: Fonction 'csr_sym' non trouvée. On suppose stockage non symétrique.")
    use_symmetric_storage = False


# --- Fonctions Wrapper Python ---
# (Identiques à la version précédente)
def py_matvec(matrix_csr, v_np):
    """Appelle la fonction C Matvec."""
    n = matrix_csr.shape[0]
    nnz = matrix_csr.nnz
    if n != matrix_csr.shape[1] or n != v_np.shape[0]:
        raise ValueError("Incompatibilité de dimensions pour Matvec")
    rows_idx = matrix_csr.indptr.astype(np.int32)
    cols = matrix_csr.indices.astype(np.int32)
    A_data = matrix_csr.data.astype(np.float64)
    v_np = v_np.astype(np.float64)
    Av_np = np.zeros(n, dtype=np.float64)
    rows_idx_ptr = rows_idx.ctypes.data_as(c_int_p)
    cols_ptr = cols.ctypes.data_as(c_int_p)
    A_data_ptr = A_data.ctypes.data_as(c_double_p)
    v_ptr = v_np.ctypes.data_as(c_double_p)
    Av_ptr = Av_np.ctypes.data_as(c_double_p)
    c_lib.Matvec(n, nnz, rows_idx_ptr, cols_ptr, A_data_ptr, v_ptr, Av_ptr)
    return Av_np

def py_cg(matrix_csr, b_np, x0_np, tol=1e-6, max_iter_factor=2):
    """Appelle la fonction C CG (Gradient Conjugué)."""
    if not hasattr(c_lib, 'CG'):
        raise NotImplementedError("La fonction CG n'est pas disponible.")
    n = matrix_csr.shape[0]
    nnz = matrix_csr.nnz
    if n != matrix_csr.shape[1] or n != b_np.shape[0] or n != x0_np.shape[0]:
        raise ValueError("Incompatibilité de dimensions pour CG")
    rows_idx = matrix_csr.indptr.astype(np.int32)
    cols = matrix_csr.indices.astype(np.int32)
    A_data = matrix_csr.data.astype(np.float64)
    b_np = b_np.astype(np.float64)
    x_np = x0_np.astype(np.float64).copy()
    rows_idx_ptr = rows_idx.ctypes.data_as(c_int_p)
    cols_ptr = cols.ctypes.data_as(c_int_p)
    A_data_ptr = A_data.ctypes.data_as(c_double_p)
    b_ptr = b_np.ctypes.data_as(c_double_p)
    x_ptr = x_np.ctypes.data_as(c_double_p)
    print(f"Démarrage de CG (n={n}, nnz={nnz})...")
    start_time = time.time()
    iterations = c_lib.CG(n, nnz, rows_idx_ptr, cols_ptr, A_data_ptr, b_ptr, x_ptr, ctypes.c_double(tol))
    end_time = time.time()
    print(f"CG terminé en {end_time - start_time:.4f} secondes.")
    if iterations < 0: print("Avertissement: CG a retourné une erreur.")
    elif iterations >= max_iter_factor * n: print(f"Avertissement: CG max iter ({iterations}).")
    else: print(f"CG a convergé en {iterations} itérations.")
    return x_np, iterations

def py_ilu(matrix_csr):
    """Appelle la fonction C ILU (Factorisation Incomplète LU)."""
    if not hasattr(c_lib, 'ILU'):
         raise NotImplementedError("La fonction ILU n'est pas disponible.")
    n = matrix_csr.shape[0]
    nnz = matrix_csr.nnz
    if n != matrix_csr.shape[1]: raise ValueError("Matrice non carrée pour ILU")
    rows_idx = matrix_csr.indptr.astype(np.int32)
    cols = matrix_csr.indices.astype(np.int32)
    A_data = matrix_csr.data.astype(np.float64)
    L_data_np = A_data.copy() # ILU modifie ceci en place (supposé)
    rows_idx_ptr = rows_idx.ctypes.data_as(c_int_p)
    cols_ptr = cols.ctypes.data_as(c_int_p)
    A_data_ptr = A_data.ctypes.data_as(c_double_p)
    L_data_ptr = L_data_np.ctypes.data_as(c_double_p)
    print(f"Calcul de la factorisation ILU (n={n}, nnz_A={nnz})...")
    start_time = time.time()
    c_lib.ILU(n, nnz, rows_idx_ptr, cols_ptr, A_data_ptr, L_data_ptr)
    end_time = time.time()
    print(f"ILU terminé en {end_time - start_time:.4f} secondes.")
    L_csr = csr_matrix((L_data_np, cols, rows_idx), shape=(n, n))
    print(f"Facteur ILU calculé, nnz_L(U)={L_csr.nnz}.")
    return L_csr

def py_pcg(matrix_csr, b_np, x0_np, tol=1e-6, max_iter_factor=2):
    """Appelle la fonction C PCG (Gradient Conjugué Préconditionné)."""
    if not hasattr(c_lib, 'PCG'):
        raise NotImplementedError("La fonction PCG n'est pas disponible.")
    n = matrix_csr.shape[0]
    nnz = matrix_csr.nnz
    if n != matrix_csr.shape[1] or n != b_np.shape[0] or n != x0_np.shape[0]:
        raise ValueError("Incompatibilité de dimensions pour PCG")
    rows_idx = matrix_csr.indptr.astype(np.int32)
    cols = matrix_csr.indices.astype(np.int32)
    A_data = matrix_csr.data.astype(np.float64)
    b_np = b_np.astype(np.float64)
    x_np = x0_np.astype(np.float64).copy()
    rows_idx_ptr = rows_idx.ctypes.data_as(c_int_p)
    cols_ptr = cols.ctypes.data_as(c_int_p)
    A_data_ptr = A_data.ctypes.data_as(c_double_p)
    b_ptr = b_np.ctypes.data_as(c_double_p)
    x_ptr = x_np.ctypes.data_as(c_double_p)
    print(f"Démarrage de PCG (n={n}, nnz={nnz})...")
    start_time = time.time()
    iterations = c_lib.PCG(n, nnz, rows_idx_ptr, cols_ptr, A_data_ptr, b_ptr, x_ptr, ctypes.c_double(tol))
    end_time = time.time()
    print(f"PCG terminé en {end_time - start_time:.4f} secondes.")
    if iterations < 0: print("Avertissement: PCG a retourné une erreur.")
    elif iterations >= max_iter_factor * n: print(f"Avertissement: PCG max iter ({iterations}).")
    else: print(f"PCG a convergé en {iterations} itérations.")
    return x_np, iterations

def calculate_bandwidth(matrix_csr):
    """Calcule la largeur de bande d'une matrice CSR."""
    if matrix_csr.nnz == 0:
        return 0
    matrix_coo = matrix_csr.tocoo(copy=False) # Utiliser copy=False pour l'efficacité si possible
    if matrix_coo.nnz == 0: return 0 # Vérifier après conversion aussi
    return np.max(np.abs(matrix_coo.row - matrix_coo.col))

# --- Exemple d'utilisation ---

if __name__ == "__main__":

    # --- Option 1: Générer une matrice FEM avec pyamg ---
    try:
        N_grid = 500 # Taille de la grille NxN -> matrice (N*N) x (N*N)
        n_mat = N_grid * N_grid
        print(f"\nGénération d'une matrice FEM (Poisson 2D) de taille {n_mat}x{n_mat} avec pyamg...")
        # stencil = pyamg.gallery.stencil_grid([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], (N_grid, N_grid), format='csr') # Exemple 9 points
        A_sparse = pyamg.gallery.poisson((N_grid, N_grid), format='csr')
        A_sparse = A_sparse.astype(np.float64) # S'assurer que c'est float64
        print("Matrice pyamg générée.")

    except ImportError:
        print("\nBibliothèque 'pyamg' non trouvée. Essayez 'pip install pyamg'.")
        print("Génération d'une matrice alternative simple (diagonale dominante).")
        n_mat = 250000 # Taille équivalente à N_grid=500
        diag_vals = -4.0 * np.ones(n_mat)
        sub_diag_vals = np.ones(n_mat - 1)
        sup_diag_vals = np.ones(n_mat - 1)
        far_sub_diag_vals = np.ones(n_mat - N_grid) # Approximation
        far_sup_diag_vals = np.ones(n_mat - N_grid) # Approximation

        A_sparse = diags(
            [far_sub_diag_vals, sub_diag_vals, diag_vals, sup_diag_vals, far_sup_diag_vals],
            [-N_grid, -1, 0, 1, N_grid], shape=(n_mat, n_mat), format='csr', dtype=np.float64
        )
        A_sparse = A_sparse.astype(np.float64)


    # --- Option 2: Charger une matrice depuis un fichier Matrix Market ---
    # matrix_file = "votre_matrice.mtx" # Mettez ici le chemin vers votre fichier
    # if os.path.exists(matrix_file):
    #      print(f"\nChargement de la matrice depuis {matrix_file}...")
    #      try:
    #          A_sparse_coo = mmread(matrix_file)
    #          A_sparse = A_sparse_coo.tocsr().astype(np.float64)
    #          n_mat = A_sparse.shape[0]
    #          if A_sparse.shape[0] != A_sparse.shape[1]:
    #               print("Erreur: La matrice chargée n'est pas carrée.")
    #               exit(1)
    #          print("Matrice chargée.")
    #      except Exception as e:
    #          print(f"Erreur lors du chargement de la matrice: {e}")
    #          exit(1)
    # else:
    #      print(f"Fichier matrice {matrix_file} non trouvé. Utilisation de la matrice générée.")
    #      # Assurez-vous que A_sparse existe déjà (via Option 1 ou une alternative)

    # --- Suite du code ---
    n_mat = A_sparse.shape[0]
    nnz_mat = A_sparse.nnz
    bandwidth_mat = calculate_bandwidth(A_sparse)
    print(f"Matrice utilisée: n = {n_mat}, nnz = {nnz_mat}, largeur de bande = {bandwidth_mat}")
    print(f"Ratio nnz / (largeur_bande * n) = {nnz_mat / (bandwidth_mat * n_mat if bandwidth_mat > 0 else 1):.3f}")
    print(f"Ratio nnz / n = {nnz_mat / n_mat:.3f} (non-nuls par ligne en moyenne)")


    # Créer un vecteur b (second membre) et x0 (estimation initiale)
    b = np.random.rand(n_mat).astype(np.float64)
    x0 = np.zeros(n_mat, dtype=np.float64)

    # --- Test Matvec ---
    print("\nTest Matvec:")
    start_mv = time.time()
    Av_py = py_matvec(A_sparse, b)
    end_mv = time.time()
    print(f"Matvec exécuté en {end_mv - start_mv:.4f} secondes.")

    # --- Test CG ---
    if hasattr(c_lib, 'CG'):
        print("\nTest CG:")
        x_cg, it_cg = py_cg(A_sparse, b, x0.copy(), tol=1e-8) # Utiliser x0.copy() pour ne pas affecter PCG
        # Vérification optionnelle
        if n_mat < 50000: # Limiter la vérification pour les très grandes matrices
            print("Vérification du résidu CG...")
            res_cg = b - A_sparse.dot(x_cg)
            norm_res_cg = np.linalg.norm(res_cg)
            norm_b = np.linalg.norm(b)
            print(f"  Norme résidu CG ||b - Ax||: {norm_res_cg:.4e}")
            if norm_b > 1e-15:
                 print(f"  Norme relative résidu CG ||b - Ax|| / ||b||: {norm_res_cg / norm_b:.4e}")
        else:
            print("Vérification du résidu CG sautée (matrice trop grande).")
    else:
        print("\nCG non testé (fonction non trouvée).")
        it_cg = -1 # Pour la comparaison finale

    # --- Test ILU (si la fonction existe) ---
    if hasattr(c_lib, 'ILU'):
        print("\nTest ILU:")
        L_ilu_csr = py_ilu(A_sparse)
    else:
        print("\nILU non testé (fonction non trouvée).")
        L_ilu_csr = None


    # --- Test PCG (si la fonction existe) ---
    if hasattr(c_lib, 'PCG'):
        print("\nTest PCG:")
        x_pcg, it_pcg = py_pcg(A_sparse, b, x0.copy(), tol=1e-8)
        # Vérification optionnelle
        if n_mat < 50000:
             print("Vérification du résidu PCG...")
             res_pcg = b - A_sparse.dot(x_pcg)
             norm_res_pcg = np.linalg.norm(res_pcg)
             norm_b = np.linalg.norm(b)
             print(f"  Norme résidu PCG ||b - Ax||: {norm_res_pcg:.4e}")
             if norm_b > 1e-15:
                 print(f"  Norme relative résidu PCG ||b - Ax|| / ||b||: {norm_res_pcg / norm_b:.4e}")
        else:
             print("Vérification du résidu PCG sautée (matrice trop grande).")

        if it_cg >= 0 and it_pcg >=0:
             print(f"\nComparaison CG vs PCG: {it_cg} vs {it_pcg} itérations.")
    else:
        print("\nPCG non testé (fonction non trouvée).")