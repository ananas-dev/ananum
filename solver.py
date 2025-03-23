import numpy as np
import subprocess

def save_matrix(filename: str, matrix: np.ndarray):
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    
    n = np.uint32(matrix.shape[0])
    
    with open(filename, "wb") as f:
        f.write(n.tobytes())
        f.write(matrix.astype(np.float64).tobytes())
    
def eig(matrix: np.ndarray):
    save_matrix("matrix.bin", matrix)
    try:
        result = subprocess.run(["./devoir1/a.out", "matrix.bin"], capture_output=True, text=True, check=True)
        return np.fromstring(result.stdout, sep="\n")
    except Exception as e:
        print(e)
        return np.array([])