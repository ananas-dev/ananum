import numpy as np

def save_matrix(filename: str, matrix: np.ndarray):
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    
    n = np.uint32(matrix.shape[0])
    
    with open(filename, "wb") as f:
        f.write(n.tobytes())
        f.write(matrix.astype(np.float64).tobytes())