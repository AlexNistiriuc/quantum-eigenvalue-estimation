import numpy as np

def eigendecompose(H: np.ndarray):
    # Returns eigenvalues (E) and eigenvectors (columns of V)
    E, V = np.linalg.eig(H)
    # normalize columns
    V = V / np.linalg.norm(V, axis=0)
    return E, V

def unitary_eigenphases(U: np.ndarray):
    # U eigen-decomposition and phases phi in [0,1)
    evals, evecs = np.linalg.eig(U)
    # angle -> phi
    phis = (np.angle(evals) / (2*np.pi)) % 1.0
    return phis, evecs