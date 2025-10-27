import numpy as np
from typing import Optional, Tuple

def build_diag_unitary(n_qubits: int,
                       phases: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a diagonal unitary U (2^n x 2^n).
    By default phases are k / 2^n for k=0..2^n-1, so eigenvalues = exp(2pi*i*k/2^n).

    Args:
        n_qubits: number of system qubits
        phases: optional 1-d array of length 2^n with phases in [0,1).
                phase k corresponds to eigenvalue exp(2pi i * phase[k]).

    Returns:
        U: (2^n, 2^n) numpy.ndarray unitary (diagonal)
        phases_used: numpy array of phases in [0,1)
    """
    dim = 2 ** n_qubits
    if phases is None:
        phases = np.arange(dim) / float(dim)   # k/2^n
    phases = np.array(phases, dtype=float)
    assert phases.shape == (dim,)
    eigvals = np.exp(2j * np.pi * phases)
    U = np.diag(eigvals)
    return U, phases

def print_eigen_decomposition(U: np.ndarray):
    """
    Diagonalizes a unitary U and prints eigenvalues as phases plus the eigenstates.
    Eigenstates are given as computational-basis vectors if U is diagonal.

    Args:
        U: numpy.ndarray unitary (d x d)
    """
    n_bits =int(np.ceil(np.log2(len(U))))
    evals, evecs = np.linalg.eig(U)

    print("Eigen-decomposition of U:")
    for i, (lam, vec) in enumerate(zip(evals, evecs.T)):
        # phase in [0,1)
        phi = (np.angle(lam) / (2*np.pi)) % 1
        # find the computational basis index with largest amplitude
        basis_index = np.argmax(np.abs(vec))
        # normalize eigenvector to avoid floating noise
        vec_norm = vec / np.linalg.norm(vec)

        formatter = f'0{n_bits}b'

        print(f"Eigenvalue {i}: λ = {lam:.3f}, phase φ = {phi:.4f}")
        print(f"   Dominant basis state ≈ |{format(basis_index, formatter)}>")
        print(f"   Eigenvector (amplitudes): {np.round(vec_norm, 3)}\n")

