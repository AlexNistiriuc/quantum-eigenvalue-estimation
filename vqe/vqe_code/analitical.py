# analitical.py

import numpy as np

def pstr_to_matrix(pauli_str):
    """
    Convert a Pauli string to its corresponding matrix using tensor products.
    """
    matrix = np.array([1], dtype=complex)
    pauli_dict = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    for p in pauli_str:
        matrix = np.kron(matrix, pauli_dict[p])
    return matrix


def analitical_minimum_energy(H_dict, n):
    """
    Compute the analytical minimum energy for a qubit Hamiltonian.
    """
    H = np.zeros((2**n, 2**n), dtype=complex)
    for ps in H_dict:
        H += H_dict[ps] * pstr_to_matrix(ps)
    eigenvalues = np.linalg.eigh(H)[0]
    return eigenvalues[0]
