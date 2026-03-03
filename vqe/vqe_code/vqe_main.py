# main.py

import os
import time
import json
import numpy as np
from itertools import product
from contextlib import redirect_stdout
from pathlib import Path

try:
    from .analitical import analitical_minimum_energy
    from .vqe_runner import run_vqe
    from .vqe_graph import plot
except ImportError:
    from analitical import analitical_minimum_energy
    from vqe_runner import run_vqe
    from vqe_graph import plot


BASE_DIR = Path(__file__).resolve().parents[1]
MOLECULES_DIR = BASE_DIR / "molecules"
RESULTS_DIR = BASE_DIR / "vqe_results"

def get_input():
    """
    Prompt user for molecule name and load its Hamiltonian JSON data.
    
    Returns:
        int: number of shots simulated.
        dict: Molecule data loaded from JSON.
    """

    name = input("Which molecule do you want to analyze? ")
    filename = MOLECULES_DIR / f"{name}_sto-3g_qubit_hamiltonian.json"
    print(f"Loading molecule data from: {filename}")

    while True:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                print(f"Molecule {name} loaded successfully!")
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Data file for {name} not found.")
        except json.JSONDecodeError:
            print(f"Error: {filename} is not a valid JSON file.")

def hermitianEmbedding(A):
    """
    Embed a non-Hermitian matrix A into a Hermitian matrix H.
    
    Args:
        A (np.ndarray): Input non-Hermitian matrix.
    Returns:
        np.ndarray: Hermitian matrix embedding A.
    """
    n = A.shape[0]
    H = np.zeros((2*n, 2*n), dtype=complex)
    H[:n, n:] = A
    H[n:, :n] = A.conj().T
    return H

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

def pauliDecomposition(H):
    """
    Decompose a Hermitian matrix H into a sum of Pauli strings.
    
    Args:
        H (np.ndarray): Input Hermitian matrix.
    Returns:
        dict: Dictionary mapping Pauli strings to their coefficients.
    """

    n = int(np.log2(H.shape[0]))
    pauli_strings = [''.join(p) for p in product('IXYZ', repeat=n)]
    H_dict = {}
    for ps in pauli_strings:
        coeff = np.trace(np.matmul(H, pstr_to_matrix(ps))) / (2**n)
        if abs(coeff) > 1e-10:  # Filter out negligible coefficients
            H_dict[ps] = coeff
    return H_dict


def main():
    print('\n' + '='*18 + " STARTING VQE " + '='*18)

    while True:
        shots = input("\nHow many shots do you want to test? ")
        try:
            shots = int(shots)
            break
        except ValueError:
            print(f"Error: {shots} is not an integer value!.\n")

    risp = input('Do you want to load a molecule (1) or a matrix (2)? Answer: ')
    if risp == '1':
        molecule_data = get_input()

        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = RESULTS_DIR / molecule_data["name"] / timestamp
        os.makedirs(output_dir, exist_ok=True)
        f = open(output_dir / "vqe_log.txt", 'w', encoding='utf-8')

        name = molecule_data['name']
        hamiltonian_dict = molecule_data['qubit_hamiltonian']
        number_of_qubits = len(list(hamiltonian_dict.keys())[0])
        num_spatial_orbitals = molecule_data['n_orbs']
        num_elec = molecule_data['n_elec']
    
    else:
        filename = MOLECULES_DIR / "Hamiltonian_8x8_example.json"
        with open(filename, 'r', encoding='utf-8') as file:
            print(f"Matrix loaded successfully!")
            matrix_data = json.load(file)

        A = np.array(matrix_data["A"], dtype=complex)

        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = RESULTS_DIR / "custom_matrix" / timestamp
        os.makedirs(output_dir, exist_ok=True)
        f = open(output_dir / "vqe_log.txt", 'w', encoding='utf-8')

        H = hermitianEmbedding(A)
        hamiltonian_dict = pauliDecomposition(H)

        name = "Matrice di prova 8x8"
        number_of_qubits = int(np.log2(H.shape[0]))
        num_spatial_orbitals = None
        num_elec = None        
    
    print(f"Data successfully prepared!\n" + '='*50)

    # Calculate analytical minimum
    print(f"....Calculating analitical energy....")
    start_time = time.time()
    min_energy, max_energy, H = analitical_minimum_energy(hamiltonian_dict, number_of_qubits)
    end_time = time.time()
    print(f"Minimum (analitical) energy level: {min_energy}\n" + '='*50)
    elapsed_analitical = end_time - start_time

    # Simulating VQE
    start_time = time.time()
    vqe_result, vqe_energies, best_circuit, best_iteration = run_vqe(hamiltonian_dict, number_of_qubits, f, num_spatial_orbitals, num_elec, shots=shots)
    end_time = time.time()
    print(f"....Ending simulations....\n" + '='*50)
    elapsed_vqe = end_time - start_time

    Hartree_eV = 27.211386245988
    min_energy_eV = min_energy * Hartree_eV
    vqe_result_eV = vqe_result.fun * Hartree_eV
    energy_diff = vqe_result.fun - min_energy
    energy_diff_eV = energy_diff * Hartree_eV
    percentage = (-energy_diff/min_energy)*100
    decomposed_circuit = best_circuit.decompose()

    with redirect_stdout(f):
        # Print results
        print(f"Hamiltonian: {hamiltonian_dict}")
        print('='*50)
        print(f"Analytical minimum energy for {name}: {min_energy} Hartree = {min_energy_eV} eV")
        print(f"Elapsed time: {elapsed_analitical:.4f} s")
        print('='*50)
        print(f"VQE Energy: {vqe_result.fun} Hartree = {vqe_result_eV} eV")
        print(f"Error: {energy_diff} Hartree = {energy_diff_eV} eV --> {percentage}%")
        print(f"Shots: {shots}")
        print(f"Time elapsed: {elapsed_vqe:.4f} s")
        print(f"Number of iterations: {len(vqe_energies)}")
        print(f"Energy list: {vqe_energies[:10]} ...")
        print('='*50)
        print(f"Best circuit (iteration {best_iteration}):")
        print(decomposed_circuit.draw(fold=60, output='text'))

        # Plot convergence
        plot(vqe_result.fun, elapsed_vqe, vqe_energies, min_energy, name, output_dir, best_iteration)
    print(f"Log and plot saved in: {output_dir}")

    print('\n' + '='*19 + " ENDING VQE " + '='*19 + '\n')

    S = max_energy - min_energy

    return H, vqe_result, best_circuit, S


if __name__ == "__main__":
    main()
