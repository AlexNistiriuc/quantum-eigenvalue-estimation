# main.py

import os
import time
import json
from contextlib import redirect_stdout
from analitical import analitical_minimum_energy
from vqe_runner import run_vqe
from vqe_graph import plot

def get_input():
    """
    Prompt user for molecule name and load its Hamiltonian JSON data.
    
    Returns:
        int: number of shots simulated.
        dict: Molecule data loaded from JSON.
    """

    while True:
        shots = input("\nHow many shots do you want to test? ")
        try:
            shots = int(shots)
            break
        except ValueError:
            print(f"Error: {shots} is not an integer value!.\n")

    name = input("Which molecule do you want to analyze? ")
    filename = os.path.join("..", "molecules", f"{name}_sto-3g_qubit_hamiltonian.json")
    print(f"Loading molecule data from: {filename}")

    while True:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                print(f"Molecule {name} loaded successfully!")
                return shots, json.load(file)
        except FileNotFoundError:
            print(f"Error: Data file for {name} not found.")
        except json.JSONDecodeError:
            print(f"Error: {filename} is not a valid JSON file.")


def main():    
    print('\n' + '='*18 + " STARTING VQE " + '='*18)

    shots, molecule_data = get_input()

    timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = os.path.join("..", "vqe_results", molecule_data["name"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, "vqe_log.txt"), 'w', encoding='utf-8')

    hamiltonian_dict = molecule_data['qubit_hamiltonian']
    number_of_qubits = len(list(hamiltonian_dict.keys())[0])
    num_spatial_orbitals = molecule_data['n_orbs']
    num_elec = molecule_data['n_elec']
    print(f"Data successfully prepared!\n" + '='*50)

    # Calculate analytical minimum
    print(f"....Calculating analitical energy....")
    start_time = time.time()
    min_energy = analitical_minimum_energy(hamiltonian_dict, number_of_qubits)
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
        print(f"Analytical minimum energy for {molecule_data['name']}: {min_energy} Hartree = {min_energy_eV} eV")
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
        plot(vqe_result.fun, elapsed_vqe, vqe_energies, min_energy, molecule_data['name'], output_dir, best_iteration)
    print(f"Log and plot saved in: {output_dir}")

    print('\n' + '='*19 + " ENDING VQE " + '='*19 + '\n')


if __name__ == "__main__":
    main()
