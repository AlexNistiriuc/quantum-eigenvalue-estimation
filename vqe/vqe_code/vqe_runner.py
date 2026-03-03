# vqe_runner.py

import numpy as np
from qiskit import ClassicalRegister
from .ansatz_factory import create_TwoLocal, create_UCCSD
from scipy.optimize import minimize
from .simulations import simulation

def run_vqe(H_dict, num_qubits, file, num_spatial_orbitals, num_elec, shots):
    """
    ansatz: Parametric circuit (TwoLocal)
    expectation_func: Function that calculates <H> given the circuit and parameters
    initial_params: Initial array of parameters
    """

    # Create ansatz
    if num_spatial_orbitals is None or num_elec is None:
        print("No molecular data available. Defaulting to TwoLocal ansatz.")
        ansatz = "1"
    else:
        while True:
            ansatz = input("Which ansatz do you wanna use?\n\t1. TwoLocal\n\t2. UCCSD\nAnswer: ")
            if ansatz in ["1", "2"]:
                break
            print(f"Error: {ansatz} is not a possible choise!\n")

    print(f"....Creating ansatz....")
    if ansatz=="1":
        ansatz = create_TwoLocal(file, num_qubits=num_qubits, reps=3)
        ansatz.add_register(ClassicalRegister(num_qubits, 'c'))
        initial_parameters = np.random.normal(0, 0.1, ansatz.num_parameters)
    elif ansatz=="2":
        ansatz = create_UCCSD(file, num_spatial_orbitals, num_elec)
        ansatz.add_register(ClassicalRegister(num_qubits, 'c'))
        initial_parameters = np.zeros(ansatz.num_parameters)    # da usare con UCCSD

    energies = []
    n_rep = 0
    best_cirq = None
    min_energy = float('inf')
    best_rep = -1
    
    def objective(params):
        nonlocal best_cirq, min_energy, best_rep, n_rep
        # Using the same ansatz, but with updated parameters
        parameterized_circuit = ansatz.assign_parameters(params)

        energy = simulation(parameterized_circuit, H_dict, shots)
        energies.append(energy)

        if energy < min_energy:
            best_cirq = parameterized_circuit
            min_energy = energy
            best_rep = n_rep
            print(f"\tIteration {n_rep} - Energy: {energy} - NEW BEST")
        else:
            print(f"\tIteration {n_rep} - Energy: {energy}")

        n_rep += 1

        return np.real(energy)
    
    
    print(f"....Starting simulations....")
    result = minimize(objective, initial_parameters, method='COBYLA', options={'maxiter': 2000})

    return result, energies, best_cirq, best_rep
