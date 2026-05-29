# vqe_runner.py

import numpy as np
from qiskit import ClassicalRegister
from scipy.optimize import minimize
import warnings

try:
    from .ansatz_factory import create_TwoLocal, create_UCCSD
    from .simulations import simulation
except ImportError:
    from ansatz_factory import create_TwoLocal, create_UCCSD
    from simulations import simulation

def run_vqe(
    hamiltonian,
    num_qubits,
    file,
    num_spatial_orbitals,
    num_elec,
    shots,
    ansatz_type="twolocal",
    maxiter=2000,
    two_local_reps=3,
    seed=None,
):
    """
    ansatz: Parametric circuit (TwoLocal)
    expectation_func: Function that calculates <H> given the circuit and parameters
    initial_params: Initial array of parameters
    """

    normalized_ansatz = str(ansatz_type).strip().lower()
    if normalized_ansatz in {"1", "twolocal", "two_local"}:
        normalized_ansatz = "twolocal"
    elif normalized_ansatz in {"2", "uccsd"}:
        normalized_ansatz = "uccsd"
    else:
        raise ValueError(
            f"Unknown ansatz_type '{ansatz_type}'. Use one of: twolocal, uccsd, 1, 2"
        )

    if seed is not None:
        np.random.seed(int(seed))

    print("....Creating ansatz....")
    if normalized_ansatz == "twolocal":
        ansatz = create_TwoLocal(file, num_qubits=num_qubits, reps=two_local_reps)
        ansatz.add_register(ClassicalRegister(num_qubits, "c"))
        initial_parameters = np.random.normal(0, 0.1, ansatz.num_parameters)
    else:
        ansatz = create_UCCSD(file, num_spatial_orbitals, num_elec)
        ansatz.add_register(ClassicalRegister(num_qubits, "c"))
        initial_parameters = np.zeros(ansatz.num_parameters)

    energies = []
    n_rep = 0
    best_cirq = None
    min_energy = float('inf')
    best_rep = -1
    
    def objective(params):
        nonlocal best_cirq, min_energy, best_rep, n_rep
        # Using the same ansatz, but with updated parameters
        parameterized_circuit = ansatz.assign_parameters(params)

        energy = simulation(parameterized_circuit, hamiltonian, shots)
        # Ensure energy is a real scalar (numerical noise can introduce tiny imaginary parts)
        energy = complex(energy)
        if abs(energy.imag) > 1e-8:
            warnings.warn(f"Energy has non-negligible imaginary part: {energy.imag}. Using real part.")
        energy = float(np.real(energy))
        energies.append(energy)

        if energy < min_energy:
            best_cirq = parameterized_circuit
            min_energy = energy
            best_rep = n_rep
            print(f"\tIteration {n_rep} - Energy: {energy} - NEW BEST")
        else:
            print(f"\tIteration {n_rep} - Energy: {energy}")

        n_rep += 1

        return energy
    
    
    print(f"....Starting simulations....")
    result = minimize(objective, initial_parameters, method="COBYLA", options={"maxiter": int(maxiter)})

    return result, energies, best_cirq, best_rep
