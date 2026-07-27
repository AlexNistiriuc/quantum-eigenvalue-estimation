# vqe_runner.py

import numpy as np
from qiskit import ClassicalRegister
from scipy.optimize import OptimizeResult, minimize
import warnings
from vqe.vqe_code.simulator_backend import build_simulator

from ansatz_factory import create_TwoLocal, create_UCCSD
from simulations import simulation, simulation_batch

def run_vqe(
    hamiltonian,
    num_qubits,
    file,
    num_spatial_orbitals,
    num_elec,
    shots,
    ansatz_type="twolocal",
    method="cobyla",
    spsa_a=0.2,
    spsa_c=0.1,
    spsa_alpha=0.602,
    spsa_gamma=0.101,
    spsa_stability_offset=None,
    maxiter=2000,
    two_local_reps=3,
    two_local_rotation_blocks=("ry",),
    two_local_entanglement="linear",
    two_local_entanglement_blocks=("cx",),
    two_local_parameter_prefix="theta",
    uccsd_reps=2,
    uccsd_generalized=False,
    uccsd_preserve_spin=True,
    uccsd_include_imaginary=True,
    seed=None,
    use_gpu=False,
    custatevec_enable=True,
    batched_shots_gpu=True,
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

    normalized_method = str(method).strip().lower()
    if normalized_method not in {"cobyla", "spsa"}:
        raise ValueError(f"Unknown optimization method '{method}'. Use one of: cobyla, spsa")

    if seed is not None:
        np.random.seed(int(seed))

    print("....Creating ansatz....")
    if normalized_ansatz == "twolocal":
        ansatz = create_TwoLocal(
            file,
            num_qubits=num_qubits,
            reps=two_local_reps,
            rotation_blocks=list(two_local_rotation_blocks),
            entanglement=two_local_entanglement,
            entanglement_blocks=list(two_local_entanglement_blocks),
            parameter_prefix=two_local_parameter_prefix,
        )
        ansatz.add_register(ClassicalRegister(num_qubits, "c"))
        initial_parameters = np.random.normal(0, 0.1, ansatz.num_parameters)
    else:
        ansatz = create_UCCSD(
            file,
            num_spatial_orbitals,
            num_elec,
            reps=uccsd_reps,
            generalized=uccsd_generalized,
            preserve_spin=uccsd_preserve_spin,
            include_imaginary=uccsd_include_imaginary,
        )
        ansatz.add_register(ClassicalRegister(num_qubits, "c"))
        initial_parameters = np.zeros(ansatz.num_parameters)

    energies = []
    n_rep = 0
    best_cirq = None
    min_energy = float('inf')
    best_rep = -1
    simulator = build_simulator(
            use_gpu=use_gpu,
            custatevec_enable=custatevec_enable,
            batched_shots_gpu=batched_shots_gpu,
        )
    
    def _to_real_energy(raw_energy):
        # Ensure energy is a real scalar (numerical noise can introduce tiny imaginary parts)
        raw_energy = complex(raw_energy)
        if abs(raw_energy.imag) > 1e-8:
            warnings.warn(f"Energy has non-negligible imaginary part: {raw_energy.imag}. Using real part.")
        return float(np.real(raw_energy))

    def evaluate(params, *, record=True, iteration_label=None):
        nonlocal best_cirq, min_energy, best_rep, n_rep
        # Using the same ansatz, but with updated parameters
        parameterized_circuit = ansatz.assign_parameters(params)

        energy = simulation(parameterized_circuit, hamiltonian, shots, simulator=simulator)
        energy = _to_real_energy(energy)
        if record:
            energies.append(energy)

            current_rep = n_rep if iteration_label is None else iteration_label
            if energy < min_energy:
                best_cirq = parameterized_circuit
                min_energy = energy
                best_rep = current_rep
                print(f"\tIteration {current_rep} - Energy: {energy} - NEW BEST")
            else:
                print(f"\tIteration {current_rep} - Energy: {energy}")

            n_rep += 1

        return energy

    def objective(params):
        return evaluate(params, record=True)

    def evaluate_pair(params_a, params_b):
        """
        Compute energies for two parameter sets (e.g. SPSA's plus/minus
        perturbations) in a single batched Aer job instead of two separate
        simulation() calls. Never recorded (mirrors the previous
        record=False behavior for plus/minus evaluations).
        """
        circuit_a = ansatz.assign_parameters(params_a)
        circuit_b = ansatz.assign_parameters(params_b)
        energy_a, energy_b = simulation_batch(
            [circuit_a, circuit_b], hamiltonian, shots, simulator=simulator
        )
        return _to_real_energy(energy_a), _to_real_energy(energy_b)

    def run_spsa(initial_params):
        current_params = np.array(initial_params, dtype=float, copy=True)
        current_energy = evaluate(current_params, record=True)

        a = float(spsa_a)
        c = float(spsa_c)
        alpha = float(spsa_alpha)
        gamma = float(spsa_gamma)
        stability_offset = (
            max(1.0, 0.1 * float(maxiter)) if spsa_stability_offset is None else float(spsa_stability_offset)
        )

        for iteration in range(int(maxiter)):
            ak = a / ((iteration + 1 + stability_offset) ** alpha)
            ck = c / ((iteration + 1) ** gamma)
            perturbation = np.random.choice([-1.0, 1.0], size=current_params.shape)

            params_plus = current_params + ck * perturbation
            params_minus = current_params - ck * perturbation
            energy_plus, energy_minus = evaluate_pair(params_plus, params_minus)
            gradient_estimate = ((energy_plus - energy_minus) / (2.0 * ck)) * perturbation

            candidate_params = current_params - ak * gradient_estimate
            current_energy = evaluate(candidate_params, record=True, iteration_label=iteration + 1)
            current_params = candidate_params

        return OptimizeResult(
            x=current_params,
            fun=current_energy,
            nit=int(maxiter),
            nfev=int(len(energies)),
            success=True,
            message="SPSA optimization completed.",
        )
    
    
    print(f"....Starting simulations....")
    if normalized_method == "spsa":
        result = run_spsa(initial_parameters)
    else:
        result = minimize(objective, initial_parameters, method="COBYLA", options={"maxiter": int(maxiter)})

    return result, energies, best_cirq, best_rep
