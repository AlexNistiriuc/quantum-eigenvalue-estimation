# simulations.py

import qiskit as qk
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from simulator_backend import build_simulator

def validate_pauli_string(pauli_str, qc):
    """
    Ensure the Pauli string is valid and matches circuit qubits.
    """
    if len(pauli_str) != len(qc.qubits):
        raise ValueError(f"Pauli string {pauli_str} length mismatch with circuit.")
    for op in pauli_str:
        if op not in ['I', 'X', 'Y', 'Z']:
            raise ValueError(f"Invalid Pauli operator {op}.")


def apply_rotations_pauli_string(qc, pauli_str):

    """
    Apply gates to rotate qubits to Z-basis for measurement.
    """
    for i, pauli in enumerate(pauli_str):
        if pauli == 'I' or pauli == 'Z':
            continue  # No rotation needed
        elif pauli == 'X':
            qc.h(qc.qubits[i])  # Rotate X to Z
        elif pauli == 'Y':
            qc.sdg(qc.qubits[i])  # S†
            qc.h(qc.qubits[i])    # Rotate Y to Z


def measure_pauli_string(qc, pauli_str):
    """
    Add measurement operations for the Pauli string in Z basis.
    """
    for i, pauli in enumerate(pauli_str):
        qc.measure(qc.qubits[i], qc.clbits[i])


def can_reuse_measurements(pauli_str, stored_strings):
    """
    Determine if measurement results of a previous string can be reused.
    """
    for string in stored_strings:
        reusable = True
        for a, b in zip(pauli_str, string):
            rotation_a = 'none' if a in ['I', 'Z'] else ('H' if a == 'X' else 'SH')
            rotation_b = 'none' if b in ['I', 'Z'] else ('H' if b == 'X' else 'SH')
            if rotation_a != rotation_b:
                reusable = False
                break
        if reusable:
            return string
    return None


def build_measurement_circuit(qc, pauli_str):
    """
    Build the measurement circuit (basis rotations + measurement) for a Pauli string.
    """
    temp_qc = qc.copy()
    apply_rotations_pauli_string(temp_qc, pauli_str)
    temp_qc.barrier()
    measure_pauli_string(temp_qc, pauli_str)
    return temp_qc


def run_batched_circuits(circuits, shots, simulator):
    """
    Transpile and run a batch of circuits in a single Aer job.

    Submitting every circuit needed for one energy evaluation together (instead of
    one simulator.run() call per Pauli string) amortizes per-job overhead and lets
    the GPU backend process the whole batch at once (especially with
    batched_shots_gpu=True).

    Returns a list of count dicts, one per input circuit, in the same order.
    """
    if not circuits:
        return []
    transpiled = qk.transpile(circuits, simulator)
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    return [result.get_counts(i) for i in range(len(circuits))]


def expectation_value(pauli_str, counts, shots):
    """
    Calculate the expectation value for a Pauli string.
    """
    total = 0
    for outcome, count in counts.items():
        outcome_rev = outcome[::-1]
        eigenvalue = 1
        for i, bit in enumerate(outcome_rev):
            if pauli_str[i] != 'I' and bit == '1':
                eigenvalue *= -1
        total += count * eigenvalue
    return total / shots


def not_optimized(qc, pauli_strings, shots, simulator):
    """
    Not-optimized expectation value computation (no measurement reuse), but still
    batches all required circuits into a single Aer job.

    Args:
        qc: Quantum circuit.
        pauli_strings: List of Pauli strings.
        shots: Number of shots.
        simulator: Quantum simulator instance.
    """
    results = {}
    non_identity = []

    for pauli_str in pauli_strings:
        validate_pauli_string(pauli_str=pauli_str, qc=qc)

        # If all-I string set result = 1
        if all(c == 'I' for c in pauli_str):
            results[pauli_str] = 1
            continue

        non_identity.append(pauli_str)

    circuits = [build_measurement_circuit(qc, ps) for ps in non_identity]
    counts_list = run_batched_circuits(circuits, shots, simulator)

    for pauli_str, counts in zip(non_identity, counts_list):
        results[pauli_str] = expectation_value(pauli_str, counts, shots)

    return results


def group_by_measurement_basis(pauli_strings):
    """
    Split Pauli strings into identity-only results plus groups that share a
    measurement basis (i.e. can reuse the same circuit/counts).

    Returns:
        results: dict mapping all-identity Pauli strings to expectation value 1.
        group_bases: one representative Pauli string per group.
        group_members: list of lists; group_members[i] all reuse group_bases[i].
    """
    results = {}
    non_identity = []

    for ps in pauli_strings:
        if all(c == 'I' for c in ps):
            results[ps] = 1
            continue
        non_identity.append(ps)

    group_bases = []
    group_members = []
    for ps in non_identity:
        base = can_reuse_measurements(ps, group_bases)
        if base is not None:
            group_members[group_bases.index(base)].append(ps)
        else:
            group_bases.append(ps)
            group_members.append([ps])

    return results, group_bases, group_members


def optimized(qc, pauli_strings, shots, simulator):
    """
    Optimized expectation value computation using reusable measurements.

    Pauli strings sharing the same measurement basis are grouped together (one
    circuit per group, as before), but instead of running one simulator job per
    group sequentially, every group's circuit is built up front and submitted to
    the backend in a single batched simulator.run() call.

    Args:
        qc: Quantum circuit.
        pauli_strings: List of Pauli strings.
        shots: Number of shots.
        simulator: Quantum simulator instance.
    """
    for ps in pauli_strings:
        validate_pauli_string(ps, qc)

    results, group_bases, group_members = group_by_measurement_basis(pauli_strings)

    circuits = [build_measurement_circuit(qc, base) for base in group_bases]
    counts_list = run_batched_circuits(circuits, shots, simulator)

    for members, counts in zip(group_members, counts_list):
        for ps in members:
            results[ps] = expectation_value(ps, counts, shots)

    return results


def simulation_batch(qcs, hamiltonian, shots, simulator):
    """
    Compute the expected energy for the same Hamiltonian across several circuits
    (e.g. SPSA's params_plus/params_minus) using a single batched Aer job.

    Each circuit still needs its own measurement circuits (rotations depend only
    on the Pauli string, but counts depend on the specific circuit/state), so no
    counts are shared *across* circuits — the win here is submitting every
    measurement circuit for every input circuit in one simulator.run() call
    instead of one call per circuit (which is what looping simulation() would do).

    Args:
        qcs: List of quantum circuits (same structure/Hamiltonian, different params).
        hamiltonian: Hamiltonian (dict of Pauli strings or matrix).
        shots: Number of shots.
        simulator: Quantum simulator instance.

    Returns:
        List of energies, one per input circuit, same order as qcs.
    """
    if isinstance(hamiltonian, dict):
        pauli_strings = list(hamiltonian.keys())

        all_circuits = []
        per_qc_layout = []  # (results_seed, group_members, n_groups) per qc

        for qc in qcs:
            for ps in pauli_strings:
                validate_pauli_string(ps, qc)
            results_seed, group_bases, group_members = group_by_measurement_basis(pauli_strings)
            for base in group_bases:
                all_circuits.append(build_measurement_circuit(qc, base))
            per_qc_layout.append((results_seed, group_members))

        counts_list = run_batched_circuits(all_circuits, shots, simulator)

        energies = []
        ptr = 0
        for results, group_members in per_qc_layout:
            for members in group_members:
                counts = counts_list[ptr]
                ptr += 1
                for ps in members:
                    results[ps] = expectation_value(ps, counts, shots)
            energies.append(sum(hamiltonian[ps] * results[ps] for ps in hamiltonian))
        return energies

    H = np.asarray(hamiltonian, dtype=np.complex128)
    return [Statevector.from_instruction(qc).expectation_value(Operator(H)) for qc in qcs]


def simulation(qc, hamiltonian, shots, simulator):
    """
    Compute expected energy for given Hamiltonian using the quantum circuit.
    
    Args:
        qc: Quantum circuit.
        hamiltonian: Hamiltonian (dict of Pauli strings or matrix).
        shots: Number of shots.
        use_gpu (bool): If True, use GPU backend; if False, use CPU (default).
        custatevec_enable (bool): Enable cuStateVec optimization on GPU (default: True).
        batched_shots_gpu (bool): Enable batched shots on GPU (default: True).
    """
    if isinstance(hamiltonian, dict):
        averages = optimized(qc, list(hamiltonian.keys()), shots, simulator)
        # averages = not_optimized(qc, list(hamiltonian.keys()), shots, use_gpu=use_gpu, ...)
        energy = sum(hamiltonian[ps] * averages[ps] for ps in hamiltonian)
        return energy

    H = np.asarray(hamiltonian, dtype=np.complex128)
    psi = Statevector.from_instruction(qc)
    return psi.expectation_value(Operator(H))
