# simulations.py

import qiskit as qk
from qiskit_aer import AerSimulator

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


def get_counts_for_pauli(qc, pauli_str, shots):
    """
    Simulate the circuit and return measurement counts.
    """
    temp_qc = qc.copy()
    apply_rotations_pauli_string(temp_qc, pauli_str)
    temp_qc.barrier()
    measure_pauli_string(temp_qc, pauli_str)

    simulator = AerSimulator()
    transpiled_qc = qk.transpile(temp_qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    return result.get_counts()


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


def not_optimized(qc, pauli_strings, shots):
    """
    Not-optimized expectation value computation using reusable measurements.
    """
    results = {}

    for pauli_str in pauli_strings:
        validate_pauli_string(pauli_str=pauli_str, qc=qc)

        # If all-I string set result = 1
        if all(c == 'I' for c in pauli_str):
            results[pauli_str] = 1
            continue
        
        counts = get_counts_for_pauli(qc, pauli_str, shots=shots)
        results[pauli_str] = expectation_value(pauli_str, counts, shots)

    return results


def optimized(qc, pauli_strings, shots):
    """
    Optimized expectation value computation using reusable measurements.
    """
    stored_counts = {}
    results = {}

    for ps in pauli_strings:
        validate_pauli_string(ps, qc)
        if all(c == 'I' for c in ps):
            results[ps] = 1
            continue

        base = can_reuse_measurements(ps, stored_counts.keys())
        if base:
            results[ps] = expectation_value(ps, stored_counts[base], shots)
        else:
            counts = get_counts_for_pauli(qc, ps, shots)
            stored_counts[ps] = counts
            results[ps] = expectation_value(ps, counts, shots)

    return results


def simulation(qc, H_dict, shots):
    """
    Compute expected energy for given Hamiltonian using the quantum circuit.
    """
    averages = optimized(qc, list(H_dict.keys()), shots)
    #averages = not_optimized(qc, list(H_dict.keys()), shots)
    energy = sum(H_dict[ps] * averages[ps] for ps in H_dict)
    return energy
