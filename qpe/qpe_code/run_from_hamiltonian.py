#!/usr/bin/env python3
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2] # repo root: quantum-vqe-simulator
sys.path.insert(0, str(repo_root))

import time
import argparse
import os
import numpy as np
from pathlib import Path
from qpe.qpe_code.io_utils import load_matrix_spec, build_unitary_from_hamiltonian
from qpe.qpe_code.matrix_utils import eigendecompose, unitary_eigenphases
from qpe.qpe_code.qpe_runner import run_qpe  
from qpe_graph import plot


def is_power_of_two(n):
    return (n & (n - 1) == 0) and n > 0

def main():
    parser = argparse.ArgumentParser(description="Run QPE from a Hamiltonian or Unitary JSON file.")
    parser.add_argument('--n', type=int, default=4, help='Number of phase (ancilla) qubits.')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots (measurements).')
    parser.add_argument('--psi-index', type=int, default=None, help='Override psi_index from file (computational basis).')
    parser.add_argument('--evec-index', type=int, default=0, help='Run QPE on the k-th eigenvector of H (after sorting eigenvalues).')
    parser.add_argument('--run-all-eigenstates', default=True, action='store_true', help='Run QPE for every eigenstate of H (slow for large dim).')
    parser.add_argument('--t', type=float, default=0.6, help='Evolution time t (overrides JSON t = 1.0 if provided).')
    parser.add_argument('--hbar', type=float, default=1, help='Reduced Planck constant (overrides JSON hbar = 1.0 if provided).')
    args = parser.parse_args()


    base_dir = Path(__file__).resolve().parent
    path = base_dir.parent.parent / "molecules" / "Hamiltonian_8x8_example.json"

    print(f"Loading matrix specification from: {path}")

    spec = load_matrix_spec(path)
    # override t and hbar from CLI if provided
    if args.t is not None:
        spec['t'] = float(args.t)
    if args.hbar is not None:
        spec['hbar'] = float(args.hbar)
    # if H present, recompute U with possibly overridden t/hbar
    if spec.get('kind') == 'hamiltonian' and 'H' in spec:
        spec['U'] = build_unitary_from_hamiltonian(spec['H'], spec['t'], spec.get('hbar', 1.0))
    dim = spec['dim']
    if not is_power_of_two(dim):
        raise ValueError(f"Dimension {dim} is not a power of two. QPE circuit expects 2^m dimension.")
    

    shots = args.shots
    n = args.n


    

    m = int(np.log2(dim))
    print(f"System qubits (m): {m} (dim={dim})")
    print(f"Phase qubits (n): {args.n}")

    U = spec['U']
    psi = spec['psi']

    #if file contains H, compute eigenpairs (eigenvalue E and eigenvector V), order them, copmute phases, and check consistency with U
    V = None
    E = None
    if spec.get('kind') == 'hamiltonian':
        H = spec['H']
        E, V = eigendecompose(H)
        # numeric safety: eigenvalues of Hermitian H should be real
        E = np.real(E)
        # sort ascending by energy and reorder eigenvectors accordingly
        sort_order = np.argsort(E)
        E = E[sort_order]
        V = V[:, sort_order]
        # compute phases phi = (-E * t) / (2*pi * hbar) mod 1
        t = spec.get('t', 1.0)
        h_bar = spec.get('hbar', 1.0)
        phis = ((-E * t) / (2 * np.pi * h_bar)) % 1.0

        print("Eigenvalues E_j and phases phi_j (phi = (-E t)/(2π) mod 1):")
        for j, (Ej, phj) in enumerate(zip(E, phis)):
            print(f"--- eigen {j} ---")
            print(f"eigenval {j}: {Ej:.6f}")
            print(f"eigenvect {j}: {np.round(V[:, j], 6)}")
            unitary_eigval = np.exp(-1j * t * Ej / h_bar)
            phase = np.angle(unitary_eigval)
            linear_phase = (phase / (2 * np.pi)) % 1.0
            print(f"> Unitary eigenvalue (from E): {unitary_eigval}")
            print(f"--> Phase [rad]: {phase:.6f}")
            print(f"--> Phase [degree]: {phase*180/np.pi:.1f}°")
            print(f"--> Phase [in units of 2pi]: {linear_phase:.6f}\n")

        # optional: eigen-decomposition of U for debug/consistency
        U_eigvals, U_evecs = np.linalg.eig(U)
        print("Eigenvalues of U (sample):", np.round(U_eigvals, 6))
        print("Absolute values (should be 1):", np.round(np.abs(U_eigvals), 6))

        # consistency check: U v_j ≈ exp(-i E_j t) v_j
        print("\nConsistency check: ||U v_j - exp(-i E_j t / hbar) v_j|| (should be ~0)")
        for j in range(len(E)):
            v = V[:, j]
            lhs = U @ v
            rhs = np.exp(-1j * t * E[j] / h_bar) * v
            resid = np.linalg.norm(lhs - rhs)
            print(f" j={j:2d}: resid={resid:.3e}")

        # compute simple t_max recommendation to avoid phase wrapping: t_max = 2π hbar / DeltaE_max
        DeltaE_max = E.max() - E.min() 
        t_max = (2.0 * np.pi * h_bar) / DeltaE_max
        print(f"\nSuggested t_max to avoid phase wrapping: DeltaE_max = {DeltaE_max:.6e}; t_max = {t_max:.6e} (hbar={h_bar})")
        if t >= t_max:
            print("WARNING: chosen t >= t_max -> some phases may wrap modulo 2π.")


    # Option: override psi by computational basis index
    if args.psi_index is not None:
        idx = args.psi_index
        if idx < 0 or idx >= dim:
            raise ValueError("psi-index out of range")
        psi = np.zeros(dim, dtype=np.complex128)
        psi[idx] = 1.0

    # Option: override psi by eigenvector index (requires H)
    if args.evec_index is not None:
        if spec.get('kind') != 'hamiltonian' or V is None:
            raise ValueError('--evec-index requires the input to be a Hamiltonian with computable eigenvectors')
        k = args.evec_index
        if k < 0 or k >= dim:
            raise ValueError('--evec-index out of range')
        psi = V[:, k]
        print(f"Using eigenvector index {k} as initial psi (normalized):")
        print(np.round(psi, 6))

    
    # If requested, run QPE for every eigenstate
    if args.run_all_eigenstates:
        if spec.get('kind') != 'hamiltonian' or V is None:
            raise ValueError('--run-all-eigenstates requires a Hamiltonian input')
        results = []
        for k in range(dim):
            psi_k = V[:, k]
            print(f"\n=== Running QPE for eigenstate {k} ===")

            # create output file path for this eigenstate and ensure parent dir exists
            timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
            base_dir = Path(__file__).resolve().parent
            print(f"Base directory for results: {base_dir}")
            output_file = base_dir.parent / "qpe_results" / "Hamiltonian_8x8" / f"{timestamp}_eigenstate_{k}.png"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            counts_k, circuit_k = run_qpe(psi_vector=psi_k, U=U, n=args.n, shots=args.shots)

            # compute estimated phase
            all_k = np.arange(0, 2**args.n)
            counts_array_k = np.zeros(2**args.n, dtype=int)
            for bitstring, c in counts_k.items():
                counts_array_k[int(bitstring, 2)] = c
            best_k = int(np.argmax(counts_array_k))
            phi_est = best_k / (2**args.n)
            print(f"Eigenstate {k}: estimated phase = {best_k}/{2**args.n} = {phi_est:.6f}")
            print(f"Counts: {counts_array_k}")
            results.append((k, best_k, phi_est, counts_array_k))

            # pass the computed all_k and save to the output_file path
            plot(counts_array_k, best_k=best_k, all_k=all_k, phi_est=phi_est, n=args.n, shots=args.shots, output_dir=str(output_file))
        # results contains tuples for each eigenstate if needed later
    else:
        # run QPE for the chosen psi (either default, psi-index or evec-index)
        counts, circuit = run_qpe(psi, U, n=args.n, shots=args.shots)

        # compute estimated phase from counts
        counts_array = np.zeros(2**args.n, dtype=int)
        for bitstring, c in counts.items():
            counts_array[int(bitstring, 2)] = c
        best_k = int(np.argmax(counts_array))
        phi_est = best_k / (2**args.n)
        print(f"Estimated phase: {best_k}/{2**args.n} = {phi_est:.6f}")
        print(f"Most frequent count: {counts_array[best_k]}/{args.shots}")
        try:
            print(circuit.draw(fold=80, output='text'))
        except Exception:
            # some backends/circuit objects may not support textual drawing in this environment
            pass

if __name__ == "__main__":
    main()