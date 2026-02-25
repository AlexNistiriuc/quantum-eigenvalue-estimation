#!/usr/bin/env python3
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2] # repo root: quantum-vqe-simulator
sys.path.insert(0, str(repo_root))

import time
import argparse
import os
import numpy as np
from contextlib import redirect_stdout
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
    parser.add_argument('--evec-index', type=int, default=0, help='Run QPE on the k-th eigenvector of H (after sorting eigenvalues).')
    parser.add_argument('--run-all-eigenstates', default=False, action='store_true', help='Run QPE for every eigenstate of H (slow for large dim).')
    parser.add_argument('--psi-coefs', type=str, default="1,0,0,1,0,0,0,0", help='Override psi by a comma-separated list of complex coefficients (e.g. "1,0.5+0.5j,0,...").')
    parser.add_argument('--psi-eig', type=str, default="0,1,2,3,4,5,6,7", help='Override psi by a comma-separated list of eigenvector indices (e.g. "0,2" for the 1st and 3rd eigenvectors).')
    parser.add_argument('--t', type=float, default=0.6, help='Evolution time t (overrides JSON t = 1.0 if provided).')
    parser.add_argument('--hbar', type=float, default=1, help='Reduced Planck constant (overrides JSON hbar = 1.0 if provided).')
    args = parser.parse_args()


    base_dir = Path(__file__).resolve().parent
    path = base_dir.parent.parent / "molecules" / "Hamiltonian_8x8_example.json"

    # target results directory (do not create extra subdirectories)
    results_root = base_dir.parent / "qpe_results" / "Hamiltonian_8x8"
    if not results_root.exists():
        raise FileNotFoundError(f"Expected results directory does not exist: {results_root}. Please create it or adjust the path.")

    # create a timestamped main run log file in results_root and capture
    timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
    main_log = results_root / f"{timestamp}_run.log"
    # Capture loading and Hamiltonian analysis output into main_log
    with open(main_log, 'w', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
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
        # The detailed prints for the Hamiltonian analysis were captured in the
        # main_log above up to the spec load; continue writing analysis into
        # the same main_log file so everything about the Hamiltonian study is
        # in one document.
        with open(main_log, 'a', encoding='utf-8') as main_lf:
            with redirect_stdout(main_lf):
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
    
    # If requested, run QPE for every eigenstate
    if args.run_all_eigenstates:
        if spec.get('kind') != 'hamiltonian' or V is None:
            raise ValueError('--run-all-eigenstates requires a Hamiltonian input')
        results = []
        for k in range(dim):
            psi_k = V[:, k]
            print(f"\n=== Running QPE for eigenstate {k} ===")

            # create per-eigenstate filename for the PNG (no per-eigenstate log)
            timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
            output_file = results_root / f"{timestamp}_eigenstate_{k}.png"
            print(f"Writing plot to: {output_file.name} and appending run info to: {main_log.name}")

            # Append all prints (including those from run_qpe and plot) into the main_log
            with open(main_log, 'a', encoding='utf-8') as lf:
                with redirect_stdout(lf):
                    print(f"\n=== Eigenstate {k} run ===")
                    print(f"Output directory: {results_root}")
                    print(f"Using psi index: {k}")
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

                    # verification vs expected phase from E
                    try:
                        expected_phi = ((-E[k] * t) / (2 * np.pi * h_bar)) % 1.0
                        # circular difference
                        d = abs(phi_est - expected_phi)
                        print(f"Expected phase (from E): {expected_phi:.6f}")
                        print(f"Phase difference: {d:.6e}")
                        # energy estimate from measured phase
                        E_est = - (phi_est) * 2 * np.pi * h_bar / t
                        print(f"Estimated energy from phi: E_est = {E_est:.6e}")
                    except Exception as _:
                        print("Could not compute expected phase (missing E/t/hbar)")

                    # write plot to file in results_root
                    plot(counts_array_k, best_k=best_k, all_k=all_k, phi_est=phi_est, n=args.n, shots=args.shots, output_dir=str(output_file))

            # brief terminal summary pointing to the saved plot and that main_log was appended
            print(f"Appended run info to: {main_log}")
            print(f"Saved plot: {output_file}\n")
        # results contains tuples for each eigenstate if needed later
    else:
        # Single run: append details to main_log and save the PNG in results_root
        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        output_file = results_root / f"{timestamp}_single_run.png"
        with open(main_log, 'a', encoding='utf-8') as lf:
            with redirect_stdout(lf):
                print("=== Single run ===")
                print(f"Output directory: {results_root}")
                if args.psi_coefs is not None and args.psi_eig is not None:
                    print(f"Overriding psi with custom coefficients: {args.psi_coefs} for eigenvector indices: {args.psi_eig} ")
                    psi_coefs = [complex(c.strip()) for c in args.psi_coefs.split(',')]
                    psi_eig_indices = [int(i.strip()) for i in args.psi_eig.split(',')]
                    if len(psi_coefs) != len(psi_eig_indices):
                        raise ValueError("Length of psi-coefs and psi-eig must match")
                    if len(psi_coefs) != dim:
                            print(f"ERROR: --psi-coefs must contain exactly {dim} coefficients (one per eigenvector) when --psi-eig is not used.")
                            print("Example: --psi-coefs 'c0,c1,...,c{dim-1}'")
                            raise ValueError("Wrong number of coefficients for --psi-coefs")
                    psi = np.zeros(dim, dtype=np.complex128)
                    for coef, eig_idx in zip(psi_coefs, psi_eig_indices):
                        psi += coef * V[:, eig_idx] # I take all the values on the column eig_idx
                    
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

                # save plot into results_root
                all_k = np.arange(0, 2**args.n)
                plot(counts_array, best_k=best_k, all_k=all_k, phi_est=phi_est, n=args.n, shots=args.shots, output_dir=str(output_file))

        print(f"Appended run info to: {main_log}")
        print(f"Saved plot: {output_file}")

if __name__ == "__main__":
    main()