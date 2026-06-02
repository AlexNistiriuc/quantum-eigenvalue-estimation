#!/usr/bin/env python3
from pathlib import Path

import argparse
import os
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.linalg import expm

try:
    from .io_utils import load_json, build_unitary_from_hamiltonian
    from .qpe_runner import run_qpe
    from .qpe_graph import plot
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from qpe.qpe_code.io_utils import load_json, build_unitary_from_hamiltonian
    from qpe.qpe_code.qpe_runner import run_qpe
    from qpe.qpe_code.qpe_graph import plot


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_cwd():
    os.chdir(REPO_ROOT)


PAULIS = {
    'I': np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    'X': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    'Y': np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    'Z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def pauli_string_to_matrix(pstr: str) -> np.ndarray:
    # pstr is like 'IZXI' with length = number of qubits
    mats = [PAULIS[c] for c in pstr]
    op = mats[0]
    for m in mats[1:]:
        op = np.kron(op, m)
    return op


def build_h_from_qubit_hamiltonian(qh_dict):
    # qh_dict: mapping 'IXYZ' -> coeff
    # returns H (dense ndarray)
    if not qh_dict:
        raise ValueError('Empty qubit_hamiltonian')
    any_key = next(iter(qh_dict.keys()))
    n_qubits = len(any_key)
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    for pstr, coeff in qh_dict.items():
        coeff_c = complex(coeff)
        if len(pstr) != n_qubits:
            raise ValueError(f'Inconsistent Pauli string length: {pstr}')
        P = pauli_string_to_matrix(pstr)
        H += coeff_c * P
    return H


def hf_bitstring_from_nocc(n_qubits: int, n_occ: int) -> str:
    # default: occupy the lowest-index spin-orbitals (least significant bits)
    # we represent bitstring as q_{n-1}...q_0 (left to right)
    bits = ['0'] * n_qubits
    for i in range(n_occ):
        bits[-1 - i] = '1'
    return ''.join(bits)


def basis_index_from_bitstring(bitstr: str) -> int:
    # bitstr is like '1100' with leftmost = qubit n-1
    return int(bitstr, 2)


def trotterize_unitary_from_terms(qh_dict, t: float, hbar: float = 1.0, steps: int = 1):
    # First-order Trotter: U ≈ (∏_j exp(-i c_j P_j t/steps / hbar))^steps
    terms = []
    for pstr, coeff in qh_dict.items():
        P = pauli_string_to_matrix(pstr)
        terms.append((complex(coeff), P))

    # build single step
    dt = t / float(steps)
    U_step = np.eye(2 ** len(next(iter(qh_dict.keys()))), dtype=np.complex128)
    for (c, P) in terms:
        # compute the matrix exponential while silencing intermediate floating warnings
        U_term = expm(-1j * c * P * (dt / hbar))
        if not np.all(np.isfinite(U_term)):
            raise RuntimeError(f"expm non-finite for term {pstr} coeff={c}")

        U_step = U_term @ U_step
        if not np.all(np.isfinite(U_step)):
            raise RuntimeError(f"U_step non-finite after term {pstr} coeff={c}")
        # ri-ortogonalizza dopo ogni termine per contenere accumulo numerico
        Q, R = np.linalg.qr(U_step)
        phases = np.diag(R) / np.abs(np.diag(R))
        U_step = Q * phases

    # full U via repeated matrix multiplication (più stabile di matrix_power)
    U = np.eye(U_step.shape[0], dtype=np.complex128)
    for _ in range(steps):
        U = U_step @ U
        # ri-ortogonalizza U ad ogni step per contenere l'accumulo numerico
        Q, R = np.linalg.qr(U)
        phases = np.diag(R) / np.abs(np.diag(R)) #sistemo la fase del QR per mantenere unitarietà anche con errori numerici - test per accumulo di errori di fase e controllare warnings
        U = Q * phases  # preserva le fasi, forza unitarietà

    # diagnostica finale
    unitary_err = np.linalg.norm(U @ U.conj().T - np.eye(U.shape[0]))
    print(f'Trotter U | unitary check ||U U† - I||_F = {unitary_err:.2e}')
    return U


def main():
    _ensure_repo_root_cwd()
    parser = argparse.ArgumentParser(description='Run QPE from a molecule JSON (qubit_hamiltonian).')
    parser.add_argument('molecule', nargs='?', default=None, help='Path to molecule JSON (qubit_hamiltonian).')
    parser.add_argument('--n', type=int, default=5, help='Number of phase (ancilla) qubits.')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots.')
    parser.add_argument('--t', type=float, default=0.6, help='Evolution time t (overrides JSON).')
    parser.add_argument('--hbar', type=float, default=1, help='Reduced Planck constant (overrides JSON).')
    parser.add_argument('--trotter-steps', type=int, default=3, help='Number of trotter steps for first-order Trotterization.')
    parser.add_argument('--hf-bits', type=str, default='1001', help="Override Hartree-Fock bitstring (e.g. '1100'). If omitted, fills lowest orbitals.")
    parser.add_argument('--peak-window', type=int, default=2, help='Neighborhood radius around the peak to compute neighborhood averages (e.g. 1 includes peak±1).')
    parser.add_argument('--eigenstate-to-overlap', type=int, default=0, help="Index of the eigenstate (ordered by energy) you aim to overlap with (default=0 for ground state). Then the algorithm will report the best HF bitstring overlap with that eigenstate.")
    parser.add_argument('--vqe-asats', default=False, help='if set, take the psi state from the VQE ASATS output instead of the Hartree-Fock state. This is useful for testing how well QPE can refine a VQE state that is close to the target eigenstate.') #TBD
    parser.add_argument('--use-trotter', action='store_true', help='If set, use the Trotterized unitary instead of the exact matrix exponential for the QPE simulation.')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    mol_path = Path(args.molecule) if args.molecule else base_dir.parent.parent / 'molecules' / 'H2_sto-3g_qubit_hamiltonian.json'
    mol_path = mol_path.resolve()

    if not mol_path.exists():
        raise FileNotFoundError(f'Molecule JSON not found: {mol_path}')

    # results directory per-molecule
    results_root = base_dir.parent / 'qpe_results' / mol_path.stem
    if not results_root.exists():
        results_root.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
    main_log = results_root / f'{timestamp}_run.log'

    with open(main_log, 'w', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f'Loading molecule JSON from: {mol_path}')
            data = load_json(str(mol_path))

    # load overrides
    spec_t = args.t
    spec_hbar = args.hbar
    qh = data.get('qubit_hamiltonian')
    if qh is None:
        raise ValueError('molecule JSON must contain key "qubit_hamiltonian"')

    # infer n_qubits from any Pauli string
    any_key = next(iter(qh.keys()))
    n_qubits = len(any_key)
    dim = 2 ** n_qubits

    # build H (dense) from qubit_hamiltonian
    H = build_h_from_qubit_hamiltonian(qh)

    # Autovalori esatti di H
    eigvals, eigvecs = np.linalg.eigh(H)
    print("\n=== Autovalori esatti di H ===")
    for i, e in enumerate(eigvals):
        # phi = (-E * t) / (2π * ħ) mod 1
        phi_exact = ((-e * args.t) / (2 * np.pi * args.hbar)) % 1.0
        k_exact = phi_exact * (2 ** args.n)
        print(f"  E_{i} = {e:.6f} Ha  |  phi = {phi_exact:.4f}  |  k atteso = {k_exact:.2f}")

    # verify Hermiticity
    anti_norm = np.linalg.norm(H - H.conj().T)
    tol = 1e-12
    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f'Anti-Hermitian norm ||H - H^†||_F = {anti_norm:.3e}')
            if anti_norm > tol:
                raise ValueError(f'Hamiltonian is not Hermitian: ||H - H^†||_F = {anti_norm:.3e}.\n'
                                 'Please check the input qubit_hamiltonian or symmetrize the matrix before running.')

    # ensure all entries finite
    if not np.all(np.isfinite(H)):
        raise ValueError('Hamiltonian contains non-finite entries (NaN or Inf)')

    t = float(spec_t) if spec_t is not None else float(data.get('t', 1.0))
    hbar = float(spec_hbar) if spec_hbar is not None else float(data.get('hbar', 1.0))

    # build U via trotterization
    steps = max(1, int(args.trotter_steps))
    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f'Constructed H from qubit_hamiltonian (n_qubits={n_qubits}, dim={dim}).')
            print(f'Trotterizing with steps={steps}, t={t}, hbar={hbar}.')
    U = trotterize_unitary_from_terms(qh, t=t, hbar=hbar, steps=steps)

    # confronta U_trotter con U_esatto
    U_exact = expm(-1j * H * (t / hbar))
    trotter_err = np.linalg.norm(U - U_exact, ord='fro')
    print(f'Trotter error ||U_trotter - U_exact||_F = {trotter_err:.4e}')

    # construct Hartree-Fock computational basis state
    n_orbs = data.get('n_orbs')
    n_elec = data.get('n_elec')
    if n_orbs is None or n_elec is None:
        # fallback: try to infer number of occupied spin-orbitals from 'n_elec_total' or 'n_elec'
        total_elec = data.get('n_elec_total') or sum(data.get('n_elec', [0, 0]))
    else:
        total_elec = int(sum(n_elec))

    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f'Inferred total electrons = {total_elec} (from JSON n_elec={n_elec})')

    if args.hf_bits is not None:
        hf_bits = args.hf_bits
    else:
        # default: occupy lowest total_elec spin-orbitals
        hf_bits = hf_bitstring_from_nocc(n_qubits, total_elec)

    psi = np.zeros(dim, dtype=np.complex128)
    idx = basis_index_from_bitstring(hf_bits)
    psi[idx] = 1.0

    Upsi = U_exact @ psi
    print("\n=== Verifica fase analitica ===")
    print(f"  ||U|psi> - e^(i*phi)*|psi>|| per ogni autovalore:")

    # psi non è necessariamente autostato — decomponi nelle componenti
    for i, (e, v) in enumerate(zip(eigvals, eigvecs.T)):
        overlap = abs(np.dot(v.conj(), psi))**2
        if overlap > 1e-6:
            phi_exact = (e * t / (2 * np.pi)) % 1.0
            print(f"  E_{i} = {e:.6f} Ha | overlap = {overlap:.4f} | phi = {phi_exact:.4f} | k = {phi_exact * 2**args.n:.2f}")

    # verifica che psi sia normalizzato
    print(f"\n  ||psi|| = {np.linalg.norm(psi):.6f}")
    # verifica che U_exact sia unitaria
    uerr = np.linalg.norm(U_exact @ U_exact.conj().T - np.eye(dim))
    print(f"  ||U U† - I|| = {uerr:.2e}")
    # mostra su quali autostati si distribuisce U|psi>
    print(f"\n  Decomposizione di U|psi> sugli autostati:")
    for i, (e, v) in enumerate(zip(eigvals, eigvecs.T)):
        overlap_Upsi = abs(np.dot(v.conj(), Upsi))**2
        if overlap_Upsi > 1e-6:
            print(f"  E_{i} = {e:.6f} | overlap con U|psi> = {overlap_Upsi:.4f}")


    # Overlap dello stato HF con ogni autostato di H
    print(f"\n=== Overlap |<E_k|psi>|^2 per hf_bits='{hf_bits}' ===")
    for i, v in enumerate(eigvecs.T):
        overlap = abs(np.dot(v.conj(), psi))**2
        print(f"  E_{i} = {eigvals[i]:.6f} Ha  |  overlap = {overlap:.4f}")
    # Find best HF computational basis bitstring for a target eigenstate and compare with provided hf_bits
    target_idx = int(args.eigenstate_to_overlap) if hasattr(args, 'eigenstate_to_overlap') else 0
    if target_idx < 0 or target_idx >= len(eigvals):
        raise ValueError(f'--eigenstate-to-overlap must be between 0 and {len(eigvals)-1}')

    v_target = eigvecs[:, target_idx]
    # generate all computational basis bitstrings with correct number of electrons
    valid_indices = []
    valid_bitstrings = []
    for i_state in range(dim):
        b = format(i_state, f'0{n_qubits}b')
        if b.count('1') == total_elec:
            valid_indices.append(i_state)
            valid_bitstrings.append(b)

    probs = np.abs(v_target[valid_indices])**2
    best = int(np.argmax(probs))
    best_idx = valid_indices[best]
    best_bitstring = valid_bitstrings[best]
    best_prob = float(probs[best])

    user_idx = basis_index_from_bitstring(hf_bits)
    user_prob = float(np.abs(v_target[user_idx])**2)

    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f"\n=== Best HF basis state for eigenstate {target_idx} ===")
            print(f"Best bitstring: {best_bitstring} (index {best_idx}) with overlap {best_prob:.6f}")
            print(f"Provided HF bitstring: {hf_bits} (index {user_idx}) with overlap {user_prob:.6f}")
            if best_bitstring != hf_bits:
                print("NOTE: the provided HF bitstring differs from the best-overlap computational basis state.")
                if best_prob > user_prob:
                    print("Recommendation: consider using the best bitstring as initial psi (higher overlap with target eigenstate).")
                else:
                    print("Provided HF bitstring has equal or better overlap than the best computational basis candidate.")
            else:
                print("Provided HF bitstring matches the best computational-basis overlap for the target eigenstate.")
            


    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print(f'Hartree-Fock bitstring (leftmost = qubit n-1): {hf_bits}')
            print(f'Using computational-basis index: {idx}') #Calculated from hf_bits

    # run QPE single run
    with open(main_log, 'a', encoding='utf-8') as main_lf:
        with redirect_stdout(main_lf):
            print('\n=== Single run (molecule) ===')
            print(f'Output directory: {results_root}')
            print('Running QPE...')
            U_to_run = U if args.use_trotter else U_exact
            counts, circuit = run_qpe(psi, U_to_run, n=args.n, shots=args.shots)

            counts_array = np.zeros(2 ** args.n, dtype=int)
            for bitstring, c in counts.items():
                counts_array[int(bitstring, 2)] = c

            M = 2 ** args.n
            total_counts = counts_array.sum()
            if total_counts == 0:
                raise RuntimeError('No counts returned from QPE (total counts = 0)')

            # peak estimate (most frequent outcome)
            best_k = int(np.argmax(counts_array))
            phi_peak = best_k / M

            # global weighted average (classical weighted mean over all outcomes)
            ks = np.arange(M)
            phi_weighted_all = (np.dot(ks, counts_array) / total_counts) / M

            # neighborhood averages around peak if requested
            w = int(args.peak_window) if hasattr(args, 'peak_window') else 0
            if w > 0:
                idxs = np.array([(best_k + i) % M for i in range(-w, w + 1)], dtype=int)
                window_counts = counts_array[idxs]
                window_total = window_counts.sum()
                if window_total == 0:
                    phi_neighbor_weighted = float('nan')
                else:
                    phi_neighbor_weighted = (np.dot(idxs, window_counts) / window_total) / M
            else:
                phi_neighbor_weighted = None

            # choose phi_est used for plotting: pick the candidate closest to the expected phase
            # first compute expected phase(s) from eigendecomposition and overlaps
            overlaps = np.array([abs(np.dot(v.conj(), psi))**2 for v in eigvecs.T])
            # eigen-phases from eigenvalues: phi_k = (-E_k * t) / (2π hbar) mod 1
            phi_k = ((-eigvals * t) / (2 * np.pi * hbar)) % 1.0
            expected_phi_weighted = float(np.dot(overlaps, phi_k))

            # prepare candidate estimates (in [0,1))
            candidates = {
                'peak': float(phi_peak % 1.0),
                'global_weighted': float(phi_weighted_all % 1.0),
            }
            if phi_neighbor_weighted is not None:
                candidates['neighbor_weighted'] = float(phi_neighbor_weighted % 1.0)

            def circ_dist(a, b):
                d = abs(a - b) % 1.0
                return min(d, 1.0 - d)

            # find candidate closest to expected_phi_weighted
            best_name = None
            best_dist = float('inf')
            for name, val in candidates.items():
                d = circ_dist(val, expected_phi_weighted)
                if d < best_dist:
                    best_dist = d
                    best_name = name

            phi_est = candidates[best_name]

            # print comparisons
            print(f'Most frequent index (peak) = {best_k} -> phi_peak = {phi_peak:.6f}')
            print(f'Global weighted phi (all outcomes) = {phi_weighted_all:.6f}')
            if w > 0:
                print(f'Neighborhood (w={w}) weighted phi = {phi_neighbor_weighted:.6f}')
            print(f'Expected phase (overlap-weighted from H,psi) = {expected_phi_weighted:.6f}')
            print(f'Best estimate chosen = {best_name} -> phi = {phi_est:.6f} (circular dist = {best_dist:.6e})')
            print(f'Most frequent count: {counts_array[best_k]}/{args.shots}')
            try:
                print(circuit.draw(fold=80, output='text'))
            except Exception:
                pass

            # save histogram plot
            all_k = np.arange(0, 2 ** args.n)
            timestamp2 = time.strftime('%Y-%m-%d_%H.%M.%S')
            output_file = results_root / f'{timestamp2}_molecule_single_run.png'
            plot(counts_array, best_k=best_k, all_k=all_k, phi_est=phi_est, n=args.n, shots=args.shots, output_dir=str(output_file))

    print(f'Appended run info to: {main_log}')
    print(f'Saved plot: {output_file}')


if __name__ == '__main__':
    main()