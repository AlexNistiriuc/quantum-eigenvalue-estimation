#!/usr/bin/env python3
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
from scipy.linalg import expm
from qpe.qpe_code.qpe_runner import run_qpe
from qpe.qpe_code.io_utils import load_json

PAULIS = {
    'I': np.array([[1,0],[0,1]], dtype=np.complex128),
    'X': np.array([[0,1],[1,0]], dtype=np.complex128),
    'Y': np.array([[0,-1j],[1j,0]], dtype=np.complex128),
    'Z': np.array([[1,0],[0,-1]], dtype=np.complex128),
}

def pauli_string_to_matrix(pstr):
    mats = [PAULIS[c] for c in pstr]
    op = mats[0]
    for m in mats[1:]:
        op = np.kron(op, m)
    return op

def build_h(qh_dict):
    n_qubits = len(next(iter(qh_dict.keys())))
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    for pstr, coeff in qh_dict.items():
        H += complex(coeff) * pauli_string_to_matrix(pstr)
    return H

def build_unitary_trotter(qh_dict, t, hbar=1.0, steps=1):
    n_qubits = len(next(iter(qh_dict.keys())))
    dim = 2 ** n_qubits
    identity_key = 'I' * n_qubits

    # separa fase globale
    global_coeff = complex(qh_dict.get(identity_key, 0.0))
    global_phase = np.exp(-1j * global_coeff * (t / hbar))
    print(f"  Fase globale (IIII): coeff={global_coeff:.6f}, phase=e^(i*{np.angle(global_phase):.4f})")

    # termini senza identità
    qh_reduced = {k: v for k, v in qh_dict.items() if k != identity_key}

    dt = t / float(steps)
    U_step = np.eye(dim, dtype=np.complex128)

    print(f"\n  --- Diagnostica per-termine (dt={dt:.4f}) ---")
    for pstr, coeff in qh_reduced.items():
        c = complex(coeff)
        P = pauli_string_to_matrix(pstr)

        arg = -1j * c * P * (dt / hbar)
        arg_norm = np.linalg.norm(arg, ord='fro')
        print(f"\n  Termine {pstr}: coeff={c:.6f}")
        print(f"    ||arg||_F = {arg_norm:.6f}")
        print(f"    arg finite = {np.all(np.isfinite(arg))}")

        U_term = expm(arg)
        u_term_norm = np.linalg.norm(U_term, ord='fro')
        u_term_finite = np.all(np.isfinite(U_term))
        print(f"    ||U_term||_F = {u_term_norm:.6f}, finite = {u_term_finite}")

        if not u_term_finite:
            raise RuntimeError(f"expm non-finite per termine {pstr} coeff={c}")

        U_step = U_term @ U_step
        u_step_norm = np.linalg.norm(U_step, ord='fro')
        u_step_finite = np.all(np.isfinite(U_step))
        print(f"    ||U_step||_F = {u_step_norm:.6f}, finite = {u_step_finite}")

        if not u_step_finite:
            raise RuntimeError(f"U_step non-finite dopo termine {pstr}")

    print(f"\n  --- Fine diagnostica ---\n")

    # applica steps
    U = np.eye(dim, dtype=np.complex128)
    for s in range(steps):
        U = U_step @ U
        u_finite = np.all(np.isfinite(U))
        print(f"  Step {s+1}/{steps}: ||U||_F={np.linalg.norm(U):.4f}, finite={u_finite}")
        if not u_finite:
            raise RuntimeError(f"U non-finite dopo step {s+1}")

    # rimetti la fase globale
    U = global_phase * U

    err = np.linalg.norm(U @ U.conj().T - np.eye(dim))
    print(f"  Unitary check ||UU†-I|| = {err:.2e}")
    return U

def run_and_print(label, psi, U, n=4, shots=1024):
    counts, _ = run_qpe(psi, U, n=n, shots=shots)
    arr = np.zeros(2**n, dtype=int)
    for bitstring, c in counts.items():
        arr[int(bitstring, 2)] = c
    best_k = int(np.argmax(arr))
    phi = best_k / (2**n)
    print(f"  QPE misura: k={best_k}, phi={phi:.4f} | {arr[best_k]}/{shots} shots")
    return best_k, phi


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── carica molecola ────────────────────────────────────────────
mol_path = Path(__file__).resolve().parents[1] / 'molecules' / 'H2_sto-3g_qubit_hamiltonian.json'
data = load_json(str(mol_path))
qh = data['qubit_hamiltonian']
t = 0.6
n_qubits = len(next(iter(qh.keys())))
dim = 2 ** n_qubits

H = build_h(qh)
eigvals, eigvecs = np.linalg.eigh(H)

print("\n=== Autovalori di H ===")
for i, e in enumerate(eigvals):
    phi_e = (e * t / (2 * np.pi)) % 1.0
    print(f"  E_{i:2d} = {e:.6f} Ha | phi = {phi_e:.4f} | k atteso (n=4) = {phi_e*16:.2f}")


# ── TEST 1 ─────────────────────────────────────────────────────
separator("TEST 1 — fase sintetica phi=0.75 (k atteso=12)")
phi_true = 0.75
U_synth = np.eye(dim, dtype=np.complex128)
U_synth[0, 0] = np.exp(1j * 2 * np.pi * phi_true)
psi_synth = np.zeros(dim, dtype=np.complex128)
psi_synth[0] = 1.0
print(f"  Atteso: k=12, phi=0.7500")
run_and_print("TEST 1", psi_synth, U_synth)


# ── TEST 2 ─────────────────────────────────────────────────────
separator("TEST 2 — autovettore esatto E_0 + U Trotter (steps=3)")
print("  Costruzione U Trotter con separazione fase globale:")
U_trotter = build_unitary_trotter(qh, t=t, steps=3)

v0 = eigvecs[:, 0]
phi_E0 = (eigvals[0] * t / (2 * np.pi)) % 1.0
print(f"  Atteso: k={phi_E0*16:.2f}, phi={phi_E0:.4f}")
run_and_print("TEST 2", v0, U_trotter)


# ── TEST 3 ─────────────────────────────────────────────────────
separator("TEST 3 — bitstring HF '1100' + U Trotter (steps=3)")
psi_hf = np.zeros(dim, dtype=np.complex128)
idx_hf = int('1100', 2)
psi_hf[idx_hf] = 1.0

print("  Overlap |<E_k|1100>|^2:")
for i, v in enumerate(eigvecs.T):
    ov = abs(np.dot(v.conj(), psi_hf))**2
    if ov > 1e-4:
        phi_e = (eigvals[i] * t / (2*np.pi)) % 1.0
        print(f"    E_{i} = {eigvals[i]:.6f} Ha | overlap={ov:.4f} | k atteso={phi_e*16:.2f}")

run_and_print("TEST 3", psi_hf, U_trotter)


# ── TEST 4 ─────────────────────────────────────────────────────
separator("TEST 4 — convergenza Trotter su autovettore E_0")
U_exact_ref = eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T
for steps in [1, 3, 5, 10, 20]:
    U_t = build_unitary_trotter(qh, t=t, steps=steps)
    err = np.linalg.norm(U_t - U_exact_ref, ord='fro')
    arr = np.zeros(16, dtype=int)
    counts, _ = run_qpe(v0, U_t, n=4, shots=1024)
    for bitstring, c in counts.items():
        arr[int(bitstring, 2)] = c
    best_k = int(np.argmax(arr))
    print(f"  steps={steps:2d} | Trotter err={err:.4e} | QPE k={best_k} | phi={best_k/16:.4f}")


separator("TEST 5 — verifica convenzione segno QPE")
# costruisci U con fase NEGATIVA equivalente a phi=0.878
phi_neg = 1.0 - 0.878  # = 0.122  →  k=2 con n=4
phi_pos = 0.878          #           →  k=14 con n=4

U_neg = np.eye(dim, dtype=np.complex128)
U_pos = np.eye(dim, dtype=np.complex128)
U_neg[0,0] = np.exp(1j * 2 * np.pi * phi_neg)
U_pos[0,0] = np.exp(1j * 2 * np.pi * phi_pos)

psi0 = np.zeros(dim, dtype=np.complex128)
psi0[0] = 1.0

print(f"  U con phi={phi_neg:.3f} (k atteso=2):")
run_and_print("U_neg", psi0, U_neg)

print(f"  U con phi={phi_pos:.3f} (k atteso=14):")
run_and_print("U_pos", psi0, U_pos)