#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from scipy.linalg import expm

p = Path('molecules/H2_sto-3g_qubit_hamiltonian.json')
if not p.exists():
    raise SystemExit(f'Missing {p}')

data = json.load(open(p, 'r'))
qh = data['qubit_hamiltonian']

PAULIS = {
    'I': np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    'X': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    'Y': np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    'Z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def pmat(pstr: str):
    mats = [PAULIS[c] for c in pstr]
    op = mats[0]
    for m in mats[1:]:
        op = np.kron(op, m)
    return op


def diag(t=0.6, hbar=1.0, steps=3):
    dt = t / steps
    any_key = next(iter(qh))
    n_qubits = len(any_key)
    dim = 2 ** n_qubits
    U_step = np.eye(dim, dtype=np.complex128)

    print(f"Diagnostics: t={t}, hbar={hbar}, steps={steps}, dt={dt}, dim={dim}")
    print(f"Number of Pauli terms: {len(qh)}")

    term_idx = 0
    for pstr, coeff in qh.items():
        term_idx += 1
        c = complex(coeff)
        P = pmat(pstr)
        arg = -1j * c * P * (dt / hbar)
        max_arg = np.max(np.abs(arg))
        has_imag = (abs(c.imag) > 0)
        print(f"\nTerm {term_idx}: '{pstr}' coeff={c} imag={has_imag}")
        print(f"  P.shape={P.shape}, ||P||_F={np.linalg.norm(P):.6f}")
        print(f"  max|arg| = {max_arg:.6e}")
        try:
            U_term = expm(arg)
        except Exception as e:
            print(f"  expm raised exception: {e}")
            raise
        norm_Uterm = np.linalg.norm(U_term)
        finite = np.all(np.isfinite(U_term))
        print(f"  ||U_term||_F = {norm_Uterm:.6e}, finite={finite}")

        # multiply and check U_step
        U_step = U_term @ U_step
        print(f"  After multiply: ||U_step||_F = {np.linalg.norm(U_step):.6e}, finite={np.all(np.isfinite(U_step))}")
        if not np.all(np.isfinite(U_step)):
            print("  --> U_step contains non-finite entries; stopping diagnostics.")
            break

    # if completed, compute final U
    if np.all(np.isfinite(U_step)):
        U = np.linalg.matrix_power(U_step, steps)
        print(f"\nFinal U computed: ||U||_F = {np.linalg.norm(U):.6e}, finite={np.all(np.isfinite(U))}")
    else:
        print('\nDiagnostics ended early due to non-finite U_step')


if __name__ == '__main__':
    import sys
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    hbar = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    diag(t=t, hbar=hbar, steps=steps)
