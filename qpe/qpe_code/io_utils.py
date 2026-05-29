import json
from typing import Any, Dict, Optional
import numpy as np
from scipy.linalg import expm
from pathlib import Path

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_complex_str(s):
    # Accept strings like "0.5+0.3j" or numbers
    if isinstance(s, (int, float, complex)):
        return complex(s)
    if isinstance(s, str):
        return complex(s.replace('J', 'j'))
    raise ValueError(f"Unsupported complex entry: {s!r}")

def parse_matrix(obj) -> np.ndarray:
    arr = np.array([[parse_complex_str(x) for x in row] for row in obj], dtype=np.complex128)
    return arr

def parse_vector(obj) -> np.ndarray:
    vec = np.array([parse_complex_str(x) for x in obj], dtype=np.complex128)
    return vec

def build_unitary_from_hamiltonian(H: np.ndarray, t: float = 1.0, h_bar: float = 1.0) -> np.ndarray:
    # U = exp(-i H t / h_bar)
    return expm(-1j * H * (t / h_bar)).astype(np.complex128)

def load_matrix_spec(path: str) -> Dict[str, Any]:
    data = load_json(path)
    spec: Dict[str, Any] = {'metadata': data}
    dim = data.get('dimension')
    if dim is None:
        # try infer from H or U
        if 'H' in data:
            dim = len(data['H'])
        elif 'molecule' in data:
            #TBD
            pass
        else:
            raise ValueError("JSON must contain 'dimension' or 'H' or 'molecule'")

    spec['dim'] = int(dim)
    spec['t'] = float(data.get('t', 1.0))
    spec['hbar'] = float(data.get('hbar', 1.0))

    if 'H' in data:
        H = parse_matrix(data['H'])
        spec['kind'] = 'hamiltonian'
        spec['H'] = H
        spec['U'] = build_unitary_from_hamiltonian(H, spec['t'], spec['hbar'])
    elif 'U' in data:
        U = parse_matrix(data['U'])
        spec['kind'] = 'unitary'
        spec['U'] = U
    else:
        raise ValueError("JSON must contain 'H' (Hamiltonian) or 'U' (unitary).")

    # psi: explicit vector or index
    if 'psi' in data:
        psi = parse_vector(data['psi'])
        spec['psi'] = psi / np.linalg.norm(psi)
    elif 'psi_index' in data:
        idx = int(data['psi_index'])
        psi = np.zeros(spec['dim'], dtype=np.complex128)
        psi[idx] = 1.0
        spec['psi'] = psi
    else:
        # default to ground basis state 0
        psi = np.zeros(spec['dim'], dtype=np.complex128)
        psi[0] = 1.0
        spec['psi'] = psi

    return spec