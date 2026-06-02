# Quantum VQE Simulator for Molecular Ground States

This project simulates the **ground-state energy** of small molecules such as H₂ and LiH using the **Variational Quantum Eigensolver (VQE)** with **Qiskit**.

## Features
- Automatic construction of parameterized ansatz circuits (`TwoLocal`, `UCCSD`).
- Expectation value evaluation for Hamiltonians expressed as **Pauli strings**.
- Classical optimization with multiple algorithms, including **COBYLA** and **SPSA**.
- Convergence analysis against theoretical reference values.
- Support for molecular Hamiltonians loaded from `.json` files.

## Entry Points
- `vqe/vqe_code/run_from_molecule.py`: runs VQE from a molecular JSON file with `qubit_hamiltonian`.
- `vqe/vqe_code/run_from_hamiltonian.py`: runs VQE from a generic Hamiltonian JSON (`H` matrix or `qubit_hamiltonian`).
- `vqe/vqe_code/vqe_runner.py`: shared optimization and circuit-building logic used by the entry points.

## CLI Examples
- `python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal`
- `python -m vqe.vqe_code.run_from_molecule LiH --ansatz uccsd --maxiter 1500`
- `python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --shots 1024 --ansatz twolocal`
- `python -m vqe.vqe_code.run_from_molecule H2 --method spsa --maxiter 400 --spsa-a 0.2 --spsa-c 0.1`
- `python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --method spsa --maxiter 600 --spsa-stability-offset 60`

## Ansatz Parameters
If you need to customize the ansatz, the available arguments are:

| Argument | Meaning |
| --- | --- |
| `--two-local-reps` | Number of repetitions in the `TwoLocal` circuit. |
| `--two-local-rotation-blocks` | Rotation blocks used by `TwoLocal`, for example `ry,rz`. |
| `--two-local-entanglement` | Entanglement pattern for `TwoLocal`, for example `linear` or `circular`. |
| `--two-local-entanglement-blocks` | Entangling gates used by `TwoLocal`, for example `cx`. |
| `--two-local-parameter-prefix` | Parameter prefix used by `TwoLocal`. |
| `--uccsd-reps` | Number of repetitions in the `UCCSD` ansatz. |
| `--uccsd-generalized` | Enables generalized `UCCSD`. |
| `--uccsd-preserve-spin` | Preserves spin in `UCCSD`. |
| `--uccsd-include-imaginary` | Includes imaginary excitations in `UCCSD`. |

List-style values such as `--two-local-rotation-blocks` and `--two-local-entanglement-blocks` must be passed as comma-separated values, for example `ry,rz` or `cx,cz`.

## Key Concepts
- Variational **quantum-state parameterization**.
- Measurement and averaging of qubit observables.
- Hybrid **quantum-classical optimization**.
- Visualization of the convergence process.

## Usage Examples

1. Activate the project virtual environment and install the dependencies, following the root README.

2. Run VQE from a molecular JSON file:

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal
```

To use SPSA instead of COBYLA:

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --method spsa --maxiter 400 --spsa-a 0.2 --spsa-c 0.1
```

Example with a custom `TwoLocal` ansatz:

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --two-local-reps 4 --two-local-rotation-blocks ry,rz --two-local-entanglement circular --two-local-entanglement-blocks cx --two-local-parameter-prefix alpha
```

Example with a custom `UCCSD` ansatz:

```powershell
python -m vqe.vqe_code.run_from_molecule LiH --ansatz uccsd --uccsd-reps 3 --uccsd-generalized true --uccsd-preserve-spin true --uccsd-include-imaginary false
```

3. Run VQE from a generic Hamiltonian JSON file:

```powershell
python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --shots 1024
```

4. Results and logs are saved under `vqe/vqe_results/<system>/<timestamp>/`.

Notes:
- The recommended entry points are the `run_from_*` modules.
