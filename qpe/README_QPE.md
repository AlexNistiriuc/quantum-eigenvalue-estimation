# Quantum Phase Estimation (QPE)

This module implements **Quantum Phase Estimation** for estimating phases associated with unitary operators or quantum Hamiltonians.

## Features
- Builds a QPE circuit with dedicated **phase** and **state** registers.
- Supports **unitary matrices** loaded from `.json` files.
- Applies **controlled-U gates** and the **inverse QFT**.
- Runs multi-shot simulations and reconstructs the phase histogram.
- Compares the measured phase against the theoretical value.

## Key Concepts
- **Eigenstates and eigenvalues** of unitary operators.
- Encoding the **quantum phase** in measurement registers.
- The **inverse quantum Fourier transform (QFT⁻¹)**.
- Numerical phase extraction from discrete outcomes.

## Usage

- Run from a unitary JSON file:

```powershell
python -m qpe.qpe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --n 5 --shots 1024
```

- Run using a previously generated circuit from VQE:

```powershell
python -m qpe.qpe_code.run_from_molecule molecules/H2_sto-3g_qubit_hamiltonian.json --shots 4096
```

## Output
- A PNG histogram and a `qpe_log.txt` file are written to `qpe/qpe_results/<timestamp>/`.
