# Quantum Simulation Framework

Integrated framework for running **VQE** and **QPE** experiments on small quantum systems.

## Features
- Unified entry point for both **VQE** and **QPE** workflows.
- JSON-based input support for both **Hamiltonians** and **unitary matrices**.
- Internal simulator compatible with **Qiskit Aer**.
- Plotting and logging for energies, phases, and histograms.

## Key Concepts
- Hybrid **quantum-classical** architecture for physics simulation.
- Tight integration between variational algorithms and phase estimation.
- Benchmarking of eigenvalue and eigenstate estimation methods.

## Project Layout
```
.
├── qpe
│   ├── qpe_code
│   ├── qpe_results
│   └── unitary_matrices
└── vqe
	├── vqe_code
	├── vqe_results
	└── molecules
```

## Quick Start

1. Create and activate a virtual environment in PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

2. Install the dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Run VQE:

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal
python -m vqe.vqe_code.run_from_molecule H2 --method spsa --maxiter 400 --spsa-a 0.2 --spsa-c 0.1
```

4. Run QPE:

```powershell
python -m qpe.qpe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json
```

5. Run the full VQE→QPE pipeline:

```powershell
python main.py --vqe-source molecule --vqe-input H2 --vqe-shots 1024 --vqe-ansatz twolocal --qpe-n 5 --qpe-shots 4096
python main.py --vqe-source hamiltonian --vqe-input molecules/Hamiltonian_8x8_example.json --vqe-maxiter 500 --qpe-n 5
```

Results and logs are written to `qpe/qpe_results/` and `vqe/vqe_results/`.
