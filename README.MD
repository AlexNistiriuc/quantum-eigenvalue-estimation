# Quantum Simulation Framework

Framework integrato per eseguire **VQE e QPE**, pensato per testare algoritmi di simulazione quantistica.

## 🔹 Features
- Interfaccia unificata per eseguire sia **VQE** che **QPE**.  
- Gestione di **file Hamiltoniani** e **unitari** in formato JSON.  
- Simulatore interno compatibile con **Qiskit Aer**.  
- Supporto per **analisi grafica** dei risultati (energie, fasi, distribuzioni).

## 💡 Key Concepts
- Architettura ibrida **quantum-classical** per simulazione di sistemi fisici.  
- **Compatibilità estesa** tra algoritmi variazionali e di stima di fase.  
- **Benchmarking** tra metodi di calcolo di autovalori e autostati.

## Structure
```
.\qpe
│   ├── qpe_code
│   ├── qpe_results
│   └── unitary_matrices
│
.\vqe
│   ├── vqe_code
│   ├── vqe_results
│   └── molecules
```

## Quick start

1. Create and activate a virtual environment (PowerShell):

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Run tests:

```powershell
python -m pytest -q
```

4. Run VQE (example):

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal
```

5. Run QPE (example):

```powershell
python -m qpe.qpe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json
```

6. Run full VQE→QPE pipeline from `main.py` (example):

```powershell
python main.py --vqe-source molecule --vqe-input H2 --vqe-shots 1024 --vqe-ansatz twolocal --qpe-n 5 --qpe-shots 4096
```

For Hamiltonian input:

```powershell
python main.py --vqe-source hamiltonian --vqe-input molecules/Hamiltonian_8x8_example.json --vqe-maxiter 500 --qpe-n 5
```

Results and logs are saved under `qpe/qpe_results/` and `vqe/vqe_results/` respectively.
