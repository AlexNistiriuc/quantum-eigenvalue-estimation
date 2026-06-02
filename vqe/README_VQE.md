# Quantum VQE Simulator for Molecular Ground States

Simula l’**energia dello stato fondamentale** di semplici molecole (H₂, LiH) utilizzando il **Variational Quantum Eigensolver (VQE)** con **Qiskit**.

## 🔹 Features
- Costruzione automatica dell’**ansatz parametrico** (`TwoLocal`, `UCCSD`).  
- Calcolo dell’**energia attesa** da Hamiltoniane sotto forma di **Pauli strings**.  
- Possibilità di eseguire ottimizzazioni classiche con differenti algoritmi.  
- Analisi di **convergenza** rispetto ai parametri e confronto con valori teorici.  
- Supporto a Hamiltoniane molecolari da file `.json`.

## 🔹 Struttura Entry Point (allineata a QPE)
- `vqe/vqe_code/run_from_molecule.py`: esegue VQE partendo da JSON molecolare con `qubit_hamiltonian`.
- `vqe/vqe_code/run_from_hamiltonian.py`: esegue VQE partendo da JSON Hamiltoniano generico (`H` matrix o `qubit_hamiltonian`).
- `vqe/vqe_code/vqe_common.py`: funzioni comuni condivise tra entrypoint (`execute_vqe_run`, `build_dense_hamiltonian`).

## 🔹 Esempi CLI
- `python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal`
- `python -m vqe.vqe_code.run_from_molecule LiH --ansatz uccsd --maxiter 1500`
- `python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --shots 1024 --ansatz twolocal`
- `python -m vqe.vqe_code.run_from_molecule H2 --method spsa --maxiter 400 --spsa-a 0.2 --spsa-c 0.1`
- `python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --method spsa --maxiter 600 --spsa-stability-offset 60`

## 💡 Key Concepts
- Parametrizzazione dello **stato quantistico variazionale**.  
- Misurazione e media di osservabili qubit-based.  
- **Ottimizzazione ibrida** tra hardware quantistico e classico.  
- Analisi e **visualizzazione del processo di convergenza**.

## Usage & Examples

1. Activate the project virtualenv and install requirements (see root README).

2. Run VQE from molecule JSON (example for H2):

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --shots 2048 --ansatz twolocal
```

To use SPSA instead of COBYLA:

```powershell
python -m vqe.vqe_code.run_from_molecule H2 --method spsa --maxiter 400 --spsa-a 0.2 --spsa-c 0.1
```

3. Run VQE from a generic Hamiltonian JSON:

```powershell
python -m vqe.vqe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --shots 1024
```

4. Output and logs are saved under `vqe/vqe_results/<system>/<timestamp>/`.

Notes:
- Recommended entrypoints are the `run_from_*` modules; shared logic lives in `vqe_common.py`.
