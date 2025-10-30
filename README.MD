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
