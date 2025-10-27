# Quantum VQE Simulator for Molecular Ground States

Simula l’**energia dello stato fondamentale** di semplici molecole (H₂, LiH) utilizzando il **Variational Quantum Eigensolver (VQE)** con **Qiskit**.

## 🔹 Features
- Costruzione automatica dell’**ansatz parametrico** (`TwoLocal`, `UCCSD`).  
- Calcolo dell’**energia attesa** da Hamiltoniane sotto forma di **Pauli strings**.  
- Possibilità di eseguire ottimizzazioni classiche con differenti algoritmi.  
- Analisi di **convergenza** rispetto ai parametri e confronto con valori teorici.  
- Supporto a Hamiltoniane molecolari da file `.json`.

## 💡 Key Concepts
- Parametrizzazione dello **stato quantistico variazionale**.  
- Misurazione e media di osservabili qubit-based.  
- **Ottimizzazione ibrida** tra hardware quantistico e classico.  
- Analisi e **visualizzazione del processo di convergenza**.
