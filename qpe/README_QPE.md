# Quantum Phase Estimation (QPE) Algorithm

Implementa l’algoritmo di **Quantum Phase Estimation** per stimare **fasi** associati a operatori unitari o Hamiltoniane quantistiche.

## 🔹 Features
- Costruzione del circuito QPE con **registri di fase e stato**.  
- Supporto per **matrici unitarie** caricate da file `.json`.  
- Applica **controlled-U gates** e **Inverse QFT**.  
- Simulazione con più shot e ricostruzione dell’istogramma delle fasi.  
- Confronto automatico con la **fase teorica**.

## 💡 Key Concepts
- **Autostati e autovalori** di operatori unitari.  
- Codifica della **fase quantistica** in registri di misura.  
- **Trasformata di Fourier quantistica inversa (QFT⁻¹)**.  
- **Estrazione numerica** della fase da risultati discreti.

## Usage

- Run from a unitary JSON file:

```powershell
python -m qpe.qpe_code.run_from_hamiltonian molecules/Hamiltonian_8x8_example.json --n 5 --shots 1024
```

- Run using a previously-generated circuit (from VQE):

```powershell
python -m qpe.qpe_code.run_from_molecule molecules/H2_sto-3g_qubit_hamiltonian.json --shots 4096
```

Output:
- A PNG histogram and a `qpe_log.txt` are written to `qpe/qpe_results/<timestamp>/`.
