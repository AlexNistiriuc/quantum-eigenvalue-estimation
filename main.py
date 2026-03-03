#main.py

import os
import time
import json
import numpy as np
from scipy.linalg import expm
from contextlib import contextmanager
from qiskit.quantum_info import Statevector
from vqe.vqe_code.vqe_main import main as vqe_main
from qpe.qpe_code.qpe_runner import run_qpe as run_qpe
from qpe.qpe_code.qpe_graph import plot
from contextlib import redirect_stdout

@contextmanager
def cambio_directory(path):
    dir_corrente = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(dir_corrente)

def main():
    with cambio_directory("./vqe/vqe_code"):
        H, ground_state, best_cirq, S = vqe_main()
    
    t = 2 * np.pi * 0.95 / S

    U = expm(-1j * H * t)

    psi = Statevector.from_instruction(best_cirq)

    print('\n' + '='*18 + " STARTING QPE " + '='*18)
    n=5
    shots=10000
    with cambio_directory("./qpe/qpe_code"):
        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = os.path.join("..", "qpe_results", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        f = open(os.path.join(output_dir, "qpe_log.txt"), 'w', encoding='utf-8')

        # simulating QPE
        print('='*50)
        start_time = time.time()
        counts, circuit = run_qpe(psi, U, n=n, shots=shots)
        end_time = time.time()
        print(f"....Ending simulations....\n" + '='*50)
        elapsed_vqe = end_time - start_time

        # converting counts in an array
        all_k = np.arange(0, 2**n)
        counts_array = np.zeros(2**n, dtype=int)
        
        for bitstring, count in counts.items():
            k = int(bitstring, 2)               # converting bitstring in integer
            counts_array[k] = count
        
        # Max index
        best_k = int(np.argmax(counts_array))
        phi_est = best_k / (2**n)

        with redirect_stdout(f):
            # Print data
            print('='*22 + " Data " + '='*22)
            print("Unitary matrix:")
            print(U)
            print(f"Eigenstate psi: {psi}")        

            # Print results
            print('='*15 + " Simulation results " + '='*15)
            print(f"Shots: {shots}")
            print(f"Time elapsed: {elapsed_vqe:.4f} s")
            print(f"QPE phase: φ = {best_k}/{2**n} = {phi_est:.6f}")
            print(f"Measurements mode: {counts_array[best_k]}/{shots} = {counts_array[best_k]/shots:.4f}")
            
            print('='*50)
            print(f"Quantum circuit:")
            print(circuit.draw(fold=60, output='text'))

        plot(counts_array, best_k=best_k, all_k=all_k, phi_est=phi_est, n=n, shots=shots, output_dir=output_dir)
        
    print(f"Log and plot saved in: {output_dir}")
    
    print('\n' + '='*19 + " ENDING QPE " + '='*19 + '\n')

if __name__ == "__main__":
    main()
