# main.py

import os
import time
import json
import numpy as np
from contextlib import redirect_stdout
from qpe_graph import plot
from simple_unitary import build_diag_unitary as build
from qpe_runner import run_qpe

def get_input():
    while True:
        shots = input("\nHow many shots do you want to test? ")
        try:
            shots = int(shots)
            break
        except ValueError:
            print(f"Error: {shots} is not an integer value!.\n")

    # while True:
    #     unitary = input("Which unitary matrix do you wanna test?\n\t1. small (4x4)\n\t2. medium (16x16)\n\t3. large (64x64)\n\t4. extra-large (256x256)\nAnswer: ")
    #     if unitary in ["1", "2", "3", "4"]:
    #         break
    #     print(f"Error: {unitary} is not a possible choise!\n")

    # if unitary == "1":
    #     unitary = "small"
    # elif unitary == "2":
    #     unitary = "medium"
    # elif unitary == "3":
    #     unitary = "large"
    # elif unitary == "4":
    #     unitary = "extra_large"

    # filename = os.path.join("..", "unitary_matrices", f"{unitary}.json")
    # print(f"Loading molecule data from: {filename}")
    # try:
    #     with open(filename, 'r', encoding='utf-8') as file:
    #         matrix = json.load(file)
    #         print(f"Matrix \"{unitary}\" loaded successfully!")
    # except FileNotFoundError:
    #     print(f"Error: Data file for matrix \"{unitary}\" not found.")
    # except json.JSONDecodeError:
    #     print(f"Error: {filename} is not a valid JSON file.")

    # U = np.array([[complex(x) for x in row] for row in matrix["U"]], dtype=np.complex128)
    # psi = np.array([complex(x) for x in matrix["psi"]], dtype=np.complex128)
    # psi /= np.linalg.norm(psi)

    U = np.array([
        [0.0026178144+0.5460550815j, 0.6058816515-0.0447616024j, -0.4707253703+0.3302756766j, -0.0398119521+0.0217124394j],
        [-0.1176999530-0.2325002357j, 0.6300340009+0.0488685476j, 0.5162626984-0.0968122797j, -0.4764436722+0.1728034150j],
        [-0.6616954539+0.1257949327j, -0.0679664289+0.0165101438j, 0.3733384687+0.4914243822j, 0.3723108323+0.1481480658j],
        [0.3987084427-0.1458550494j, 0.2915109863+0.3764334877j, 0.0842049672-0.0739310482j, 0.6482503054+0.4003652354j]
    ], dtype=np.complex128)

    # Eigen-decomposition
    evals, evecs = np.linalg.eig(U)

    # Normalizziamo gli autostati
    evecs = evecs / np.linalg.norm(evecs, axis=0)

    print("Autovalori (fasi):")
    for j, lam in enumerate(evals):
        phi_j = (np.angle(lam) / (2*np.pi)) % 1
        print(f"λ_{j} = {lam}, φ_{j} = {phi_j:.4f}")
        
    # Scegliamo autostati 0, 1, 2
    c0, c1, c2 = 0.2, 0.3, 0.5  # coefficenti
    psi = c0*evecs[:,0] + c1*evecs[:,1] + c2*evecs[:,2]
    psi /= np.linalg.norm(psi)  # normalizzazione

    while True:
        n = input("How many phase registers do you want to test with? ")
        try:
            n = int(n)
            break
        except ValueError:
            print(f"Error: {n} is not an integer value!.\n")

    return shots, U, psi, n

def main():
    print('\n' + '='*18 + " STARTING QPE " + '='*18)

    shots, U, psi, n = get_input()
    
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
