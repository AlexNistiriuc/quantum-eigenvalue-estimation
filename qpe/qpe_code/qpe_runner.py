# qpe_runner.py

import numpy as np
import qiskit as qk
from typing import Tuple
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit_aer import AerSimulator

def simulator(qc: qk.QuantumCircuit,
              shots: int
              ) -> np.array:
    """
    Simulate the circuit and return measurement counts.

    Args:
        qc: qiskit.QuantumCircuit, the addressed quantum circuit
        shots: int

    Return:
        counts: np.array
    """

    simulator = AerSimulator()
    transpiled_qc = qk.transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    return result.get_counts()


def run_qpe(psi_vector: np.array,
            U: np.ndarray,
            n: int,
            shots: int,
            ) -> Tuple[np.array, qk.QuantumCircuit]:
    """
    Args:
        psi_vector: numpy.array, the initial autostate psi
        U: numpy.ndarray unitary (d x d), the unitary operation
        n: int, accuracy, number of phase registers
    """

    print("....Creating circuit....")
    # local variables
    m = int(np.ceil(np.log2(len(U))))   # number of state regiter
    U_gate = UnitaryGate(U, label='U')               # unitary operation as Qiskit's gate

    # create the registers
    phase_reg = qk.QuantumRegister(n, "phase")              # initial phase registers, set in |0>
    state_reg = qk.QuantumRegister(m, "state")              # state registers set in |0>
    c_reg = qk.ClassicalRegister(n, 'c')                    # classical register
    qc = qk.QuantumCircuit(phase_reg, state_reg, c_reg)     # quantum circuit

    # initialize the state register as psi
    qc.initialize(psi_vector, state_reg)

    # put an Hadamard's gate on each phase register
    qc.h(phase_reg)
    qc.barrier()

    # put 2^k U_gates on the state_reg controlled by the k-th phase_reg
    for k in range(n):
        number_of_gates = 2**k                                      # 2^k gates
        for i in range(number_of_gates):
            qc.append(U_gate.control(1), [phase_reg[k]] + list(state_reg))  # U_gates on state_reg controlled by the k-th phase_reg
    
    qc.barrier()

    # anti quantum Fourier transform
    iqft = QFT(n, inverse=True)
    qc.append(iqft, phase_reg)
    qc.barrier()

    # put measeure block on phase_registers
    qc.measure(phase_reg, c_reg)

    # simulations
    print(f"....Starting simulations....")
    counts = simulator(qc, shots=shots)

    return counts, qc
