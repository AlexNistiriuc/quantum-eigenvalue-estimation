#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from qpe.qpe_code.qpe_graph import plot
from qpe.qpe_code.qpe_runner import run_qpe
from vqe.vqe_code.run_from_hamiltonian import main as run_vqe_from_hamiltonian
from vqe.vqe_code.run_from_molecule import main as run_vqe_from_molecule


REPO_ROOT = Path(__file__).resolve().parent
QPE_RESULTS_DIR = REPO_ROOT / "qpe" / "qpe_results"


def _ensure_repo_root_cwd():
    os.chdir(REPO_ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run end-to-end VQE -> QPE pipeline.")

    parser.add_argument(
        "--vqe-source",
        choices=["molecule", "hamiltonian"],
        default="molecule",
        help="VQE input type: molecule JSON/name or generic Hamiltonian JSON.",
    )
    parser.add_argument(
        "--vqe-input",
        default="H2",
        help="VQE input value: molecule name/path or Hamiltonian JSON path.",
    )
    parser.add_argument("--vqe-shots", type=int, default=1024, help="VQE shots.")
    parser.add_argument(
        "--vqe-ansatz",
        type=str,
        default="twolocal",
        choices=["twolocal", "uccsd", "1", "2"],
        help="VQE ansatz type.",
    )
    parser.add_argument(
        "--vqe-method",
        type=str,
        default="cobyla",
        choices=["cobyla", "spsa"],
        help="VQE optimization method.",
    )
    parser.add_argument("--vqe-spsa-a", type=float, default=0.2, help="VQE SPSA step-size coefficient a.")
    parser.add_argument("--vqe-spsa-c", type=float, default=0.1, help="VQE SPSA perturbation coefficient c.")
    parser.add_argument("--vqe-spsa-alpha", type=float, default=0.602, help="VQE SPSA decay exponent alpha.")
    parser.add_argument("--vqe-spsa-gamma", type=float, default=0.101, help="VQE SPSA perturbation decay exponent gamma.")
    parser.add_argument(
        "--vqe-spsa-stability-offset",
        type=float,
        default=None,
        help="VQE SPSA stability offset A; if omitted, a heuristic based on maxiter is used.",
    )
    parser.add_argument("--vqe-maxiter", type=int, default=2000, help="VQE max iterations.")
    parser.add_argument("--vqe-two-local-reps", type=int, default=3, help="VQE TwoLocal repetitions.")
    parser.add_argument("--vqe-seed", type=int, default=None, help="VQE random seed.")
    parser.add_argument("--vqe-pauli-tol", type=float, default=1e-10, help="Tolerance for matrix->Pauli pruning.")
    parser.add_argument("--vqe-name", type=str, default=None, help="Optional VQE system name override.")

    parser.add_argument("--qpe-n", type=int, default=5, help="QPE phase qubits.")
    parser.add_argument("--qpe-shots", type=int, default=4096, help="QPE shots.")
    parser.add_argument(
        "--qpe-scale-factor",
        type=float,
        default=0.95,
        help="QPE time scaling factor in t = 2*pi*scale_factor/|energy_scale|.",
    )
    return parser.parse_args(argv)


def _resolve_pipeline_name(args):
    if args.vqe_name:
        return args.vqe_name
    input_path = Path(str(args.vqe_input))
    if input_path.suffix.lower() == ".json":
        return input_path.stem
    return str(args.vqe_input)


def _circuit_summary(circuit):
    count_ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
    payload = {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "num_parameters": int(circuit.num_parameters),
        "count_ops": count_ops,
    }
    hash_input = (
        f"{payload['depth']}|{payload['size']}|{payload['num_qubits']}|"
        f"{payload['num_clbits']}|{payload['num_parameters']}|{sorted(count_ops.items())}"
    )
    payload["circuit_hash"] = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    return payload


def _counts_to_array(counts, n_phase):
    counts_array = np.zeros(2**int(n_phase), dtype=int)
    for bitstring, count in counts.items():
        counts_array[int(bitstring, 2)] = int(count)
    return counts_array


def _write_summary(file_obj, summary_dict):
    print("SUMMARY_START", file=file_obj)
    print(json.dumps(summary_dict, sort_keys=True), file=file_obj)
    print("SUMMARY_END", file=file_obj)


def _run_vqe(args):
    if args.vqe_source == "molecule":
        vqe_args = argparse.Namespace(
            molecule=args.vqe_input,
            shots=args.vqe_shots,
            ansatz=args.vqe_ansatz,
            method=args.vqe_method,
            spsa_a=args.vqe_spsa_a,
            spsa_c=args.vqe_spsa_c,
            spsa_alpha=args.vqe_spsa_alpha,
            spsa_gamma=args.vqe_spsa_gamma,
            spsa_stability_offset=args.vqe_spsa_stability_offset,
            maxiter=args.vqe_maxiter,
            two_local_reps=args.vqe_two_local_reps,
            seed=args.vqe_seed,
        )
        return run_vqe_from_molecule(vqe_args)

    vqe_args = argparse.Namespace(
        input=args.vqe_input,
        shots=args.vqe_shots,
        ansatz=args.vqe_ansatz,
        method=args.vqe_method,
        spsa_a=args.vqe_spsa_a,
        spsa_c=args.vqe_spsa_c,
        spsa_alpha=args.vqe_spsa_alpha,
        spsa_gamma=args.vqe_spsa_gamma,
        spsa_stability_offset=args.vqe_spsa_stability_offset,
        maxiter=args.vqe_maxiter,
        two_local_reps=args.vqe_two_local_reps,
        seed=args.vqe_seed,
        pauli_tol=args.vqe_pauli_tol,
        name=args.vqe_name,
    )
    return run_vqe_from_hamiltonian(vqe_args)


def main(argv=None):
    _ensure_repo_root_cwd()
    args = parse_args(argv)
    H, min_energy, best_cirq, energy_scale = _run_vqe(args)

    if best_cirq is None:
        raise RuntimeError("VQE did not return a circuit to use as QPE input")

    scale = max(abs(float(energy_scale)), 1e-12)
    t = 2 * np.pi * float(args.qpe_scale_factor) / scale

    U = expm(-1j * H * t)
    psi = Statevector.from_instruction(best_cirq)

    print("\n" + "=" * 18 + " STARTING QPE " + "=" * 18)
    n = int(args.qpe_n)
    shots = int(args.qpe_shots)
    pipeline_name = _resolve_pipeline_name(args)

    timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = QPE_RESULTS_DIR / f"pipeline_{pipeline_name}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "qpe_results.png"
    log_path = output_dir / "qpe_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("=" * 18 + " PIPELINE DATA " + "=" * 18)
            print(f"vqe_source: {args.vqe_source}")
            print(f"vqe_input: {args.vqe_input}")
            print(f"vqe_shots: {args.vqe_shots}")
            print(f"vqe_ansatz: {args.vqe_ansatz}")
            print(f"vqe_maxiter: {args.vqe_maxiter}")
            print(f"qpe_n: {n}")
            print(f"qpe_shots: {shots}")
            print(f"qpe_scale_factor: {args.qpe_scale_factor}")
            print("=" * 50)

        start_time = time.time()
        counts, circuit = run_qpe(psi, U, n=n, shots=shots)
        end_time = time.time()
        print("....Ending simulations....\n" + "=" * 50)
        elapsed_qpe = end_time - start_time

        all_k = np.arange(0, 2**n)
        counts_array = _counts_to_array(counts, n)

        best_k = int(np.argmax(counts_array))
        phi_est = best_k / (2**n)
        circuit_info = _circuit_summary(circuit)

        phi_expected = float(((-float(min_energy) * t) / (2 * np.pi)) % 1.0)
        summary = {
            "system_or_file": str(pipeline_name),
            "n_phase": int(n),
            "shots": int(shots),
            "phi_peak": float(phi_est),
            "phi_neighbor": None,
            "phi_expected": phi_expected,
            "phase_diff": None,
            "most_freq_count": int(counts_array[best_k]),
            "elapsed_s": float(elapsed_qpe),
            "output_dir": str(output_dir),
            "vqe_source": str(args.vqe_source),
            "vqe_ansatz": str(args.vqe_ansatz),
            "circuit_summary": circuit_info,
        }

        with redirect_stdout(f):
            print("=" * 15 + " QPE results " + "=" * 15)
            print(f"Shots: {shots}")
            print(f"Time elapsed: {elapsed_qpe:.4f} s")
            print(f"QPE phase: phi = {best_k}/{2 ** n} = {phi_est:.6f}")
            print(f"Most frequent count: {counts_array[best_k]}/{shots}")
            print("Circuit summary:")
            print(circuit_info)

            _write_summary(f, summary)

        plot(counts_array, best_k=best_k, all_k=all_k, phi_est=phi_est, n=n, shots=shots, output_dir=output_file)

    print(f"Log saved in: {log_path}")
    print(f"Plot saved in: {output_file}")
    print("\n" + "=" * 19 + " ENDING QPE " + "=" * 19 + "\n")


if __name__ == "__main__":
    main()
