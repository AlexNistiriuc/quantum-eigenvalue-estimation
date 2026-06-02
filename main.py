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
from qpe.qpe_code.run_from_molecule import trotterize_unitary_from_terms
from qpe.qpe_code.qpe_runner import run_qpe
from vqe.vqe_code.run_from_hamiltonian import execute_vqe_run as execute_vqe_hamiltonian
from vqe.vqe_code.run_from_hamiltonian import resolve_input_path as resolve_h_path
from vqe.vqe_code.run_from_molecule import execute_vqe_run as execute_vqe_molecule
from vqe.vqe_code.run_from_molecule import resolve_molecule_path as resolve_m_path


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"


def _ensure_repo_root_cwd():
    os.chdir(REPO_ROOT)


def _parse_csv_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


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
    parser.add_argument("--vqe-two-local-reps", type=int, default=3, help="VQE TwoLocal repetitions.")
    parser.add_argument(
        "--vqe-two-local-rotation-blocks",
        type=_parse_csv_list,
        default=["ry"],
        help="VQE TwoLocal rotation blocks, comma-separated.",
    )
    parser.add_argument(
        "--vqe-two-local-entanglement",
        type=str,
        default="linear",
        help="VQE TwoLocal entanglement pattern.",
    )
    parser.add_argument(
        "--vqe-two-local-entanglement-blocks",
        type=_parse_csv_list,
        default=["cx"],
        help="VQE TwoLocal entanglement blocks, comma-separated.",
    )
    parser.add_argument(
        "--vqe-two-local-parameter-prefix",
        type=str,
        default="theta",
        help="VQE TwoLocal parameter prefix.",
    )
    parser.add_argument("--vqe-uccsd-reps", type=int, default=2, help="VQE UCCSD repetitions.")
    parser.add_argument(
        "--vqe-uccsd-generalized",
        type=_parse_bool,
        default=False,
        help="Enable generalized VQE UCCSD.",
    )
    parser.add_argument(
        "--vqe-uccsd-preserve-spin",
        type=_parse_bool,
        default=True,
        help="Preserve spin in VQE UCCSD.",
    )
    parser.add_argument(
        "--vqe-uccsd-include-imaginary",
        type=_parse_bool,
        default=True,
        help="Include imaginary excitations in VQE UCCSD.",
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
    parser.add_argument("--vqe-seed", type=int, default=None, help="VQE random seed.")
    parser.add_argument("--vqe-pauli-tol", type=float, default=1e-10, help="Tolerance for matrix->Pauli pruning.")
    parser.add_argument("--vqe-name", type=str, default=None, help="Optional VQE system name override.")

    parser.add_argument("--qpe-n", type=int, default=5, help="QPE phase qubits.")
    parser.add_argument("--qpe-shots", type=int, default=4096, help="QPE shots.")
    parser.add_argument("--qpe-t", type=float, default=None, help="QPE evolution time t override. If set, it overrides scale-factor timing.")
    parser.add_argument("--qpe-hbar", type=float, default=1.0, help="QPE reduced Planck constant used in U = exp(-i H t / hbar).")
    parser.add_argument("--qpe-peak-window", type=int, default=2, help="Neighborhood radius around the dominant peak for neighborhood estimates.")
    parser.add_argument("--qpe-evec-index", type=int, default=None, help="Run QPE on the selected eigenvector index of H.")
    parser.add_argument(
        "--qpe-run-all-eigenstates",
        default=False,
        action="store_true",
        help="Run QPE on every eigenstate of H and generate one plot per eigenstate.",
    )
    parser.add_argument(
        "--qpe-psi-coefs",
        type=str,
        default=None,
        help="Override psi with comma-separated complex coefficients.",
    )
    parser.add_argument(
        "--qpe-psi-eig",
        type=str,
        default=None,
        help="Comma-separated eigenvector indices used with --qpe-psi-coefs.",
    )
    parser.add_argument("--qpe-use-trotter", action="store_true", help="Use Trotterized unitary instead of exact matrix exponential.")
    parser.add_argument("--qpe-trotter-steps", type=int, default=3, help="Compatibility option from qpe run_from_molecule (ignored in pipeline mode).")
    parser.add_argument("--qpe-hf-bits", type=str, default=None, help="Override psi with a computational basis bitstring.")
    parser.add_argument(
        "--qpe-eigenstate-to-overlap",
        type=int,
        default=0,
        help="Target eigenstate index used to report overlap of the selected initial state.",
    )
    parser.add_argument(
        "--qpe-vqe-asats",
        default=False,
        action="store_true",
        help="Use the VQE output state as QPE input state (default pipeline behavior).",
    )
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
    
    # Risolviamo il percorso effettivo per dare il nome corretto alla cartella
    module_dir = REPO_ROOT / "vqe" / "vqe_code"
    if args.vqe_source == "molecule":
        input_path = resolve_m_path(args.vqe_input, module_dir)
    else:
        input_path = resolve_h_path(args.vqe_input, module_dir)
    
    if input_path.exists():
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


def _normalize_statevector(vec):
    arr = np.asarray(vec, dtype=np.complex128)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Statevector norm is zero; cannot normalize.")
    return arr / norm


def _parse_csv_ints(text):
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_csv_complex(text):
    return [complex(part.strip()) for part in str(text).split(",") if part.strip()]


def _hf_bitstring_from_nocc(n_qubits, n_occ):
    bits = ["0"] * int(n_qubits)
    for i in range(int(n_occ)):
        bits[-1 - i] = "1"
    return "".join(bits)


def _resolve_molecule_json_path(vqe_input):
    candidate = Path(str(vqe_input))
    if candidate.exists():
        return candidate.resolve()
    inferred = REPO_ROOT / "molecules" / f"{vqe_input}_sto-3g_qubit_hamiltonian.json"
    if inferred.exists():
        return inferred.resolve()
    return None


def _load_qubit_hamiltonian(args):
    """Helper to reload qubit_hamiltonian dict for Trotterization."""
    if args.vqe_source == "molecule":
        path = _resolve_molecule_json_path(args.vqe_input)
    else:
        path = Path(args.vqe_input)
    
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("qubit_hamiltonian")
    return None


def _run_vqe(args, output_dir):
    module_dir = REPO_ROOT / "vqe" / "vqe_code"
    if args.vqe_source == "molecule":
        mol_path = resolve_m_path(args.vqe_input, module_dir)
        with open(mol_path, "r") as f:
            spec = json.load(f)
        return execute_vqe_molecule(
            hamiltonian_dict=spec["qubit_hamiltonian"],
            system_name=spec["name"],
            num_spatial_orbitals=spec["n_orbs"],
            num_elec=spec["n_elec"],
            results_root=None,
            shots=args.vqe_shots,
            ansatz=args.vqe_ansatz,
            two_local_reps=args.vqe_two_local_reps,
            two_local_rotation_blocks=args.vqe_two_local_rotation_blocks,
            two_local_entanglement=args.vqe_two_local_entanglement,
            two_local_entanglement_blocks=args.vqe_two_local_entanglement_blocks,
            two_local_parameter_prefix=args.vqe_two_local_parameter_prefix,
            uccsd_reps=args.vqe_uccsd_reps,
            uccsd_generalized=args.vqe_uccsd_generalized,
            uccsd_preserve_spin=args.vqe_uccsd_preserve_spin,
            uccsd_include_imaginary=args.vqe_uccsd_include_imaginary,
            method=args.vqe_method,
            spsa_a=args.vqe_spsa_a,
            spsa_c=args.vqe_spsa_c,
            spsa_alpha=args.vqe_spsa_alpha,
            spsa_gamma=args.vqe_spsa_gamma,
            spsa_stability_offset=args.vqe_spsa_stability_offset,
            maxiter=args.vqe_maxiter,
            seed=args.vqe_seed,
            output_dir_override=output_dir
        )
    h_path = resolve_h_path(args.vqe_input, module_dir)
    with open(h_path, "r") as f:
        spec = json.load(f)
    
    # Logica minima per estrarre H o qubit_hamiltonian
    h_dict = spec.get("qubit_hamiltonian")
    h_mat = spec.get("H")
    
    return execute_vqe_hamiltonian(
        hamiltonian_dict=h_dict,
        hamiltonian_matrix=h_mat,
        system_name=args.vqe_name or spec.get("name", h_path.stem),
        num_spatial_orbitals=spec.get("n_orbs", 1),
        num_elec=spec.get("n_elec", [1,1]),
        results_root=None,
        shots=args.vqe_shots,
        ansatz=args.vqe_ansatz,
        two_local_reps=args.vqe_two_local_reps,
        two_local_rotation_blocks=args.vqe_two_local_rotation_blocks,
        two_local_entanglement=args.vqe_two_local_entanglement,
        two_local_entanglement_blocks=args.vqe_two_local_entanglement_blocks,
        two_local_parameter_prefix=args.vqe_two_local_parameter_prefix,
        uccsd_reps=args.vqe_uccsd_reps,
        uccsd_generalized=args.vqe_uccsd_generalized,
        uccsd_preserve_spin=args.vqe_uccsd_preserve_spin,
        uccsd_include_imaginary=args.vqe_uccsd_include_imaginary,
        method=args.vqe_method,
        spsa_a=args.vqe_spsa_a,
        spsa_c=args.vqe_spsa_c,
        spsa_alpha=args.vqe_spsa_alpha,
        spsa_gamma=args.vqe_spsa_gamma,
        spsa_stability_offset=args.vqe_spsa_stability_offset,
        maxiter=args.vqe_maxiter,
        seed=args.vqe_seed,
        output_dir_override=output_dir
    )


def main(argv=None):
    _ensure_repo_root_cwd()
    args = parse_args(argv)

    pipeline_name = _resolve_pipeline_name(args)
    timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
    output_dir = RESULTS_DIR / pipeline_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    H, min_energy, best_cirq, energy_scale = _run_vqe(args, output_dir)

    if best_cirq is None:
        raise RuntimeError("VQE did not return a circuit to use as QPE input")

    hbar = float(args.qpe_hbar)
    if hbar == 0.0:
        raise ValueError("--qpe-hbar must be non-zero.")

    if args.qpe_t is not None:
        t = float(args.qpe_t)
    else:
        scale = max(abs(float(energy_scale)), 1e-12)
        t = 2 * np.pi * float(args.qpe_scale_factor) / scale

    if args.qpe_use_trotter:
        qh = _load_qubit_hamiltonian(args)
        if qh:
            U = trotterize_unitary_from_terms(qh, t, hbar, steps=args.qpe_trotter_steps)
        else:
            U = expm(-1j * H * (t / hbar))
    else:
        U = expm(-1j * H * (t / hbar))

    eigvals, eigvecs = np.linalg.eigh(H)
    vqe_psi = _normalize_statevector(Statevector.from_instruction(best_cirq).data)

    # Default behavior mirroring run_from_molecule: use HF-like state for molecule source,
    # unless qpe_vqe_asats is enabled or explicit overrides are provided.
    psi = vqe_psi.copy()
    psi_source = "vqe"
    explicit_state_override = any(
        [
            args.qpe_hf_bits is not None,
            args.qpe_psi_coefs is not None,
            args.qpe_psi_eig is not None,
            args.qpe_evec_index is not None,
        ]
    )

    if args.vqe_source == "molecule" and not args.qpe_vqe_asats and not explicit_state_override:
        mol_path = _resolve_molecule_json_path(args.vqe_input)
        if mol_path is not None:
            with open(mol_path, "r", encoding="utf-8") as mf:
                mol_spec = json.load(mf)
            n_elec = mol_spec.get("n_elec")
            if isinstance(n_elec, int):
                total_elec = int(n_elec)
            elif isinstance(n_elec, (list, tuple)) and len(n_elec) > 0:
                total_elec = int(sum(n_elec))
            else:
                total_elec = 0
            n_sys = int(np.log2(H.shape[0]))
            bits = _hf_bitstring_from_nocc(n_sys, max(0, min(total_elec, n_sys)))
            psi = np.zeros(H.shape[0], dtype=np.complex128)
            psi[int(bits, 2)] = 1.0
            psi_source = "hf_default"

    if args.qpe_vqe_asats and not explicit_state_override:
        psi = vqe_psi.copy()
        psi_source = "vqe_asats"

    if args.qpe_hf_bits is not None:
        bits = str(args.qpe_hf_bits).strip()
        n_sys = int(np.log2(H.shape[0]))
        if len(bits) != n_sys or any(ch not in {"0", "1"} for ch in bits):
            raise ValueError(f"--qpe-hf-bits must be a {n_sys}-bit binary string.")
        psi = np.zeros(H.shape[0], dtype=np.complex128)
        psi[int(bits, 2)] = 1.0
        psi_source = "hf_bits"

    if (args.qpe_psi_coefs is None) ^ (args.qpe_psi_eig is None):
        raise ValueError("--qpe-psi-coefs and --qpe-psi-eig must be provided together.")
    if args.qpe_psi_coefs is not None and args.qpe_psi_eig is not None:
        coefs = _parse_csv_complex(args.qpe_psi_coefs)
        eig_ids = _parse_csv_ints(args.qpe_psi_eig)
        if len(coefs) != len(eig_ids):
            raise ValueError("Length mismatch between --qpe-psi-coefs and --qpe-psi-eig.")
        psi = np.zeros(H.shape[0], dtype=np.complex128)
        for coef, idx in zip(coefs, eig_ids):
            if idx < 0 or idx >= H.shape[0]:
                raise ValueError(f"Eigenvector index out of range in --qpe-psi-eig: {idx}")
            psi += coef * eigvecs[:, idx]
        psi = _normalize_statevector(psi)
        psi_source = "psi_combo"

    if args.qpe_evec_index is not None:
        idx = int(args.qpe_evec_index)
        if idx < 0 or idx >= H.shape[0]:
            raise ValueError(f"--qpe-evec-index must be between 0 and {H.shape[0]-1}.")
        psi = _normalize_statevector(eigvecs[:, idx])
        psi_source = f"eigenvector_{idx}"

    print("\n" + "=" * 18 + " STARTING QPE " + "=" * 18)
    n = int(args.qpe_n)
    shots = int(args.qpe_shots)

    qpe_plot_path = output_dir / "qpe_results.png"
    log_path = output_dir / "qpe_log.txt"

    with open(log_path, "a", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("=" * 18 + " PIPELINE DATA " + "=" * 18)
            print(f"vqe_source: {args.vqe_source}")
            print(f"vqe_input: {args.vqe_input}")
            print(f"vqe_shots: {args.vqe_shots}")
            print(f"vqe_ansatz: {args.vqe_ansatz}")
            print(f"vqe_maxiter: {args.vqe_maxiter}")
            print(f"qpe_n: {n}")
            print(f"qpe_shots: {shots}")
            print(f"qpe_t: {t}")
            print(f"qpe_hbar: {hbar}")
            print(f"qpe_peak_window: {args.qpe_peak_window}")
            print(f"qpe_scale_factor: {args.qpe_scale_factor}")
            print(f"qpe_psi_source: {psi_source}")
            print(f"qpe_run_all_eigenstates: {bool(args.qpe_run_all_eigenstates)}")
            print(f"qpe_use_trotter: {args.qpe_use_trotter}")
            print(f"qpe_trotter_steps: {args.qpe_trotter_steps}")
            print("=" * 50)

        peak_window = max(0, int(args.qpe_peak_window))
        all_k = np.arange(0, 2**n)

        def _run_one_qpe(psi_vec, run_label, expected_e=None, plot_path=None):
            start_time = time.time()
            counts, circuit = run_qpe(psi_vec, U, n=n, shots=shots)
            end_time = time.time()
            elapsed = end_time - start_time

            counts_array = _counts_to_array(counts, n)
            best_k = int(np.argmax(counts_array))
            phi_est = best_k / (2**n)
            circuit_info = _circuit_summary(circuit)

            if expected_e is None:
                phi_expected_local = float(((-float(min_energy) * t) / (2 * np.pi * hbar)) % 1.0)
            else:
                phi_expected_local = float(((-float(expected_e) * t) / (2 * np.pi * hbar)) % 1.0)

            phase_diff_local = float(abs(((phi_est - phi_expected_local + 0.5) % 1.0) - 0.5))
            lo = max(0, best_k - peak_window)
            hi = min((2**n) - 1, best_k + peak_window)
            local_counts = counts_array[lo : hi + 1]
            local_k = np.arange(lo, hi + 1)
            if local_counts.sum() > 0:
                phi_neighbor_local = float(np.dot(local_k, local_counts) / local_counts.sum() / (2**n))
            else:
                phi_neighbor_local = None

            print("=" * 15 + f" QPE results ({run_label}) " + "=" * 15, file=f)
            print(f"Shots: {shots}", file=f)
            print(f"Time elapsed: {elapsed:.4f} s", file=f)
            print(f"QPE phase: phi = {best_k}/{2 ** n} = {phi_est:.6f}", file=f)
            print(f"Most frequent count: {counts_array[best_k]}/{shots}", file=f)
            print(f"Expected phase: {phi_expected_local:.6f}", file=f)
            print(f"Phase diff (circular): {phase_diff_local:.6e}", file=f)
            if phi_neighbor_local is not None:
                print(f"Neighborhood phase estimate (window={peak_window}): {phi_neighbor_local:.6f}", file=f)
            print("Circuit summary:", file=f)
            print(circuit_info, file=f)

            if plot_path is not None:
                plot(counts_array, best_k=best_k, all_k=all_k, phi_est=phi_est, n=n, shots=shots, output_dir=plot_path)

            return {
                "run_label": str(run_label),
                "phi_peak": float(phi_est),
                "phi_neighbor": phi_neighbor_local,
                "phi_expected": phi_expected_local,
                "phase_diff": phase_diff_local,
                "most_freq_count": int(counts_array[best_k]),
                "elapsed_s": float(elapsed),
                "circuit_summary": circuit_info,
            }

        if args.qpe_run_all_eigenstates:
            runs = []
            for k in range(H.shape[0]):
                psi_k = _normalize_statevector(eigvecs[:, k])
                plot_k = output_dir / f"qpe_results_eigenstate_{k}.png"
                runs.append(_run_one_qpe(psi_k, run_label=f"eigenstate_{k}", expected_e=eigvals[k], plot_path=plot_k))

            summary = {
                "system_or_file": str(pipeline_name),
                "n_phase": int(n),
                "shots": int(shots),
                "run_all_eigenstates": True,
                "n_runs": int(len(runs)),
                "output_dir": str(output_dir),
                "vqe_source": str(args.vqe_source),
                "vqe_ansatz": str(args.vqe_ansatz),
                "runs": runs,
            }
        else:
            overlap_target = int(args.qpe_eigenstate_to_overlap)
            if overlap_target < 0 or overlap_target >= H.shape[0]:
                raise ValueError(f"--qpe-eigenstate-to-overlap must be between 0 and {H.shape[0]-1}.")
            overlap_value = float(abs(np.vdot(eigvecs[:, overlap_target], psi)) ** 2)
            print(f"Overlap |<E_{overlap_target}|psi>|^2 = {overlap_value:.6f}", file=f)

            run = _run_one_qpe(psi, run_label="single", expected_e=None, plot_path=qpe_plot_path)
            summary = {
                "system_or_file": str(pipeline_name),
                "n_phase": int(n),
                "shots": int(shots),
                "run_all_eigenstates": False,
                "psi_source": str(psi_source),
                "overlap_target_idx": int(overlap_target),
                "overlap_target_prob": overlap_value,
                "output_dir": str(output_dir),
                "vqe_source": str(args.vqe_source),
                "vqe_ansatz": str(args.vqe_ansatz),
                **run,
            }

        _write_summary(f, summary)

    print(f"Log saved in: {log_path}")
    if args.qpe_run_all_eigenstates:
        print(f"Plots saved in: {output_dir}")
    else:
        print(f"Plot saved in: {qpe_plot_path}")
    print("\n" + "=" * 19 + " ENDING QPE " + "=" * 19 + "\n")


if __name__ == "__main__":
    main()
