#!/usr/bin/env python3
from pathlib import Path

import argparse
import hashlib
import json
import os
import time
from contextlib import redirect_stdout

import numpy as np

try:
    from .analitical import analitical_minimum_energy, pstr_to_matrix
    from .vqe_graph import plot
    from .vqe_runner import run_vqe
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vqe.vqe_code.analitical import analitical_minimum_energy, pstr_to_matrix
    from vqe.vqe_code.vqe_graph import plot
    from vqe.vqe_code.vqe_runner import run_vqe


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_cwd():
    os.chdir(REPO_ROOT)


def build_dense_hamiltonian(hamiltonian_dict):
    H = pstr_to_matrix(next(iter(hamiltonian_dict.keys()))) * 0
    for pauli_string, coeff in hamiltonian_dict.items():
        H += complex(coeff) * pstr_to_matrix(pauli_string)
    return H


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


def _write_summary(file_obj, summary_dict):
    print("SUMMARY_START", file=file_obj)
    print(json.dumps(summary_dict, sort_keys=True), file=file_obj)
    print("SUMMARY_END", file=file_obj)


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


def execute_vqe_run(
    *,
    hamiltonian_dict=None,
    hamiltonian_matrix=None,
    system_name,
    num_spatial_orbitals,
    num_elec,
    results_root,
    shots,
    ansatz,
    two_local_reps,
    two_local_rotation_blocks,
    two_local_entanglement,
    two_local_entanglement_blocks,
    two_local_parameter_prefix,
    uccsd_reps,
    uccsd_generalized,
    uccsd_preserve_spin,
    uccsd_include_imaginary,
    method,
    spsa_a,
    spsa_c,
    spsa_alpha,
    spsa_gamma,
    spsa_stability_offset,
    maxiter,
    seed,
    exact_min_energy=None,
    output_dir_override=None,
):
    if hamiltonian_matrix is not None:
        H = np.array(hamiltonian_matrix, dtype=np.complex128)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("hamiltonian_matrix must be a square matrix.")
        number_of_qubits = int(np.log2(H.shape[0]))
    elif hamiltonian_dict is not None:
        number_of_qubits = len(list(hamiltonian_dict.keys())[0])
        H = build_dense_hamiltonian(hamiltonian_dict)
    else:
        raise ValueError("Either hamiltonian_dict or hamiltonian_matrix must be provided.")

    print("\n" + "=" * 18 + " STARTING VQE " + "=" * 18)
    print(f"System: {system_name}")
    print(f"Data successfully prepared!\n" + "=" * 50)

    start_time = time.time()
    if exact_min_energy is None and hamiltonian_dict is not None:
        min_energy = analitical_minimum_energy(hamiltonian_dict, number_of_qubits)
    elif exact_min_energy is None:
        min_energy = float(np.linalg.eigvalsh(H).min())
    else:
        min_energy = float(exact_min_energy)
    end_time = time.time()
    print(f"Minimum (analitical) energy level: {min_energy}\n" + "=" * 50)
    elapsed_analitical = end_time - start_time

    if output_dir_override:
        output_dir = Path(output_dir_override)
    else:
        results_root = Path(results_root)
        results_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = results_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "vqe_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        start_time = time.time()
        vqe_result, vqe_energies, best_circuit, best_iteration = run_vqe(
            hamiltonian_dict if hamiltonian_dict is not None else H,
            number_of_qubits,
            f,
            num_spatial_orbitals,
            num_elec,
            shots=shots,
            ansatz_type=ansatz,
            two_local_reps=two_local_reps,
            two_local_rotation_blocks=two_local_rotation_blocks,
            two_local_entanglement=two_local_entanglement,
            two_local_entanglement_blocks=two_local_entanglement_blocks,
            two_local_parameter_prefix=two_local_parameter_prefix,
            uccsd_reps=uccsd_reps,
            uccsd_generalized=uccsd_generalized,
            uccsd_preserve_spin=uccsd_preserve_spin,
            uccsd_include_imaginary=uccsd_include_imaginary,
            method=method,
            spsa_a=spsa_a,
            spsa_c=spsa_c,
            spsa_alpha=spsa_alpha,
            spsa_gamma=spsa_gamma,
            spsa_stability_offset=spsa_stability_offset,
            maxiter=maxiter,
            seed=seed,
        )
        end_time = time.time()
        elapsed_vqe = end_time - start_time

        hartree_ev = 27.211386245988
        min_energy_ev = min_energy * hartree_ev
        vqe_result_ev = vqe_result.fun * hartree_ev
        energy_diff = vqe_result.fun - min_energy
        energy_diff_ev = energy_diff * hartree_ev
        percentage = ((-energy_diff / min_energy) * 100) if min_energy != 0 else float("inf")
        decomposed_circuit = best_circuit.decompose()
        circuit_summary = _circuit_summary(decomposed_circuit)
        best_energy = float(vqe_result.fun)
        abs_err = float(abs(energy_diff))
        rel_err = (abs_err / abs(float(min_energy))) if min_energy != 0 else None

        with redirect_stdout(f):
            print(f"Hamiltonian: {hamiltonian_dict}")
            print("=" * 50)
            print(f"Analytical minimum energy for {system_name}: {min_energy} Hartree = {min_energy_ev} eV")
            print(f"Elapsed time: {elapsed_analitical:.4f} s")
            print("=" * 50)
            print(f"VQE Energy: {vqe_result.fun} Hartree = {vqe_result_ev} eV")
            print(f"Error: {energy_diff} Hartree = {energy_diff_ev} eV --> {percentage}%")
            print(f"Shots: {shots}")
            print(f"Time elapsed: {elapsed_vqe:.4f} s")
            print(f"Number of iterations: {len(vqe_energies)}")
            print(f"Energy list: {vqe_energies[:10]} ...")
            print("=" * 50)
            print(f"Best circuit (iteration {best_iteration}) summary:")
            print(circuit_summary)
            plot(vqe_result.fun, elapsed_vqe, vqe_energies, min_energy, system_name, str(output_dir), best_iteration)

            summary = {
                "system": system_name,
                "shots": int(shots),
                "ansatz": str(ansatz),
                "two_local_reps": int(two_local_reps),
                "two_local_rotation_blocks": list(two_local_rotation_blocks),
                "two_local_entanglement": str(two_local_entanglement),
                "two_local_entanglement_blocks": list(two_local_entanglement_blocks),
                "two_local_parameter_prefix": str(two_local_parameter_prefix),
                "uccsd_reps": int(uccsd_reps),
                "uccsd_generalized": bool(uccsd_generalized),
                "uccsd_preserve_spin": bool(uccsd_preserve_spin),
                "uccsd_include_imaginary": bool(uccsd_include_imaginary),
                "method": str(method),
                "spsa_a": float(spsa_a),
                "spsa_c": float(spsa_c),
                "spsa_alpha": float(spsa_alpha),
                "spsa_gamma": float(spsa_gamma),
                "spsa_stability_offset": float(spsa_stability_offset) if spsa_stability_offset is not None else None,
                "seed": seed,
                "best_energy": best_energy,
                "analytic_energy": float(min_energy),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "iterations": int(len(vqe_energies)),
                "elapsed_s": float(elapsed_vqe),
                "output_dir": str(output_dir),
            }
            _write_summary(f, summary)

    print(f"Log and plot saved in: {output_dir}")
    print("\n" + "=" * 19 + " ENDING VQE " + "=" * 19 + "\n")

    return H, min_energy, best_circuit, abs(min_energy)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run VQE from a molecule JSON (qubit_hamiltonian).")
    parser.add_argument(
        "molecule",
        nargs="?",
        default="H2",
        help="Molecule name (e.g. H2, LiH) or explicit path to a molecule JSON file.",
    )
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots.")
    parser.add_argument(
        "--ansatz",
        type=str,
        default="twolocal",
        choices=["twolocal", "uccsd", "1", "2"],
        help="Ansatz type: twolocal or uccsd.",
    )
    parser.add_argument("--two-local-reps", type=int, default=3, help="Repetitions for TwoLocal ansatz.")
    parser.add_argument(
        "--two-local-rotation-blocks",
        type=_parse_csv_list,
        default=["ry"],
        help="Comma-separated rotation blocks for TwoLocal.",
    )
    parser.add_argument(
        "--two-local-entanglement",
        type=str,
        default="linear",
        help="Entanglement pattern for TwoLocal.",
    )
    parser.add_argument(
        "--two-local-entanglement-blocks",
        type=_parse_csv_list,
        default=["cx"],
        help="Comma-separated entanglement blocks for TwoLocal.",
    )
    parser.add_argument(
        "--two-local-parameter-prefix",
        type=str,
        default="theta",
        help="Parameter prefix for TwoLocal.",
    )
    parser.add_argument("--uccsd-reps", type=int, default=2, help="Repetitions for UCCSD ansatz.")
    parser.add_argument(
        "--uccsd-generalized",
        type=_parse_bool,
        default=False,
        help="Enable generalized UCCSD.",
    )
    parser.add_argument(
        "--uccsd-preserve-spin",
        type=_parse_bool,
        default=True,
        help="Preserve spin in UCCSD.",
    )
    parser.add_argument(
        "--uccsd-include-imaginary",
        type=_parse_bool,
        default=True,
        help="Include imaginary excitations in UCCSD.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cobyla",
        choices=["cobyla", "spsa"],
        help="Optimization method to use.",
    )
    parser.add_argument("--spsa-a", type=float, default=0.2, help="SPSA step-size coefficient a.")
    parser.add_argument("--spsa-c", type=float, default=0.1, help="SPSA perturbation coefficient c.")
    parser.add_argument("--spsa-alpha", type=float, default=0.602, help="SPSA decay exponent alpha.")
    parser.add_argument("--spsa-gamma", type=float, default=0.101, help="SPSA perturbation decay exponent gamma.")
    parser.add_argument(
        "--spsa-stability-offset",
        type=float,
        default=None,
        help="Optional SPSA stability offset A; if omitted, a heuristic based on maxiter is used.",
    )
    parser.add_argument("--maxiter", type=int, default=2000, help="Maximum optimization iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible initialization.")
    return parser.parse_args(argv)


def resolve_molecule_path(molecule_arg: str, module_dir: Path) -> Path:
    candidate = Path(molecule_arg)
    if candidate.exists():
        return candidate.resolve()

    repo_root = module_dir.parent.parent
    root_molecules = repo_root / "molecules" / f"{molecule_arg}_sto-3g_qubit_hamiltonian.json"
    if root_molecules.exists():
        return root_molecules.resolve()

    raise FileNotFoundError(
        f"Could not resolve molecule input '{molecule_arg}'. "
        "Provide a valid JSON path or a known molecule name."
    )


def load_molecule_json(molecule_path: Path):
    with open(molecule_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main(cli_args=None):
    _ensure_repo_root_cwd()
    if cli_args is None:
        cli_args = argparse.Namespace(
            molecule="H2",
            shots=1024,
            ansatz="twolocal",
            two_local_reps=3,
            two_local_rotation_blocks=["ry"],
            two_local_entanglement="linear",
            two_local_entanglement_blocks=["cx"],
            two_local_parameter_prefix="theta",
            uccsd_reps=2,
            uccsd_generalized=False,
            uccsd_preserve_spin=True,
            uccsd_include_imaginary=True,
            method="cobyla",
            spsa_a=0.2,
            spsa_c=0.1,
            spsa_alpha=0.602,
            spsa_gamma=0.101,
            spsa_stability_offset=None,
            maxiter=2000,
            seed=None,
        )

    module_dir = Path(__file__).resolve().parent
    molecule_path = resolve_molecule_path(cli_args.molecule, module_dir)
    molecule_data = load_molecule_json(molecule_path)

    if "qubit_hamiltonian" not in molecule_data:
        raise ValueError("Molecule JSON must contain key 'qubit_hamiltonian'.")
    if "n_orbs" not in molecule_data or "n_elec" not in molecule_data:
        raise ValueError("Molecule JSON must contain keys 'n_orbs' and 'n_elec' for VQE.")

    print(f"Loading molecule data from: {molecule_path}")

    results_root = module_dir.parent / "vqe_results" / molecule_data["name"]
    return execute_vqe_run(
        hamiltonian_dict=molecule_data["qubit_hamiltonian"],
        hamiltonian_matrix=None,
        system_name=molecule_data["name"],
        num_spatial_orbitals=molecule_data["n_orbs"],
        num_elec=molecule_data["n_elec"],
        results_root=results_root,
        shots=int(cli_args.shots),
        ansatz=cli_args.ansatz,
        two_local_reps=int(cli_args.two_local_reps),
        two_local_rotation_blocks=cli_args.two_local_rotation_blocks,
        two_local_entanglement=cli_args.two_local_entanglement,
        two_local_entanglement_blocks=cli_args.two_local_entanglement_blocks,
        two_local_parameter_prefix=cli_args.two_local_parameter_prefix,
        uccsd_reps=int(cli_args.uccsd_reps),
        uccsd_generalized=bool(cli_args.uccsd_generalized),
        uccsd_preserve_spin=bool(cli_args.uccsd_preserve_spin),
        uccsd_include_imaginary=bool(cli_args.uccsd_include_imaginary),
        method=cli_args.method,
        spsa_a=float(cli_args.spsa_a),
        spsa_c=float(cli_args.spsa_c),
        spsa_alpha=float(cli_args.spsa_alpha),
        spsa_gamma=float(cli_args.spsa_gamma),
        spsa_stability_offset=(float(cli_args.spsa_stability_offset) if cli_args.spsa_stability_offset is not None else None),
        maxiter=int(cli_args.maxiter),
        seed=cli_args.seed,
    )


if __name__ == "__main__":
    main(parse_args())
