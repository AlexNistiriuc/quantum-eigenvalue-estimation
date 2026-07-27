# simulator_backend.py

"""
Factory for building Qiskit Aer simulators with optional GPU support and optimizations.
"""

import os

from qiskit_aer import AerSimulator


def _gpu_max_qubits() -> int:
    env_value = os.getenv("VQE_GPU_BATCHED_SHOTS_MAX_QUBITS")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return 16


def build_simulator(
    use_gpu: bool = False,
    custatevec_enable: bool = True,
    batched_shots_gpu: bool = True,
) -> AerSimulator:
    """
    Build and return an AerSimulator configured for CPU or GPU with optimizations.
    
    Args:
        use_gpu (bool): If True, use GPU backend when available. If False, use CPU (default).
        custatevec_enable (bool): If True and use_gpu=True, enable cuStateVec acceleration
            via cuQuantum library for faster state vector operations (default: True).
        batched_shots_gpu (bool): If True and use_gpu=True, enable multi-shot batching on GPU
            for improved throughput (default: True).
        
    Returns:
        AerSimulator: Configured simulator instance.
        
    Raises:
        RuntimeError: If GPU is requested but GPU/Aer support is unavailable.
    """
    if not use_gpu:
        # CPU backend (default) — optimizations not applicable
        return AerSimulator(method="statevector")
    
    # GPU backend requested — apply optimizations
    try:
        options = {
            "method": "statevector",
            "device": "GPU",
        }
        
        # Add GPU-specific optimizations
        if custatevec_enable:
            options["cuStateVec_enable"] = True
        
        if batched_shots_gpu:
            options["batched_shots_gpu"] = True
            options["batched_shots_gpu_max_qubits"] = _gpu_max_qubits()
        
        simulator = AerSimulator(**options)
        
        # Verify GPU availability by checking device
        if simulator.options.device != "GPU":
            raise RuntimeError(
                "GPU backend not available on this device. Ensure the GPU-capable Aer package "
                "is installed and the selected backend supports GPU execution."
            )
        return simulator
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize GPU backend: {e}\n"
            "GPU Requirements:\n"
            "  - A GPU-capable Qiskit Aer installation\n"
            "  - A working GPU driver/runtime for the detected device\n"
            "  - Install the GPU build of Aer if your environment requires it\n"
            "Check that the GPU is visible and supported by your local runtime."
        ) from e
