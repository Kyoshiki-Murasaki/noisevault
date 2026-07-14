from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters.qiskit_adapter import _sequence_to_quantum_error, to_qiskit_aer
from .channels import channel_sequence_for_operation, find_gate_calibration
from .metrics import total_variation_distance


@dataclass
class NativeRoundtripResult:
    status: str
    backend_name: str
    num_qubits: int
    circuits: list[dict[str, object]]
    max_tvd: float | None
    calibration_checks: list[dict[str, object]]
    notes: list[str]


def _backend_name(backend) -> str:
    value = getattr(backend, "name", None)
    return value() if callable(value) else str(value)


def _canonical_diagonal(rho: np.ndarray, n: int) -> np.ndarray:
    diagonal = np.real_if_close(np.diag(rho)).real
    canonical = np.zeros_like(diagonal, dtype=float)
    for canonical_index in range(2**n):
        bits = [(canonical_index >> (n - 1 - wire)) & 1 for wire in range(n)]
        qiskit_index = sum(bit << wire for wire, bit in enumerate(bits))
        canonical[canonical_index] = diagonal[qiskit_index]
    return canonical / canonical.sum()


def compare_native_aer(backend, snapshot, num_qubits: int = 2) -> NativeRoundtripResult:
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
    except ImportError as exc:
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            [],
            [f"Qiskit dependencies unavailable: {exc}"],
        )

    if snapshot.num_qubits < num_qubits:
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            [],
            ["Backend has fewer qubits than requested."],
        )
    prefix = list(range(num_qubits))
    basis = set(snapshot.basis_gates)
    one_gate = next(
        (
            name
            for name in ("sx", "x")
            if name in basis
            and all(find_gate_calibration(snapshot, name, (q,)) is not None for q in prefix)
        ),
        None,
    )
    two_gate = next(
        (
            name
            for name in ("cx", "ecr", "cz")
            if name in basis
            and all(
                find_gate_calibration(snapshot, name, (q, q + 1)) is not None
                for q in range(num_qubits - 1)
            )
        ),
        None,
    )
    if one_gate is None or (num_qubits > 1 and two_gate is None):
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            [],
            [f"Unsupported basis for native comparison: {sorted(basis)}"],
        )

    try:
        native = NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=False,
            thermal_relaxation=True,
        )
        reconstructed = to_qiskit_aer(snapshot, prefix, include_readout=False).noise_model
    except Exception as exc:
        return NativeRoundtripResult(
            "error",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            [],
            [f"Could not construct noise models: {type(exc).__name__}: {exc}"],
        )

    from qiskit.quantum_info import average_gate_fidelity

    calibration_checks = []
    check_specs = [(one_gate, (q,)) for q in prefix]
    if "rz" in basis:
        check_specs.extend(("rz", (q,)) for q in prefix)
    if two_gate is not None:
        check_specs.extend((two_gate, (q, q + 1)) for q in range(num_qubits - 1))
    for name, physical_qargs in check_specs:
        calibration = find_gate_calibration(snapshot, name, physical_qargs)
        if calibration is None or calibration.error is None:
            continue
        logical_qargs = physical_qargs
        sequence = channel_sequence_for_operation(
            snapshot, name, logical_qargs, physical_qargs
        )
        reconstructed_error = _sequence_to_quantum_error(sequence, logical_qargs)
        reconstructed_infidelity = (
            0.0
            if reconstructed_error is None
            else 1.0 - float(average_gate_fidelity(reconstructed_error.to_quantumchannel()))
        )
        relaxation_only = [channel for channel in sequence if channel.kind == "thermal_relaxation"]
        relaxation_error = _sequence_to_quantum_error(relaxation_only, logical_qargs)
        relaxation_infidelity = (
            0.0
            if relaxation_error is None
            else 1.0 - float(average_gate_fidelity(relaxation_error.to_quantumchannel()))
        )
        native_model_target = max(float(calibration.error), relaxation_infidelity)
        calibration_checks.append(
            {
                "gate": name,
                "qargs": list(physical_qargs),
                "reported_avg_gate_error": float(calibration.error),
                "relaxation_avg_gate_error": relaxation_infidelity,
                "reported_target_reachable": calibration.error >= relaxation_infidelity,
                "native_model_target_avg_gate_error": native_model_target,
                "reconstructed_avg_gate_error": reconstructed_infidelity,
                "absolute_difference": abs(reconstructed_infidelity - native_model_target),
            }
        )

    circuits = []
    for variant in range(4):
        qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            if one_gate == "sx":
                qc.sx(q)
            else:
                qc.x(q)
            qc.rz((variant + 1) * np.pi / 7, q)
        if num_qubits > 1:
            for q in range(num_qubits - 1):
                getattr(qc, two_gate)(q, q + 1)
        if variant % 2:
            for q in range(num_qubits):
                getattr(qc, one_gate)(q)
        qc.save_density_matrix()

        distributions = {}
        for label, noise_model in (("native", native), ("reconstructed", reconstructed)):
            simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
            result = simulator.run(qc).result()
            rho = np.asarray(result.data(0)["density_matrix"], dtype=complex)
            distributions[label] = _canonical_diagonal(rho, num_qubits)
        tvd = total_variation_distance(distributions["native"], distributions["reconstructed"])
        circuits.append(
            {
                "name": f"basis_variant_{variant}",
                "tvd": tvd,
                "native": distributions["native"].tolist(),
                "reconstructed": distributions["reconstructed"].tolist(),
            }
        )

    max_tvd = max(float(row["tvd"]) for row in circuits)
    return NativeRoundtripResult(
        "complete",
        _backend_name(backend),
        num_qubits,
        circuits,
        max_tvd,
        calibration_checks,
        [
            "Comparison uses prefix qubits so native and reconstructed qargs align without private NoiseModel remapping.",
            "Every entangling operation uses an exact direction-aligned calibration.",
            "Residual depolarization is solved after relaxation so the composed channel targets the archived average gate error.",
            "Readout error is disabled in both models; Experiment B/D apply readout through the shared exact post-processor.",
        ],
    )
