from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters.qiskit_adapter import to_qiskit_aer
from .metrics import total_variation_distance


@dataclass
class NativeRoundtripResult:
    status: str
    backend_name: str
    num_qubits: int
    circuits: list[dict[str, object]]
    max_tvd: float | None
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
            [f"Qiskit dependencies unavailable: {exc}"],
        )

    if snapshot.num_qubits < num_qubits:
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            ["Backend has fewer qubits than requested."],
        )
    prefix = list(range(num_qubits))
    edges = {tuple(edge) for edge in snapshot.coupling_map}
    if num_qubits > 1 and not all(
        (i, i + 1) in edges or (i + 1, i) in edges for i in range(num_qubits - 1)
    ):
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
            ["The prefix qubits are not connected; native Aer qarg remapping is intentionally not guessed."],
        )

    basis = set(snapshot.basis_gates)
    one_gate = "sx" if "sx" in basis else "x" if "x" in basis else None
    two_gate = next((name for name in ("cx", "ecr", "cz") if name in basis), None)
    if one_gate is None or (num_qubits > 1 and two_gate is None):
        return NativeRoundtripResult(
            "skipped",
            _backend_name(backend),
            num_qubits,
            [],
            None,
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
            [f"Could not construct noise models: {type(exc).__name__}: {exc}"],
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
        [
            "Comparison uses prefix qubits so native and reconstructed qargs align without private NoiseModel remapping.",
            "Readout error is disabled in both models; Experiment B/D apply readout through the shared exact post-processor.",
        ],
    )
