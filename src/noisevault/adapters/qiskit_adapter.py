from __future__ import annotations

from dataclasses import dataclass

from ..channels import channel_sequence_for_operation
from ..models import DeviceNoiseSnapshot
from ..readout import confusion_matrix


@dataclass
class QiskitAerConversion:
    noise_model: object
    logical_to_physical: dict[int, int]
    notes: list[str]


def _lift_one_qubit_error(error, logical_wire: int, qargs: tuple[int, int]):
    from qiskit_aer.noise import pauli_error

    identity = pauli_error([("I", 1.0)])
    # Qiskit tensor ordering is q1 ⊗ q0 for qargs [q0, q1].
    if logical_wire == qargs[0]:
        return identity.tensor(error)
    return error.tensor(identity)


def to_qiskit_aer(
    snapshot: DeviceNoiseSnapshot,
    physical_qubits: list[int],
    *,
    include_readout: bool = True,
) -> QiskitAerConversion:
    """Convert a snapshot subset to a Qiskit Aer ``NoiseModel``."""
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, kraus_error
    except ImportError as exc:
        raise RuntimeError("Install noisevault[pilot] to use the Qiskit converter.") from exc

    model = NoiseModel()
    logical_to_physical = {logical: physical for logical, physical in enumerate(physical_qubits)}
    physical_to_logical = {physical: logical for logical, physical in logical_to_physical.items()}

    calibrated_specs: set[tuple[str, tuple[int, ...]]] = set()
    for gate in snapshot.gates:
        if not gate.operational or not set(gate.qubits).issubset(physical_to_logical):
            continue
        logical_qargs = tuple(physical_to_logical[q] for q in gate.qubits)
        calibrated_specs.add((gate.name, logical_qargs))

    # The abstract pilot circuits use these portable names even when they are mapped to a
    # provider's closest calibrated primitive by channel_sequence_for_operation().
    for logical in logical_to_physical:
        for name in ("h", "x", "rx", "rz"):
            calibrated_specs.add((name, (logical,)))
    for a in logical_to_physical:
        for b in logical_to_physical:
            if a != b:
                calibrated_specs.add(("cx", (a, b)))

    for operation_name, logical_qargs in sorted(calibrated_specs):
        physical_qargs = tuple(logical_to_physical[q] for q in logical_qargs)
        sequence = channel_sequence_for_operation(
            snapshot, operation_name, logical_qargs, physical_qargs
        )
        if not sequence:
            continue
        combined = None
        for channel in sequence:
            error = kraus_error(list(channel.kraus))
            if len(logical_qargs) == 2 and len(channel.wires) == 1:
                error = _lift_one_qubit_error(error, channel.wires[0], logical_qargs)
            combined = error if combined is None else combined.compose(error)
        model.add_quantum_error(combined, operation_name, list(logical_qargs))

    if include_readout:
        by_index = {qubit.index: qubit for qubit in snapshot.qubits}
        for logical, physical in logical_to_physical.items():
            q = by_index[physical]
            model.add_readout_error(
                ReadoutError(confusion_matrix(q.prob_meas0_prep1, q.prob_meas1_prep0)),
                [logical],
            )

    notes = [
        "Gate errors are canonical depolarizing-plus-relaxation channels derived from the snapshot.",
        "The converter includes exact asymmetric single-qubit readout confusion when requested.",
        "The cross-framework density-matrix benchmark disables simulator-native readout and applies one shared classical convention after simulation.",
    ]
    return QiskitAerConversion(model, logical_to_physical, notes)
