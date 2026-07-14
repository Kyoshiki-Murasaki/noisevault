from __future__ import annotations

from dataclasses import dataclass

from ..channels import CalibrationUnavailableError, channel_sequence_for_operation
from ..models import DeviceNoiseSnapshot


@dataclass
class PennyLaneConversion:
    noise_model: object
    logical_to_physical: dict[int, int]
    notes: list[str]


def to_pennylane(snapshot: DeviceNoiseSnapshot, physical_qubits: list[int]) -> PennyLaneConversion:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise RuntimeError("Install noisevault[pilot] to use the PennyLane converter.") from exc

    logical_to_physical = {logical: physical for logical, physical in enumerate(physical_qubits)}
    operation_types = {
        "h": qml.Hadamard,
        "x": qml.PauliX,
        "rx": qml.RX,
        "rz": qml.RZ,
        "cx": qml.CNOT,
    }
    model_map = {}

    for name, operation_type in operation_types.items():
        arity = 2 if name == "cx" else 1
        if arity == 1:
            wire_tuples = [(wire,) for wire in logical_to_physical]
        else:
            wire_tuples = [
                (a, b)
                for a in logical_to_physical
                for b in logical_to_physical
                if a != b
            ]
        for logical_wires in wire_tuples:
            physical_wires = tuple(logical_to_physical[wire] for wire in logical_wires)
            try:
                channels = channel_sequence_for_operation(
                    snapshot, name, logical_wires, physical_wires
                )
            except CalibrationUnavailableError:
                continue
            if not channels:
                continue

            ordered_wires = qml.BooleanFn(
                lambda op, *, _wires=logical_wires: tuple(op.wires) == _wires,
                name=f"ordered_wires_eq_{logical_wires}",
            )
            condition = qml.noise.op_eq(operation_type) & ordered_wires

            def noise_fn(op, *, _channels=channels, **kwargs):
                del op, kwargs
                for channel in _channels:
                    qml.QubitChannel(list(channel.kraus), wires=list(channel.wires))

            model_map[condition] = noise_fn

    notes = [
        "The returned qml.NoiseModel inserts QubitChannel operations with the canonical Kraus representation.",
        "Directed multi-qubit operations use order-sensitive wire predicates.",
        "Readout error remains common post-processing because PennyLane's cross-conversion support does not uniformly preserve readout errors.",
    ]
    return PennyLaneConversion(qml.NoiseModel(model_map), logical_to_physical, notes)
