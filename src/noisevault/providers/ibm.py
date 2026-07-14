from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models import DeviceNoiseSnapshot, GateCalibration, Provenance, QubitCalibration
from ..snapshot_io import raw_payload_hash, save_snapshot


class CredentialsUnavailable(RuntimeError):
    pass


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", None)
    return value() if callable(value) else str(value)


def _convert(value: float | None, unit: str | None, target: str) -> float | None:
    if value is None:
        return None
    unit = (unit or "").lower()
    if target == "us":
        factors = {"s": 1e6, "ms": 1e3, "us": 1.0, "µs": 1.0, "ns": 1e-3}
    elif target == "ns":
        factors = {"s": 1e9, "ms": 1e6, "us": 1e3, "µs": 1e3, "ns": 1.0}
    elif target == "ghz":
        factors = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}
    else:
        factors = {"": 1.0}
    return float(value) * factors.get(unit, 1.0)


def _parameter_map(parameters: list[Any]) -> dict[str, tuple[float, str | None]]:
    result: dict[str, tuple[float, str | None]] = {}
    for parameter in parameters:
        name = getattr(parameter, "name", None)
        value = getattr(parameter, "value", None)
        if name is not None and value is not None:
            result[str(name)] = (float(value), getattr(parameter, "unit", None))
    return result


def _value(mapping: dict[str, tuple[float, str | None]], name: str, target: str = "") -> float | None:
    item = mapping.get(name)
    if item is None:
        return None
    return _convert(item[0], item[1], target)


def build_snapshot_from_backend(
    backend: Any,
    *,
    source: str | None = None,
    captured_at: datetime | None = None,
) -> DeviceNoiseSnapshot:
    name = _backend_name(backend)
    config = backend.configuration() if hasattr(backend, "configuration") else None
    props = backend.properties() if hasattr(backend, "properties") else None
    if props is None:
        raise RuntimeError(f"Backend {name} exposes no calibration properties.")

    config_dict = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
    props_dict = props.to_dict() if hasattr(props, "to_dict") else {}
    raw = {"configuration": config_dict, "properties": props_dict}

    num_qubits = int(
        getattr(config, "n_qubits", None)
        or getattr(backend, "num_qubits", None)
        or len(getattr(props, "qubits", []))
    )
    coupling_map = getattr(config, "coupling_map", None)
    if coupling_map is None:
        coupling_map = getattr(backend, "coupling_map", None)
        if hasattr(coupling_map, "get_edges"):
            coupling_map = list(coupling_map.get_edges())
    coupling_map = [list(map(int, edge)) for edge in (coupling_map or [])]
    basis_gates = list(getattr(config, "basis_gates", None) or getattr(backend, "operation_names", []))

    faulty_qubits = set()
    try:
        faulty_qubits = set(props.faulty_qubits())
    except Exception:
        pass

    qubits: list[QubitCalibration] = []
    for index, parameters in enumerate(getattr(props, "qubits", [])):
        values = _parameter_map(parameters)
        readout_error = _value(values, "readout_error")
        p01 = _value(values, "prob_meas0_prep1")
        p10 = _value(values, "prob_meas1_prep0")
        if readout_error is not None and p01 is None and p10 is None:
            p01 = p10 = readout_error
        qubits.append(
            QubitCalibration(
                index=index,
                t1_us=_value(values, "T1", "us"),
                t2_us=_value(values, "T2", "us"),
                frequency_ghz=_value(values, "frequency", "ghz"),
                readout_error=readout_error,
                prob_meas0_prep1=p01,
                prob_meas1_prep0=p10,
                operational=index not in faulty_qubits,
            )
        )

    gates: list[GateCalibration] = []
    for gate in getattr(props, "gates", []):
        values = _parameter_map(getattr(gate, "parameters", []))
        gates.append(
            GateCalibration(
                name=str(getattr(gate, "gate", getattr(gate, "name", "unknown"))),
                qubits=[int(q) for q in getattr(gate, "qubits", [])],
                error=_value(values, "gate_error"),
                duration_ns=_value(values, "gate_length", "ns"),
                operational=True,
            )
        )

    timestamp = captured_at or datetime.now(UTC)
    source_timestamp = getattr(props, "last_update_date", None)
    if source_timestamp is not None:
        source_timestamp = source_timestamp.isoformat()
    return DeviceNoiseSnapshot(
        provider="ibm",
        backend_name=name,
        captured_at=timestamp.isoformat().replace("+00:00", "Z"),
        num_qubits=num_qubits,
        coupling_map=coupling_map,
        basis_gates=sorted(set(str(g) for g in basis_gates)),
        qubits=qubits,
        gates=gates,
        provenance=Provenance(
            source=source or f"qiskit backend properties: {type(backend).__module__}.{type(backend).__name__}",
            raw_hash=raw_payload_hash(raw),
            source_timestamp=source_timestamp,
        ),
    )


def snapshot_destination(root: str | Path, snapshot: DeviceNoiseSnapshot) -> Path:
    stamp = snapshot.captured_at.replace(":", "-").replace("+", "_")
    return Path(root) / snapshot.provider / snapshot.backend_name / f"{stamp}.json"


def _runtime_service():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("Install noisevault[pilot] to harvest IBM calibrations.") from exc

    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform"}
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    try:
        return QiskitRuntimeService(**kwargs)
    except Exception as exc:
        raise CredentialsUnavailable(
            "IBM Quantum authentication is unavailable. Set IBM_QUANTUM_TOKEN or save an IBM Quantum Platform account."
        ) from exc


def harvest_ibm(
    output_root: str | Path = "snapshots",
    *,
    max_backends: int = 2,
    backend_names: list[str] | None = None,
) -> list[Path]:
    service = _runtime_service()
    if backend_names:
        backends = [service.backend(name) for name in backend_names]
    else:
        backends = service.backends(simulator=False, operational=True)
        backends = sorted(backends, key=_backend_name)[:max_backends]
    paths: list[Path] = []
    for backend in backends:
        snapshot = build_snapshot_from_backend(backend, source="IBM Quantum Platform live calibration")
        paths.append(save_snapshot(snapshot, snapshot_destination(output_root, snapshot)))
    return paths
