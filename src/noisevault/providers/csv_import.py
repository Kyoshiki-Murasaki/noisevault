from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from ..models import DeviceNoiseSnapshot, GateCalibration, Provenance, QubitCalibration
from ..snapshot_io import raw_payload_hash


def import_ibm_calibration_csv(
    path: str | Path,
    *,
    backend_name: str,
    captured_at: str | None = None,
) -> DeviceNoiseSnapshot:
    """Import a dashboard-style calibration CSV.

    IBM has changed column labels over time, so this importer uses normalized aliases and
    intentionally rejects rows it cannot identify instead of silently inventing values.
    """
    path = Path(path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    normalized = [{key.strip().lower().replace(" ", "_"): value for key, value in row.items()} for row in rows]

    def first(row, *names):
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    qubits: dict[int, QubitCalibration] = {}
    gates: list[GateCalibration] = []
    coupling: set[tuple[int, int]] = set()
    for row in normalized:
        qubit_raw = first(row, "qubit", "qubit_index", "q")
        gate_name = first(row, "gate", "gate_name", "name")
        if gate_name:
            qubit_text = first(row, "qubits", "gate_qubits", "qubit") or ""
            indices = [int(token) for token in qubit_text.replace("[", "").replace("]", "").replace("-", ",").split(",") if token.strip().isdigit()]
            if indices:
                gates.append(
                    GateCalibration(
                        name=str(gate_name).lower(),
                        qubits=indices,
                        error=float(first(row, "gate_error", "error") or 0.0),
                        duration_ns=float(first(row, "gate_length_ns", "duration_ns", "gate_length") or 0.0),
                    )
                )
                if len(indices) == 2:
                    coupling.add((indices[0], indices[1]))
            continue
        if qubit_raw is None:
            continue
        index = int(qubit_raw)
        qubits[index] = QubitCalibration(
            index=index,
            t1_us=float(first(row, "t1_us", "t1") or 0.0) or None,
            t2_us=float(first(row, "t2_us", "t2") or 0.0) or None,
            frequency_ghz=float(first(row, "frequency_ghz", "frequency") or 0.0) or None,
            readout_error=float(first(row, "readout_error") or 0.0),
            prob_meas0_prep1=float(first(row, "prob_meas0_prep1", "p0_given_1") or 0.0),
            prob_meas1_prep0=float(first(row, "prob_meas1_prep0", "p1_given_0") or 0.0),
        )
    if not qubits:
        raise ValueError("No qubit calibration rows could be identified in the CSV.")
    count = max(qubits) + 1
    for index in range(count):
        qubits.setdefault(index, QubitCalibration(index=index, operational=False))
    raw = path.read_text(encoding="utf-8-sig")
    return DeviceNoiseSnapshot(
        provider="ibm_csv",
        backend_name=backend_name,
        captured_at=captured_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        num_qubits=count,
        coupling_map=[list(edge) for edge in sorted(coupling)],
        basis_gates=sorted({gate.name for gate in gates}),
        qubits=[qubits[i] for i in range(count)],
        gates=gates,
        provenance=Provenance(source=f"IBM calibration CSV: {path.name}", raw_hash=raw_payload_hash(raw)),
    )
