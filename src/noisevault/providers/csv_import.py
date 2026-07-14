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

    def optional_float(row, *names):
        value = first(row, *names)
        return None if value is None else float(value)

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
                        error=optional_float(row, "gate_error", "error"),
                        duration_ns=optional_float(
                            row, "gate_length_ns", "duration_ns", "gate_length"
                        ),
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
            t1_us=optional_float(row, "t1_us", "t1"),
            t2_us=optional_float(row, "t2_us", "t2"),
            frequency_ghz=optional_float(row, "frequency_ghz", "frequency"),
            readout_error=optional_float(row, "readout_error"),
            prob_meas0_prep1=optional_float(row, "prob_meas0_prep1", "p0_given_1"),
            prob_meas1_prep0=optional_float(row, "prob_meas1_prep0", "p1_given_0"),
        )
    if not qubits:
        raise ValueError("No qubit calibration rows could be identified in the CSV.")
    count = max(qubits) + 1
    for index in range(count):
        qubits.setdefault(index, QubitCalibration(index=index, operational=False))
    raw = path.read_text(encoding="utf-8-sig")
    archive_timestamp = captured_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return DeviceNoiseSnapshot(
        provider="ibm_csv",
        backend_name=backend_name,
        captured_at=archive_timestamp,
        num_qubits=count,
        coupling_map=[list(edge) for edge in sorted(coupling)],
        basis_gates=sorted({gate.name for gate in gates}),
        qubits=[qubits[i] for i in range(count)],
        gates=gates,
        provenance=Provenance(
            source=f"IBM calibration CSV: {path.name}",
            raw_hash=raw_payload_hash(raw),
            source_timestamp=captured_at,
            notes=(
                []
                if captured_at is not None
                else [
                    "Calibration/source timestamp was unavailable; captured_at is the import time and this snapshot is ineligible for temporal-drift evidence."
                ]
            ),
        ),
    )
