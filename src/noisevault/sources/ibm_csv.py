"""IBM calibration CSV files as downloaded from the IBM Quantum platform.

The file has one row per qubit. Two-qubit errors and durations are packed into one cell per
row, either as ``1_2:0.0106; 1_0:0.0053`` (explicit pairs, 2023 to 2025 files) or as
``1:0.0013;10:0.0011`` (partners of the row's qubit, 2026 files). Headers may carry trailing
spaces and any time unit in parentheses.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import units
from ..profile import Profile
from .qiskit_backend import (
    Calibration,
    Instruction,
    QubitCalibration,
    as_utc,
    sha256_bytes,
    to_profile,
)

# normalized header stem -> gate; a column "<stem> error" holds that gate's error
_GATE_COLUMNS = {
    "id": "id",
    "√x (sx)": "sx",
    "sx": "sx",
    "pauli-x": "x",
    "x": "x",
    "rx": "rx",
    "z-axis rotation (rz)": "rz",
    "rz": "rz",
    "cnot": "cx",
    "cx": "cx",
    "ecr": "ecr",
    "cz": "cz",
    "rzz": "rzz",
}
_QUBIT_COLUMNS = {
    "t1": "t1",
    "t2": "t2",
    "readout assignment error": "readout_error",
    "prob meas0 prep1": "p0_given_1",
    "prob meas1 prep0": "p1_given_0",
    "readout length": "readout_length",
    "single-qubit gate length": "gate_length_1q",
    "gate time": "gate_length_2q",
    "gate length": "gate_length_2q",
    "operational": "operational",
}
_REDUNDANT = {"measure error", "frequency", "anharmonicity"}  # read and deliberately unused
_TIME_UNITS = {
    "t1": "us",
    "t2": "us",
    "readout_length": "ns",
    "gate_length_1q": "ns",
    "gate_length_2q": "ns",
}
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}
_HEADER = re.compile(r"^(?P<stem>.*?)\s*(?:\((?P<unit>[^()]*)\))?$")
_UNITS = {"µs": "us", "μs": "us"}
_SHOWN_HEADERS = 12  # an error message lists at most this many of the headers it found


@dataclass(frozen=True)
class _Column:
    header: str  # as written in the file
    key: str  # a _QUBIT_COLUMNS value, or "error:<gate>"
    unit: str | None


def from_ibm_csv(path: str | Path, *, device: str, calibrated_at: str | datetime) -> Profile:
    """A profile from an IBM calibration CSV; ``device`` is e.g. ``"ibm_brisbane"``."""
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if text.lstrip().startswith(("{", "[")):
        raise ValueError(
            f"{path.name} looks like IBM properties JSON, not a calibration CSV; for a device use"
            " nv.pull(<name>), or from_qiskit_backend(<backend>) for a Qiskit backend"
        )
    reader = csv.DictReader(io.StringIO(text))
    columns, ignored = _columns(reader.fieldnames or [], path.name)
    rows = [_row(row, columns, path.name, line) for line, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError(f"{path.name} has a header but no qubit rows")
    cal = _calibration(rows, device, as_utc(calibrated_at, name="calibrated_at"))
    notes = [f"Ignored columns: {', '.join(ignored)}."] if ignored else []
    return to_profile(
        cal,
        {
            "data_kind": "measured",
            "source_kind": "user_file",
            "source": f"IBM calibration CSV {path.name}",
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "source_hash": sha256_bytes(raw),
            "notes": notes,
        },
    )


def _columns(headers: list[str], name: str) -> tuple[dict[str, _Column], list[str]]:
    columns: dict[str, _Column] = {}
    ignored = []
    for header in headers:
        match = _HEADER.match(" ".join(header.split()).lower())
        stem, unit = match["stem"], match["unit"]  # type: ignore[index]
        full = " ".join(header.split()).lower()
        if full.endswith(" error") and full.removesuffix(" error") in _GATE_COLUMNS:
            key = "error:" + _GATE_COLUMNS[full.removesuffix(" error")]
            columns[key] = _Column(header, key, None)
        elif stem in _QUBIT_COLUMNS:
            key = _QUBIT_COLUMNS[stem]
            columns[key] = _Column(header, key, _UNITS.get(unit, unit) if unit else None)
        elif header.strip() and stem != "qubit" and stem not in _REDUNDANT:
            ignored.append(header.strip())
        if stem == "qubit":
            columns["qubit"] = _Column(header, "qubit", None)
    missing = [label for label, ok in _required(columns).items() if not ok]
    if missing:
        shown = [repr(h.strip()) for h in headers[:_SHOWN_HEADERS]]
        more = ", ..." if len(headers) > _SHOWN_HEADERS else ""
        found = (", ".join(shown) + more) or "none"
        raise ValueError(
            f"{name} is not an IBM calibration CSV this reader knows: it has no"
            f" {' and no '.join(missing)} column (found: {found}); download the calibration CSV"
            " from the device's page on the IBM Quantum platform"
        )
    return columns, ignored


def _required(columns: Mapping[str, _Column]) -> dict[str, bool]:
    return {
        "'Qubit'": "qubit" in columns,
        "'T1 (us)'": "t1" in columns,
        "'T2 (us)'": "t2" in columns,
        "readout error ('Readout assignment error' or 'Prob meas0 prep1' + 'Prob meas1 prep0')": (
            "readout_error" in columns or {"p0_given_1", "p1_given_0"} <= columns.keys()
        ),
        "gate error (e.g. '√x (sx) error')": any(k.startswith("error:") for k in columns),
    }


@dataclass(frozen=True)
class _Row:
    where: str  # "<file> line <n>", for error messages
    index: int
    values: dict[str, float | None]
    operational: bool
    packed: dict[str, dict[tuple[int, int], float]]  # key -> {pair: value} for 2-qubit cells


_TWO_QUBIT_KEYS = {"error:cx", "error:ecr", "error:cz", "error:rzz", "gate_length_2q"}


def _row(
    row: Mapping[str, str | None], columns: Mapping[str, _Column], name: str, line: int
) -> _Row:
    def cell(column: _Column) -> str:
        return (row.get(column.header) or "").strip()

    where = f"{name} line {line}"
    text = cell(columns["qubit"])
    try:
        index = _qubit(text)
    except ValueError:
        qubit = columns["qubit"].header.strip()
        raise ValueError(f"{where}, {qubit!r}: {text!r} is not a qubit number") from None
    values: dict[str, float | None] = {}
    packed: dict[str, dict[tuple[int, int], float]] = {}
    operational = True
    for key, column in columns.items():
        text = cell(column)
        if key == "qubit":
            continue
        if key == "operational":
            operational = _flag(text, f"{where}, {column.header.strip()!r}")
        elif key in _TWO_QUBIT_KEYS:
            packed[key] = _pairs(text, index, f"{where}, {column.header.strip()!r}")
            if key == "gate_length_2q":
                packed[key] = {p: _time(v, column, key) for p, v in packed[key].items()}
        else:
            number = _number(text, f"{where}, {column.header.strip()!r}")
            values[key] = None if number is None else _time(number, column, key)
    return _Row(where, index, values, operational, packed)


def _time(value: float, column: _Column, key: str) -> float:
    target = _TIME_UNITS.get(key)
    if target is None:
        return value
    return units.convert(value, column.unit or target, target)


def _number(text: str, where: str) -> float | None:
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{where}: {text!r} is not a number") from None


def _qubit(text: str) -> int:
    """A qubit number; a spreadsheet may have saved 3 as 3.0."""
    number = float(text)
    if not number.is_integer() or number < 0:
        raise ValueError(text)
    return int(number)


def _flag(text: str, where: str) -> bool:
    if not text or text.lower() in _TRUE:
        return True
    if text.lower() in _FALSE:
        return False
    raise ValueError(f"{where}: {text!r} is not yes/no or true/false")


def _pairs(text: str, row_qubit: int, where: str) -> dict[tuple[int, int], float]:
    """``1_2:0.01; 1_0:0.02`` (explicit pairs) or ``2:0.01;0:0.02`` (partners of the row)."""
    found: dict[tuple[int, int], float] = {}
    for item in filter(None, (part.strip() for part in text.split(";"))):
        locus, sep, value = item.partition(":")
        try:
            if not sep:
                raise ValueError
            qubits = [_qubit(q) for q in locus.strip().split("_")]
            pair = (row_qubit, qubits[0]) if len(qubits) == 1 else (qubits[0], qubits[1])
            if len(qubits) > 2:
                raise ValueError
            number = float(value)
        except ValueError:
            raise ValueError(f"{where}: {item!r} is not 'a_b:value' or 'partner:value'") from None
        if found.setdefault(pair, number) != number:
            raise ValueError(f"{where}: pair {pair} is given twice, as {found[pair]} and {number}")
    return found


def _calibration(rows: list[_Row], device: str, calibrated_at: datetime) -> Calibration:
    found: dict[tuple[str, tuple[int, ...]], tuple[Instruction, str]] = {}

    def add(inst: Instruction, where: str) -> None:
        first, first_where = found.setdefault((inst.name, inst.qubits), (inst, where))
        if first != inst:
            raise ValueError(
                f"{where}: {inst.name} on qubits {inst.qubits} has error {inst.error} and"
                f" duration {inst.duration_ns} ns, but {first_where} gives error {first.error}"
                f" and duration {first.duration_ns} ns"
            )

    with_error_either_way = {
        (key, locus)
        for row in rows
        for key, cells in row.packed.items()
        if key != "gate_length_2q"
        for pair in cells
        for locus in (pair, pair[::-1])
    }
    lines: dict[int, str] = {}
    qubits: dict[int, QubitCalibration] = {}
    seen_rz = False
    for row in rows:
        if row.index in lines:
            raise ValueError(
                f"{row.where}: qubit {row.index} is already listed on {lines[row.index]}"
            )
        lines[row.index] = row.where
        v = row.values
        qubits[row.index] = QubitCalibration(
            t1_us=v.get("t1"),
            t2_us=v.get("t2"),
            p1_given_0=v.get("p1_given_0"),
            p0_given_1=v.get("p0_given_1"),
            operational=row.operational,
        )
        if v.get("readout_error") is not None or v.get("readout_length") is not None:
            add(
                Instruction(
                    "measure", (row.index,), v.get("readout_error"), v.get("readout_length")
                ),
                row.where,
            )
        for key, error in v.items():
            if not key.startswith("error:"):
                continue
            gate = key.removeprefix("error:")
            duration = None if gate == "rz" else v.get("gate_length_1q")
            if error is not None or duration is not None:
                seen_rz |= gate == "rz"
                add(Instruction(gate, (row.index,), error, duration), row.where)
        durations = row.packed.get("gate_length_2q", {})
        for key, cells in row.packed.items():
            if key == "gate_length_2q":
                continue
            gate = key.removeprefix("error:")
            for pair, error in cells.items():
                add(Instruction(gate, pair, error, durations.get(pair)), row.where)
            for pair, duration in durations.items():
                if (key, pair) not in with_error_either_way:
                    add(Instruction(gate, pair, None, duration), row.where)
    if not seen_rz:  # IBM's rz is always virtual; older files leave out its column
        for row in rows:
            add(Instruction("rz", (row.index,), 0.0, 0.0), row.where)
    return Calibration(
        name=device,
        num_qubits=max(qubits) + 1,
        instructions=tuple(inst for inst, _ in found.values()),
        qubits=qubits,
        calibrated_at=calibrated_at,
    )


def bundled_profiles() -> list[Profile]:
    """CSV files are the user's own downloads; none are bundled."""
    return []
