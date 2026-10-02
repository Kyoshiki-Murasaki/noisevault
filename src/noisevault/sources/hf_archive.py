"""IBM calibration history from the Hugging Face dataset phanerozoic/qiskit-calibration-drift.

The dataset's poller reads IBM's ``backend.properties()`` every 30 minutes and keeps one row per
new calibration of a property: ``(backend, property, qubit_a, qubit_b, calibrated_time)``. The
importer never reads its ``SN`` column, which is CC-BY-NC-4.0.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple

from ..errors import (
    SourceDataError,
    SourceUnavailable,
    did_you_mean,
    install_hint,
    joined,
    qubit_loci,
)
from ..gates import is_symmetric
from ..profile import Profile, iso_z
from . import OLDER_HINT, Origin
from .qiskit_backend import as_utc, calibration_from_properties, to_profile

if TYPE_CHECKING:
    import pyarrow as pa

DATASET = "phanerozoic/qiskit-calibration-drift"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
DOWNLOAD = f"hf download {DATASET} --repo-type dataset --include 'data/*'"

_COLUMN_TYPES = {
    "backend": "string",
    "property": "string",
    "qubit_a": "number",
    "qubit_b": "number",
    "value": "number",
    "unit": "string",
    "calibrated_time": "timestamp",
    "observed_time": "timestamp",
}
_GATE_PARAMETERS = ("gate_error", "gate_length", "threshold")
_IN_SECONDS_WITHOUT_UNIT = {"T1", "T2"}
_PROCESSORS = {
    "ibm_fez": "Heron r2",
    "ibm_kingston": "Heron r2",
    "ibm_marrakesh": "Heron r2",
    "ibm_torino": "Heron r1",
}
_READOUT_PAIR = {"p1_given_0": "prob_meas1_prep0", "p0_given_1": "prob_meas0_prep1"}
_REVISION = re.compile(r"[0-9a-f]{40}")
_STALE_AFTER = timedelta(days=7)
_QUBIT_VALUE_NAMES = {
    "T1": "T1",
    "T2": "T2",
    "init_error": "prep",
    "readout_error": "readout",
    "readout_length": "readout",
    "prob_meas0_prep1": "readout",
    "prob_meas1_prep0": "readout",
}
_READ_FROM_QUBIT_ROWS_OR_VIRTUAL = frozenset({"measure", "rz"})


class ArchiveSpan(NamedTuple):
    """The ``at`` times the archive can answer for one device.

    ``first`` is when the archive first recorded the device, the earliest ``at`` accepted.
    ``last`` is the device's newest calibration, the default ``at``.
    """

    first: datetime
    last: datetime


def calibration_archive_devices(path: str | Path) -> dict[str, ArchiveSpan]:
    """Each device in a local copy of the dataset's parquet file, with its ``at`` range."""
    table = _read(Path(path), ("backend", "observed_time", "calibrated_time"))
    spans = table.group_by("backend").aggregate(
        [("observed_time", "min"), ("calibrated_time", "max")]
    )
    return {
        row["backend"]: ArchiveSpan(
            as_utc(row["observed_time_min"]), as_utc(row["calibrated_time_max"])
        )
        for row in spans.sort_by("backend").to_pylist()
    }


def from_calibration_archive(
    path: str | Path, device: str, *, at: str | date | datetime | None = None
) -> Profile:
    """The calibration of ``device`` at ``at`` from a local copy of the dataset's parquet file.

    Each property takes its newest row calibrated at or before ``at``; by default, its newest
    row. ``at`` is a datetime, a date or an ISO 8601 string, read as UTC when it has no zone.
    A provenance note names the values calibrated more than 7 days before ``at``.
    """
    path = Path(path)
    name = device.strip().lower()
    rows = _read(path, tuple(_COLUMN_TYPES), name)
    if rows.num_rows == 0:
        held = sorted(calibration_archive_devices(path))
        raise SourceDataError(
            f"{path.name} has no rows for {name}; it holds {', '.join(held)}",
            hint=did_you_mean(name, held).strip() or None,
        )
    first = as_utc(_pyarrow().compute.min(rows["observed_time"]).as_py())
    stamp = None if at is None else as_utc(at, name="at")
    if stamp is not None and stamp < first:
        raise SourceDataError(
            f"{path.name} has no complete {name} calibration before {iso_z(first)},"
            f" when the archive first recorded {name}",
            hint="pass an at= time on or after that",
        )
    picked = _newest(rows, stamp)
    origin = Origin(f"the {name} rows of {path.name}", hint=OLDER_HINT)
    cal = calibration_from_properties(_backend_properties(name, picked), origin=origin)
    notes = [
        "The dataset is CC-BY-4.0, but its numbers are IBM Quantum calibrations under IBM's"
        " terms, so redistributable is unknown, as for nv pull."
    ]
    if all(row["unit"] is None for row in picked):
        notes.append(
            "Every row behind this profile was recorded before 8 May 2026, when the archive began"
            " to record units. Those rows hold only T1, T2, the readout errors and the sx and cz"
            " errors, so this profile has no gate or readout durations and no other gates."
        )
    elif stamp is not None:
        newest = calibration_from_properties(
            _backend_properties(name, _newest(rows, None)),
            origin=Origin(f"the newest {name} rows of {path.name}"),
        )
        missing = sorted({i.name for i in newest.instructions} - {i.name for i in cal.instructions})
        if missing:
            notes.append(
                f"This profile has no {joined(missing, 'or')} gate. The archive calibrates"
                f" {joined(missing, 'and')} on {name} only after {iso_z(stamp)}."
            )
    stale = _stale_note(picked, stamp, cal.skipped)
    if stale:
        notes.append(stale)
    dead = {}
    for index, qubit in sorted(cal.qubits.items()):
        stuck = [label for field, label in _READOUT_PAIR.items() if getattr(qubit, field) == 1]
        if stuck:
            dead[index] = replace(qubit, operational=False)
            notes.append(
                f"Qubit {index} is disabled: its readout calibration gives {stuck[0]} = 1."
            )
    cal = replace(cal, processor=_PROCESSORS.get(name), qubits={**cal.qubits, **dead})
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    revision = _revision(path.absolute())
    source = f"Hugging Face dataset {DATASET} (CC-BY-4.0)"
    source_url, extra = DATASET_URL, {}
    if revision:
        sha, in_repo = revision
        source += f", revision {sha[:12]}"
        source_url, extra = f"{DATASET_URL}/blob/{sha}/{in_repo}", {"revision": sha}
    return to_profile(
        cal,
        {
            "data_kind": "measured",
            "source_kind": "published_data",
            "source": f"{source}, converted by NoiseVault",
            "source_url": source_url,
            "license": "CC-BY-4.0",
            "attribution": f"IBM Quantum, via {DATASET}",
            "redistributable": "unknown",
            "source_hash": f"sha256:{digest}",
            "notes": notes,
            "extra": extra,
        },
        origin=origin,
    )


def bundled_profiles() -> list[Profile]:
    """None: the archive's numbers are IBM's, whose terms are not an open license."""
    return []


def _pyarrow() -> ModuleType:
    try:
        import pyarrow
        import pyarrow.compute
        import pyarrow.parquet
    except ImportError:
        raise SourceUnavailable(
            "reading the calibration archive needs pyarrow, which is not installed",
            hint=install_hint("hf"),
        ) from None
    return pyarrow


def _read(path: Path, columns: tuple[str, ...], device: str | None = None) -> pa.Table:
    arrow = _pyarrow()
    try:
        schema = arrow.parquet.read_schema(path)
        missing = [c for c in _COLUMN_TYPES if c not in schema.names]
        if missing:
            raise SourceDataError(
                f"{path.name} is not a {DATASET} data file: it has no {joined(missing, 'or')}"
                " column"
            )
        for column in columns:
            kind, expected = schema.field(column).type, _COLUMN_TYPES[column]
            if not _is(arrow.types, kind, expected):
                raise SourceDataError(
                    f"{path.name}: column {column} holds {kind}, expected a {expected}"
                )
        filters = None if device is None else [("backend", "=", device)]
        return arrow.parquet.read_table(path, columns=list(columns), filters=filters)
    except FileNotFoundError:
        raise
    except (arrow.ArrowException, OSError) as exc:
        raise SourceDataError(
            f"{path.name} cannot be read as parquet: {str(exc).rstrip('.')}",
            hint=f"get the data file with {DOWNLOAD}; a git clone without Git LFS gives a small"
            " pointer file instead",
        ) from None


def _is(types: Any, kind: Any, expected: str) -> bool:
    if expected == "string":
        return types.is_string(kind) or types.is_large_string(kind)
    if expected == "number":
        return types.is_integer(kind) or types.is_floating(kind)
    return types.is_timestamp(kind)


def _newest(rows: pa.Table, at: datetime | None) -> list[dict[str, Any]]:
    arrow = _pyarrow()
    pc = arrow.compute
    if at is not None:
        limit = arrow.scalar(at, type=rows.schema.field("calibrated_time").type)
        rows = rows.filter(pc.less_equal(rows["calibrated_time"], limit))
    # Arrow joins never match a null key, and one-qubit rows have a null qubit_b.
    rows = rows.append_column("pair", pc.fill_null(rows["qubit_b"], -1))
    key = ["property", "qubit_a", "pair"]
    latest = rows.group_by(key).aggregate([("calibrated_time", "max")])
    latest = latest.select([*key, "calibrated_time_max"]).rename_columns([*key, "calibrated_time"])
    picked = rows.join(latest, keys=[*key, "calibrated_time"], join_type="inner")
    return picked.sort_by([(column, "ascending") for column in [*key, "value"]]).to_pylist()


def _backend_properties(device: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    qubits: dict[int, list[dict[str, Any]]] = {}
    gates: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    top = -1
    for row in rows:
        name, unit = row["property"], row["unit"]
        if unit is None and name in _IN_SECONDS_WITHOUT_UNIT:
            unit = "s"
        locus = _locus(row)
        top = max(top, *locus)
        split = _gate_parameter(name)
        if split is None:
            qubits.setdefault(locus[0], []).append(
                {"name": name, "value": row["value"], "unit": unit}
            )
        else:
            gate, param = split
            gates.setdefault((gate, locus), []).append(
                {"name": param, "value": row["value"], "unit": unit}
            )
    return {
        "backend_name": device,
        "last_update_date": max(row["calibrated_time"] for row in rows),
        "qubits": [qubits.get(i, []) for i in range(top + 1)],
        "gates": [
            {"gate": gate, "qubits": list(locus), "parameters": params}
            for (gate, locus), params in gates.items()
        ],
    }


def _locus(row: dict[str, Any]) -> tuple[int, ...]:
    a, b = int(row["qubit_a"]), row["qubit_b"]
    return (a,) if b is None else (a, int(b))


def _gate_parameter(name: str) -> tuple[str, str] | None:
    param = next((p for p in _GATE_PARAMETERS if name.endswith("_" + p)), None)
    return None if param is None else (name.removesuffix("_" + param), param)


def _stale_note(
    rows: list[dict[str, Any]], at: datetime | None, unconverted: tuple[str, ...]
) -> str | None:
    before = at or max(row["calibrated_time"] for row in rows)
    oldest: dict[str, datetime] = {}
    loci: dict[str, set[tuple[int, ...]]] = {}
    for row in rows:
        calibrated = row["calibrated_time"]
        if before - calibrated <= _STALE_AFTER:
            continue
        name = row["property"]
        split = _gate_parameter(name)
        if split is None:
            label = _QUBIT_VALUE_NAMES.get(name)
        elif split[0] not in _READ_FROM_QUBIT_ROWS_OR_VIRTUAL:
            label = split[0]
        else:
            continue
        if label is None or label in unconverted:
            continue
        locus = _locus(row)
        if len(locus) == 2 and is_symmetric(label):
            locus = tuple(sorted(locus))
        loci.setdefault(label, set()).add(locus)
        oldest[label] = min(oldest.get(label, calibrated), calibrated)
    if not loci:
        return None
    stale = sorted((oldest[label], label, sorted(on)) for label, on in loci.items())
    named = [f"{label} on {qubit_loci(*on)}" for _, label, on in stale]
    if len(named) > 4:
        named = [*named[:3], f"also {joined([label for _, label, _ in stale[3:]], 'and')}"]
    when = "the newest calibration" if at is None else iso_z(at)
    return (
        f"These values were calibrated more than {_STALE_AFTER.days} days before {when}, the"
        f" oldest on {stale[0][0].date().isoformat()}: {'; '.join(named)}."
    )


def _revision(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for i, part in enumerate(parts[:-2]):
        if part == "snapshots" and _REVISION.fullmatch(parts[i + 1]):
            return parts[i + 1], "/".join(parts[i + 2 :])
    return None
