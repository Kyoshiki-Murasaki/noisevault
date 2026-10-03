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
    plural,
    qubit_loci,
)
from ..gates import is_symmetric
from ..profile import Profile, iso_z
from . import OLDER_HINT, Origin
from .qiskit_backend import (
    Calibration,
    _is_sentinel,
    _without_invalid_coherence,
    as_utc,
    calibration_from_properties,
    to_profile,
)

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
_COHERENCE_ROWS = {"t1_us": "T1", "t2_us": "T2"}
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

    ``first`` is the earliest ``at`` that gives a profile: the later of the time the archive first
    recorded the device and the device's earliest calibration.
    ``last`` is the device's newest calibration, the default ``at``.
    """

    first: datetime
    last: datetime


def calibration_archive_devices(path: str | Path) -> dict[str, ArchiveSpan]:
    """Each device in a local copy of the dataset's parquet file, with its ``at`` range."""
    path = Path(path)
    table = _read(path, ("backend", "property", "observed_time", "calibrated_time"))
    unnamed = table["backend"].null_count
    if unnamed:
        raise SourceDataError(f"{path.name} has {plural(unnamed, 'row')} with no backend")
    table = table.append_column("usable", _usable(table))
    spans = table.group_by("backend").aggregate(
        [("observed_time", "min"), ("calibrated_time", "max"), ("usable", "min")]
    )
    return {
        row["backend"]: ArchiveSpan(
            _first_usable(path, row["backend"], row["observed_time_min"], row["usable_min"]),
            as_utc(row["calibrated_time_max"]),
        )
        for row in spans.sort_by("backend").to_pylist()
    }


def from_calibration_archive(
    path: str | Path, device: str, *, at: str | date | datetime | None = None
) -> Profile:
    """The calibration of ``device`` at ``at`` from a local copy of the dataset's parquet file.

    Each property takes its newest row calibrated at or before ``at``. With no ``at``, each
    property takes its newest row. ``at`` is a datetime, a date or an ISO 8601 string. An ``at``
    with no time zone is UTC.
    A provenance note names the values calibrated more than 7 days before ``at``.
    """
    path = Path(path)
    name = device.strip().lower()
    rows = _read(path, tuple(_COLUMN_TYPES), name)
    if rows.num_rows == 0:
        held = sorted(calibration_archive_devices(path))
        raise SourceDataError(
            f"{path.name} has no rows for {name}. The file has rows for {', '.join(held)}",
            hint=did_you_mean(name, held).strip() or None,
        )
    pc = _pyarrow().compute
    usable = _usable(rows)
    unplaced = usable.null_count
    recorded = pc.min(rows["observed_time"]).as_py()
    first = _first_usable(path, name, recorded, pc.min(usable).as_py())
    stamp = None if at is None else as_utc(at, name="at")
    if stamp is not None and stamp < first:
        problem = (
            f"has no complete {name} calibration before {iso_z(first)}, when the archive first"
            f" recorded {name}"
            if first == as_utc(recorded)
            else f"has no {name} calibration before {iso_z(first)}, the earliest calibrated_time"
            f" of the {name} rows"
        )
        raise SourceDataError(
            f"{path.name} {problem}", hint="pass an at= time on or after that time"
        )
    origin = Origin(f"the {name} rows of {path.name}", hint=OLDER_HINT)
    picked = _newest(rows, stamp, origin)
    cal = calibration_from_properties(_backend_properties(name, picked), origin=origin)
    notes = [
        "The dataset is CC-BY-4.0, but its numbers are IBM Quantum calibrations under IBM's"
        " terms, so redistributable is unknown, as for nv pull."
    ]
    if unplaced:
        notes.append(
            f"This profile does not use {plural(unplaced, 'row')} of {name} with no property or"
            " no calibrated_time."
        )
    if all(row["unit"] is None for row in picked):
        notes.append(
            "The archive recorded every row behind this profile before 8 May 2026, when the"
            " archive began to record units. Those rows hold only T1, T2, the readout errors and"
            " the sx and cz errors. As a result, this profile has no gate or readout durations and"
            " no other gates."
        )
    elif stamp is not None:
        newest_origin = Origin(f"the newest {name} rows of {path.name}")
        newest = calibration_from_properties(
            _backend_properties(name, _newest(rows, None, newest_origin)), origin=newest_origin
        )
        missing = sorted({i.name for i in newest.instructions} - {i.name for i in cal.instructions})
        if missing:
            notes.append(
                f"This profile has no {joined(missing, 'or')} gate. The archive calibrates"
                f" {joined(missing, 'and')} on {name} only after {iso_z(stamp)}."
            )
    stale = _stale_note(picked, stamp, cal)
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
    """The ``columns`` of ``path``, with each timestamp in UTC. The dataset stores UTC, so a
    timestamp with no time zone is UTC.
    """
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
        table = arrow.parquet.read_table(path, columns=list(columns), filters=filters)
        in_utc = [
            arrow.field(f.name, arrow.timestamp(f.type.unit, "UTC"))
            if arrow.types.is_timestamp(f.type)
            else f
            for f in table.schema
        ]
        return table.cast(arrow.schema(in_utc))
    except FileNotFoundError:
        raise
    except (arrow.ArrowException, OSError) as exc:
        raise SourceDataError(
            f"{path.name} is not a readable parquet file: {str(exc).rstrip('.')}",
            hint=f"get the data file with {DOWNLOAD}. A git clone without Git LFS gives only a"
            " small pointer file",
        ) from None


def _usable(rows: pa.Table) -> pa.ChunkedArray:
    """For each row, its calibrated_time, or null when the row has no property, so a profile
    cannot use it.
    """
    pc = _pyarrow().compute
    return pc.if_else(pc.is_valid(rows["property"]), rows["calibrated_time"], None)


def _first_usable(
    path: Path, device: str, recorded: datetime | None, calibrated: datetime | None
) -> datetime:
    """The earliest ``at`` that gives ``device`` a profile, in UTC. It is the later of
    ``recorded``, the earliest observed_time, and ``calibrated``, the earliest calibrated_time of
    a usable row.

    A SourceDataError names the column when no row of ``device`` has a value in it.
    """
    if recorded is None:
        raise SourceDataError(f"{path.name} has no {device} row with an observed_time")
    if calibrated is None:
        raise SourceDataError(
            f"{path.name} has no {device} row with both a property and a calibrated_time"
        )
    return max(as_utc(recorded), as_utc(calibrated))


def _is(types: Any, kind: Any, expected: str) -> bool:
    if expected == "string":
        return types.is_string(kind) or types.is_large_string(kind)
    if expected == "number":
        return types.is_integer(kind) or types.is_floating(kind)
    return types.is_timestamp(kind)


def _newest(rows: pa.Table, at: datetime | None, origin: Origin) -> list[dict[str, Any]]:
    """The newest row of each property on each locus, with its ``qubits`` read from the row."""
    arrow = _pyarrow()
    pc = arrow.compute
    if at is not None:
        limit = arrow.scalar(at, type=rows.schema.field("calibrated_time").type)
        rows = rows.filter(pc.less_equal(rows["calibrated_time"], limit))
    # Arrow joins never match a null key, and one-qubit rows have a null qubit_b.
    rows = rows.append_column("first", pc.fill_null(rows["qubit_a"], -1))
    rows = rows.append_column("pair", pc.fill_null(rows["qubit_b"], -1))
    key = ["property", "first", "pair"]
    latest = rows.group_by(key).aggregate([("calibrated_time", "max")])
    latest = latest.select([*key, "calibrated_time_max"]).rename_columns([*key, "calibrated_time"])
    picked = rows.join(latest, keys=[*key, "calibrated_time"], join_type="inner")
    ordered = picked.sort_by([(column, "ascending") for column in [*key, "value"]]).to_pylist()
    return [{**row, "qubits": _locus(row, origin)} for row in ordered]


def _backend_properties(device: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    qubits: dict[int, list[dict[str, Any]]] = {}
    gates: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    top = -1
    for row in rows:
        name, unit = row["property"], row["unit"]
        if unit is None and name in _IN_SECONDS_WITHOUT_UNIT:
            unit = "s"
        locus = row["qubits"]
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


def _locus(row: dict[str, Any], origin: Origin) -> tuple[int, ...]:
    """The row's qubits; ``origin`` refuses an index that is not a whole number 0 or more, and a
    qubit_b on a value of one qubit.
    """
    columns = ("qubit_a",) if row["qubit_b"] is None else ("qubit_a", "qubit_b")
    for column in columns:
        value = row[column]
        if value is None or value < 0 or not float(value).is_integer():
            problem = "not a qubit index"
        elif column == "qubit_b" and _gate_parameter(row["property"]) is None:
            problem = f"but {row['property']} is a value of one qubit"
        else:
            continue
        shown = "null" if value is None else repr(value)
        calibrated = iso_z(as_utc(row["calibrated_time"]))
        row_name = f"the {row['property']} row calibrated at {calibrated}"
        raise origin.refuse(f"{column} of {row_name} is {shown}, {problem}")
    return tuple(int(row[column]) for column in columns)


def _gate_parameter(name: str) -> tuple[str, str] | None:
    param = next((p for p in _GATE_PARAMETERS if name.endswith("_" + p)), None)
    return None if param is None else (name.removesuffix("_" + param), param)


def _stale_note(rows: list[dict[str, Any]], at: datetime | None, cal: Calibration) -> str | None:
    """The note on the rows that IBM calibrated more than 7 days before ``at``.

    Only rows that give the profile a value count.
    """
    before = at or max(row["calibrated_time"] for row in rows)
    unread = _unread(cal)
    oldest: dict[str, datetime] = {}
    loci: dict[str, set[tuple[int, ...]]] = {}
    for row in rows:
        calibrated = row["calibrated_time"]
        name = row["property"]
        if before - calibrated <= _STALE_AFTER or (name, row["qubits"]) in unread:
            continue
        split = _gate_parameter(name)
        if split is None:
            label = _QUBIT_VALUE_NAMES.get(name)
        elif split[0] not in _READ_FROM_QUBIT_ROWS_OR_VIRTUAL:
            label = split[0]
        else:
            continue
        if label is None or label in cal.skipped:
            continue
        locus = row["qubits"]
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
        f"IBM calibrated these values more than {_STALE_AFTER.days} days before {when}, the"
        f" oldest on {stale[0][0].date().isoformat()}: {'; '.join(named)}."
    )


def _unread(cal: Calibration) -> set[tuple[str, tuple[int, ...]]]:
    """The property and qubits of each row that gives the profile no value.

    A qubit with both asymmetric readout errors uses them and not its readout_error. A qubit
    without both uses its readout_error, and its readout_length only with that error. A disabled
    gate keeps only the error that disables it. The conversion treats a T1 or T2 that is not a
    positive finite number as missing.
    """
    _, invalid = _without_invalid_coherence(cal)
    unread: set[tuple[str, tuple[int, ...]]] = {
        (prop, (index,)) for key, prop in _COHERENCE_ROWS.items() for index, _ in invalid[key]
    }
    readout = {i.qubits[0]: i.error for i in cal.instructions if i.name == "measure"}
    for index, qubit in cal.qubits.items():
        if qubit.p1_given_0 is not None and qubit.p0_given_1 is not None:
            unread.add(("readout_error", (index,)))
            continue
        unread |= {(name, (index,)) for name in _READOUT_PAIR.values()}
        if readout.get(index) is None:
            unread.add(("readout_length", (index,)))
    for i in cal.instructions:
        if not i.operational or _is_sentinel(i.error, len(i.qubits)):
            unread.add((f"{i.name}_gate_length", i.qubits))
    return unread


def _revision(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for i, part in enumerate(parts[:-2]):
        if part == "snapshots" and _REVISION.fullmatch(parts[i + 1]):
            return parts[i + 1], "/".join(parts[i + 2 :])
    return None
