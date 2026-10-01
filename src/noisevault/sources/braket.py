"""Amazon Braket standardized device properties that a user saved from their own account.

Save them with ``AwsDevice(arn).properties.json()`` (the whole capabilities document, which
also carries the qubit count, native gates and connectivity) or just its ``standardized`` part.
Standardized v1 and v2 give per-qubit T1, T2 and fidelities and per-pair gate fidelities; v3
gives device-level values. Braket's terms restrict redistribution, so profiles stay local.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__, gates, units
from ..profile import Profile

_STANDARDIZED = "braket.device_schema.standardized_gate_model_qpu_device_properties"
_TECHNOLOGY = {
    "iqm": "superconducting",
    "rigetti": "superconducting",
    "oqc": "superconducting",
    "ionq": "trapped_ion",
    "aqt": "trapped_ion",
}
# Braket gate names, lowercased, to the canonical gate with the same matrix (up to global
# phase); other names are kept as they are. GPI and GPI2 are r at theta pi and pi/2.
_GATES = {
    "ccnot": "ccx",
    "cnot": "cx",
    "gpi": "r",
    "gpi2": "r",
    "i": "id",
    "phaseshift": "p",
    "prx": "r",
    "si": "sdg",
    "ti": "tdg",
    "v": "sx",
    "vi": "sxdg",
    "xx": "rxx",
    "yy": "ryy",
    "zz": "rzz",
}
# fidelityType.name -> (method, measured); for a locus with several, the first listed wins
_RB_TYPES: Mapping[int, Mapping[str, tuple[str, str | None]]] = {
    1: {
        "RANDOMIZED_BENCHMARKING": ("rb", None),
        "SIMULTANEOUS_RANDOMIZED_BENCHMARKING": ("srb", "simultaneous"),
    },
    2: {
        "INTERLEAVED_RANDOMIZED_BENCHMARKING": ("irb", None),
        "RANDOMIZED_BENCHMARKING": ("rb", None),
        "SIMULTANEOUS_RANDOMIZED_BENCHMARKING": ("srb", "simultaneous"),
    },
}
_READOUT = "READOUT"
_V3_KEYS = {  # arity -> (device-level fidelity list, duration)
    1: ("singleQubitFidelity", "singleQubitGateDuration"),
    2: ("twoQubitGateFidelity", "twoQubitGateDuration"),
}
_ASSUMPTION = (
    "Braket gives a fidelity; 1 - fidelity is read as the average gate infidelity, as the Braket"
    " SDK's local emulator does"
)


def bundled_profiles() -> list[Profile]:
    """None: Braket's terms restrict redistribution; import your own saved file instead."""
    return []


def from_braket(
    path_or_dict: str | Path | Mapping[str, Any], *, device: str | None = None
) -> Profile:
    """A profile from saved Braket device properties (a JSON file path or the parsed dict).

    ``device`` names the profile; by default it is the file name without its suffix.
    """
    if isinstance(path_or_dict, Mapping):
        data = dict(path_or_dict)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        name, hashed = device or "braket_device", "the canonical JSON of the dict passed in"
    else:
        path = Path(path_or_dict)
        raw = path.read_bytes()
        data = json.loads(raw)
        name, hashed = device or path.stem, f"the bytes of {path.name}"
    caps = data if "standardized" in data else {"standardized": data}
    std = caps.get("standardized") or {}
    header = std.get("braketSchemaHeader") or {}
    if header.get("name") != _STANDARDIZED:
        raise ValueError(
            "this is not Braket standardized gate-model properties; save"
            " `AwsDevice(arn).properties.json()` (or its `.standardized`) and pass that file"
        )
    version = str(header.get("version"))
    if version not in ("1", "2", "3"):
        raise ValueError(f"Braket standardized properties version {version} is not supported")
    vendor = _vendor(caps)
    paradigm = caps.get("paradigm") or {}
    notes = [
        f"source_hash is over {hashed}",
        "Z rotations are taken as virtual (rz); Braket does not say",
        "readout is 1 - the READOUT fidelity, the same for both prepared states",
    ]
    build = _device_level if version == "3" else _per_element
    physics = build(std, paradigm, notes)
    calibrated_at = _calibrated_at(caps, std, notes)
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": vendor,
                "name": name,
                "technology": _TECHNOLOGY.get(vendor or "", "other"),
                "num_qubits": physics.pop("num_qubits"),
                "calibrated_at": calibrated_at,
            },
            **physics,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "user_file",
                "source": f"Amazon Braket device properties (standardized v{version})",
                "license": "AWS Customer Agreement (not an open license)",
                "attribution": f"{vendor or 'the hardware provider'} via Amazon Braket",
                "redistributable": "no",
                "source_hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "tool": f"noisevault {__version__}",
                "notes": notes,
            },
        }
    )


def _calibrated_at(
    caps: Mapping[str, Any], std: Mapping[str, Any], notes: list[str]
) -> datetime | None:
    """The characterization time (standardized v3) or else when Braket refreshed the service.

    Braket's schema accepts a timestamp without a time zone, and pydantic writes it back so;
    such a value is read as UTC.
    """
    std_at, service_at = std.get("updatedAt"), (caps.get("service") or {}).get("updatedAt")
    where, value = ("standardized", std_at) if std_at else ("service", service_at)
    if not value:
        return None
    when = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    notes.append(f"calibrated_at is the {where} updatedAt, when Braket last refreshed it")
    if when.tzinfo is None:
        notes.append(f"the {where} updatedAt had no time zone; read as UTC")
        when = when.replace(tzinfo=UTC)
    return when


def _per_element(
    std: Mapping[str, Any], paradigm: Mapping[str, Any], notes: list[str]
) -> dict[str, Any]:
    """Standardized v1 and v2: T1, T2 and fidelities per qubit, gate fidelities per pair."""
    one = std.get("oneQubitProperties") or {}
    two = std.get("twoQubitProperties") or {}
    labels = {*one, *(q for key in two for q in key.split("-"))}
    pairs = [tuple(key.split("-")) for key in two]
    index, num_qubits, connectivity = _layout(paradigm, labels, pairs, notes)
    one_natives, defs = _definitions(paradigm, notes)
    skipped: set[str] = set()
    qubits = _disabled_unnamed(index, num_qubits, notes)
    records = []
    for label, props in one.items():
        qubit: dict[str, Any] = {"index": index[label], "label": label}
        qubit |= {f"{k.lower()}_us": _us(props[k]) for k in ("T1", "T2") if props.get(k)}
        fidelities = props.get("oneQubitFidelity") or []
        readout = [f for f in fidelities if f["fidelityType"]["name"] == _READOUT]
        if readout:
            qubit["readout"] = {"error": _error(readout[0]["fidelity"])}
        qubits.append(qubit)
        best = _preferred(fidelities, 1, skipped)
        records += [_record(g, [index[label]], best, 1) for g in one_natives if best]
    for key, props in two.items():
        by_locus: dict[tuple[str, tuple[str, str]], list[Mapping[str, Any]]] = {}
        for entry in props.get("twoQubitGateFidelity") or []:
            gate = _canonical(entry["gateName"])
            for pair in _directions(entry, key, gate):
                by_locus.setdefault((gate, pair), []).append(entry)
        for (gate, pair), entries in by_locus.items():
            best = _preferred(entries, 2, skipped)
            if best is None:
                continue
            defs.setdefault(gate, {"qubits": 2, "assumption": _ASSUMPTION})
            records.append(_record(gate, [index[q] for q in pair], best, 2))
    _note_skipped(skipped, notes)
    notes += [
        "the one-qubit fidelity is given per qubit, not per gate; it is applied to every"
        f" one-qubit native ({', '.join(one_natives)})",
        "T2 is Braket's T2; whether it is an echo or Ramsey value is not stated",
    ]
    return {
        "num_qubits": num_qubits,
        "connectivity": connectivity,
        "gates": defs,
        "qubits": qubits,
        "calibrations": records,
    }


def _device_level(
    std: Mapping[str, Any], paradigm: Mapping[str, Any], notes: list[str]
) -> dict[str, Any]:
    if not paradigm.get("qubitCount") or not paradigm.get("nativeGateSet"):
        raise ValueError(
            "Braket v3 properties hold device-level values only; save the whole"
            " `AwsDevice(arn).properties.json()` so the qubit count and native gates are known"
        )
    one = std.get("oneQubitProperties") or {}
    index, num_qubits, connectivity = _layout(paradigm, set(one), [], notes)
    one_natives, defs = _definitions(paradigm, notes)
    qubits = _disabled_unnamed(index, num_qubits, notes)
    skipped: set[str] = set()
    for spec in defs.values():
        if spec.get("virtual"):
            continue
        arity = spec.get("qubits", 1)
        fidelity_key, duration_key = _V3_KEYS[arity]
        best = _preferred(std.get(fidelity_key) or [], arity, skipped)
        if best is not None:
            spec |= _metric(best, arity)
        if std.get(duration_key):
            spec["duration_ns"] = _us(std[duration_key]) * 1000
    records = []
    for label, props in one.items():
        best = _preferred(props.get("oneQubitFidelity") or [], 1, skipped)
        records += [_record(g, [index[label]], best, 1) for g in one_natives if best]
    _note_skipped(skipped, notes)
    notes.append("v3 values are device-wide; every qubit and pair gets them")
    readout = std.get("readoutFidelity") or []
    readout_spec = None
    if readout:
        readout_spec = {"error": _error(readout[0]["fidelity"])}
        if std.get("readoutDuration"):
            readout_spec["duration_ns"] = _us(std["readoutDuration"]) * 1000
    idle = {f"{k.lower()}_us": _us(std[k]) for k in ("T1", "T2") if std.get(k)}
    return {
        "num_qubits": num_qubits,
        "connectivity": connectivity,
        "gates": defs,
        "qubits": qubits,
        "readout": readout_spec,
        "idle": idle or None,
        "calibrations": records,
    }


# shared ------------------------------------------------------------------------------------


def _layout(
    paradigm: Mapping[str, Any], labels: set[str], pairs: list[tuple[str, ...]], notes: list[str]
) -> tuple[dict[str, int], int, str | dict[str, Any]]:
    """Braket qubit ids -> profile indices, the qubit count and the connectivity.

    When every id is an integer the id is the index, so a Braket circuit's qubit numbers are the
    profile's physical qubits; indices with no id (IQM counts from 1) are disabled by the caller.
    A fully connected paradigm is all to all. Otherwise the edges are those of its connectivity
    graph, even when the graph has none. With no graph they are the calibrated ``pairs``, and
    with no pairs either the device is all to all.
    """
    connectivity = paradigm.get("connectivity") or {}
    graph: Mapping[str, list[str]] | None = connectivity.get("connectivityGraph")
    stated = [(a, b) for a, targets in (graph or {}).items() for b in targets]
    edges_by_id = pairs if graph is None else stated
    labels = labels | set(graph or {}) | {q for edge in edges_by_id for q in edge}
    count = paradigm.get("qubitCount") or 0
    if all(label.isdigit() for label in labels):
        index = {label: int(label) for label in labels}
        num_qubits = max([count, *(i + 1 for i in index.values())])
    else:
        index = {label: i for i, label in enumerate(sorted(labels))}
        num_qubits = max(count, len(index))
    if connectivity.get("fullyConnected") or (graph is None and not edges_by_id):
        return index, num_qubits, "all_to_all"
    if not edges_by_id:
        notes.append(
            "no qubit pair is connected; Braket's connectivity graph has no edges and is not"
            " fully connected"
        )
    edges = {tuple(sorted((index[a], index[b]))) for a, b in edges_by_id}
    return index, num_qubits, {"edges": sorted(edges), "directed": False}


def _disabled_unnamed(
    index: Mapping[str, int], num_qubits: int, notes: list[str]
) -> list[dict[str, Any]]:
    if not index:
        return []
    named = set(index.values())
    qubits = [{"index": i, "disabled": True} for i in range(num_qubits) if i not in named]
    if qubits:
        notes.append(f"qubits {[q['index'] for q in qubits]} have no Braket id and are disabled")
    return qubits


def _definitions(
    paradigm: Mapping[str, Any], notes: list[str]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The one-qubit natives and the gate definitions the paradigm's native gate set implies."""
    natives: dict[int, list[str]] = {1: [], 2: []}
    left_out: set[str] = set()
    for name in paradigm.get("nativeGateSet") or []:
        info = gates.lookup(_canonical(name))
        if info is None or info.arity not in natives:
            left_out.add(name.lower())
        elif info.family != "z" and info.name not in natives[info.arity]:
            natives[info.arity].append(info.name)
    if left_out:
        notes.append(
            "native gates that are not a known one- or two-qubit gate were left out:"
            f" {sorted(left_out)}"
        )
    if not natives[1]:
        natives[1] = ["r"]
        notes.append("no one-qubit native gate is listed, so the one-qubit fidelity goes to r")
    defs: dict[str, dict[str, Any]] = {"rz": {"virtual": True}}
    defs |= {g: {"assumption": _ASSUMPTION} for g in natives[1]}
    defs |= {g: {"qubits": 2, "assumption": _ASSUMPTION} for g in natives[2]}
    return natives[1], defs


def _preferred(
    entries: list[Mapping[str, Any]], arity: int, skipped: set[str]
) -> tuple[Mapping[str, Any], str] | None:
    """The entry of the most preferred known fidelity type, with that type.

    Readout entries are not gate fidelities. A v3 entry may carry no type; it is read as RB.
    """
    ranked = []
    order = list(_RB_TYPES[arity])
    for entry in entries:
        kind = (entry.get("fidelityType") or {}).get("name", "RANDOMIZED_BENCHMARKING")
        if kind == _READOUT:
            continue
        if kind not in order:
            skipped.add(kind)
            continue
        ranked.append((order.index(kind), kind, entry))
    if not ranked:
        return None
    _, kind, entry = min(ranked, key=lambda item: item[0])
    return entry, kind


def _directions(entry: Mapping[str, Any], key: str, gate: str) -> list[tuple[str, str]]:
    """The (control, target) orders a two-qubit entry calibrates.

    Braket reads an entry without a direction as bidirectional; a symmetric gate needs only one
    record for that.
    """
    direction = entry.get("direction")
    if direction:
        return [(str(direction["control"]), str(direction["target"]))]
    a, b = key.split("-")
    return [(a, b)] if gates.is_symmetric(gate) else [(a, b), (b, a)]


def _metric(best: tuple[Mapping[str, Any], str], arity: int) -> dict[str, Any]:
    entry, kind = best
    method, measured = _RB_TYPES[arity][kind]
    return {
        "avg_infidelity": _error(entry["fidelity"]),
        "stderr": entry.get("standardError"),
        "method": method,
        "measured": measured,
    }


def _record(
    gate: str, qubits: list[int], best: tuple[Mapping[str, Any], str], arity: int
) -> dict[str, Any]:
    return {"gate": gate, "qubits": qubits, **_metric(best, arity)}


def _note_skipped(skipped: set[str], notes: list[str]) -> None:
    if skipped:
        notes.append(f"fidelity types with no known meaning were skipped: {sorted(skipped)}")


def _error(fidelity: float) -> float:
    """1 - fidelity without the binary round-off of decimal input (1 - 0.9991 -> 0.0009)."""
    return float(f"{1 - fidelity:.12g}")


def _canonical(name: str) -> str:
    return _GATES.get(name.lower(), name.lower())


def _us(time: Mapping[str, Any]) -> float:
    unit = time.get("unit", "s")
    if unit not in ("ns", "us", "ms", "s"):
        raise ValueError(f"unknown Braket time unit {unit!r}; expected ns, us, ms or s")
    return units.convert(float(time["value"]), unit, "us")


def _vendor(caps: Mapping[str, Any]) -> str | None:
    """``iqm`` from a header name such as braket.device_schema.iqm.iqm_device_capabilities."""
    parts = ((caps.get("braketSchemaHeader") or {}).get("name") or "").split(".")
    return parts[2] if len(parts) > 3 and parts[:2] == ["braket", "device_schema"] else None
