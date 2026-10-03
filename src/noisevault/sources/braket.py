"""Amazon Braket standardized device properties that a user saved from their own account.

Save them with ``AwsDevice(arn).properties.json()``, or save only its ``standardized`` part.
The whole capabilities document also has the qubit count, native gates and connectivity.
Standardized v1 and v2 give per-qubit T1, T2 and fidelities and per-pair gate fidelities. v3
gives device-level values. Braket's terms restrict redistribution, so profiles stay local.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from .. import __version__, gates, units
from ..errors import SourceDataError
from ..profile import Profile
from . import Origin, json_path, source_json

_STANDARDIZED = "braket.device_schema.standardized_gate_model_qpu_device_properties"
_TECHNOLOGY = {
    "iqm": "superconducting",
    "rigetti": "superconducting",
    "oqc": "superconducting",
    "ionq": "trapped_ion",
    "aqt": "trapped_ion",
}
# Braket gate names, lowercased, to the canonical gate with the same matrix (up to global
# phase). Other names stay as they are. GPI and GPI2 are r at theta pi and pi/2.
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
# fidelityType.name -> (method, measured). For a locus with several types, the first one wins.
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
    "Braket gives a fidelity. NoiseVault reads 1 - fidelity as the average gate infidelity, as the"
    " Braket SDK's local emulator does"
)
_SAVE = "save AwsDevice(arn).properties.json() and pass that file"
_QUBIT_ID = "0|[1-9][0-9]*"
_QubitId = Annotated[str, StringConstraints(pattern=f"^(?:{_QUBIT_ID})$")]


class _Shape(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)


class _Header(_Shape):
    name: str | None = None
    version: str | None = None


class _Time(_Shape):
    value: float
    unit: Literal["ns", "us", "ms", "s"]


class _FidelityType(_Shape):
    name: str


class _Fidelity(_Shape):
    fidelity: float
    standardError: float | None = None
    fidelityType: _FidelityType | None = None


class _TypedFidelity(_Fidelity):
    fidelityType: _FidelityType


class _Direction(_Shape):
    control: int
    target: int


class _GateFidelity(_TypedFidelity):
    gateName: str
    direction: _Direction | None = None


class _Qubit(_Shape):
    T1: _Time | None = None
    T2: _Time | None = None
    oneQubitFidelity: list[_TypedFidelity] | None = None


class _Pair(_Shape):
    twoQubitGateFidelity: list[_GateFidelity] | None = None


class _Standardized(_Shape):
    braketSchemaHeader: _Header


class _PerElement(_Standardized):
    oneQubitProperties: dict[_QubitId, _Qubit] | None = None
    twoQubitProperties: dict[str, _Pair] | None = None


class _DeviceLevelQubit(_Shape):
    oneQubitFidelity: list[_Fidelity] | None = None


class _DeviceLevel(_Standardized):
    oneQubitProperties: dict[_QubitId, _DeviceLevelQubit] | None = None
    T1: _Time | None = None
    T2: _Time | None = None
    readoutFidelity: list[_Fidelity] | None = None
    readoutDuration: _Time | None = None
    singleQubitFidelity: list[_Fidelity] | None = None
    singleQubitGateDuration: _Time | None = None
    twoQubitGateFidelity: list[_Fidelity] | None = None
    twoQubitGateDuration: _Time | None = None


class _Connectivity(_Shape):
    fullyConnected: bool | None = None
    connectivityGraph: dict[_QubitId, list[_QubitId]] | None = None


class _Paradigm(_Shape):
    qubitCount: int | None = None
    nativeGateSet: list[str] | None = None
    connectivity: _Connectivity | None = None


class _Capabilities(_Shape):
    braketSchemaHeader: _Header | None = None
    service: dict[str, Any] | None = None
    paradigm: _Paradigm | None = None


def bundled_profiles() -> list[Profile]:
    """None, because Braket's terms restrict redistribution. Import your own saved file."""
    return []


def from_braket(
    path_or_dict: str | Path | Mapping[str, Any], *, device: str | None = None
) -> Profile:
    """A profile from saved Braket device properties (a JSON file path or the parsed dict).

    ``device`` names the profile. The default name is the file name without its suffix.
    """
    data: Any
    if isinstance(path_or_dict, Mapping):
        data = dict(path_or_dict)
        source = "the dict passed in"
        try:
            raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        except TypeError:
            problem = _key_problem(data)
            if problem is None:
                raise
            raise _origin(source).refuse(problem) from None
        name, hashed = device or "braket_device", "the canonical JSON of the dict passed in"
    else:
        path = Path(path_or_dict)
        raw = path.read_bytes()
        source = path.name
        data = source_json(raw, source, _SAVE)
        name, hashed = device or path.stem, f"the bytes of {path.name}"
    whole = isinstance(data, Mapping) and "standardized" in data
    caps = data if whole else {"standardized": data}
    std = caps["standardized"] or {}
    header = std.get("braketSchemaHeader") if isinstance(std, Mapping) else None
    if not isinstance(header, Mapping) or header.get("name") != _STANDARDIZED:
        raise SourceDataError(
            f"{source} is not Braket standardized gate-model properties", hint=_SAVE
        )
    version = str(header.get("version"))
    if version not in ("1", "2", "3"):
        raise SourceDataError(
            f"{source}: Braket standardized properties version {version} is not supported"
        )
    origin = _origin(source)
    shape, build = (_DeviceLevel, _device_level) if version == "3" else (_PerElement, _per_element)
    prefix = ("standardized",) if whole else ()
    for checked, at, model in ((std, prefix, shape), (caps, (), _Capabilities)):
        try:
            model.model_validate(checked)
        except ValidationError as exc:
            raise _shape_error(exc, at, source, origin) from None
    paradigm = caps.get("paradigm") or {}
    problem = _pair_problem(std, prefix)
    if problem:
        raise origin.refuse(problem)
    vendor = _vendor(caps)
    notes = [
        f"source_hash is over {hashed}",
        "Braket does not say if Z rotations are virtual, so NoiseVault takes them as virtual (rz)",
        "readout is 1 - the READOUT fidelity, the same for both prepared states",
    ]
    physics = build(std, paradigm, notes)
    calibrated_at = _calibrated_at(caps, std, notes, origin)
    return origin.profile(
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
    caps: Mapping[str, Any], std: Mapping[str, Any], notes: list[str], origin: Origin
) -> datetime | None:
    """The characterization time (standardized v3) or else when Braket refreshed the service.

    Braket's schema accepts a timestamp without a time zone, and pydantic writes the timestamp
    back without one. NoiseVault reads such a value as UTC.
    """
    std_at, service_at = std.get("updatedAt"), (caps.get("service") or {}).get("updatedAt")
    where, value = ("standardized", std_at) if std_at is not None else ("service", service_at)
    if value is None:
        return None
    try:
        when = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise origin.refuse(f"the {where} updatedAt is {value!r}, not an ISO 8601 time") from None
    notes.append(f"calibrated_at is the {where} updatedAt, when Braket last refreshed it")
    if when.tzinfo is None:
        notes.append(f"the {where} updatedAt had no time zone, so NoiseVault reads it as UTC")
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
        "Braket gives the one-qubit fidelity per qubit, not per gate, so NoiseVault applies it to"
        f" every one-qubit native gate ({', '.join(one_natives)})",
        "T2 is Braket's T2, and Braket does not say if T2 is an echo or a Ramsey value",
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
        raise SourceDataError(
            "Braket v3 standardized properties hold device-level values only, so the qubit count"
            " and native gates are missing",
            hint="pass the whole AwsDevice(arn).properties.json(), not only its standardized part",
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
    notes.append("v3 values are device-wide, so every qubit and pair gets the same values")
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

    Each id is its index, so a Braket circuit's qubit numbers are the profile's physical qubits.
    The caller disables indices with no id (IQM counts from 1).
    """
    connectivity = paradigm.get("connectivity") or {}
    graph: Mapping[str, list[str]] | None = connectivity.get("connectivityGraph")
    graph_edges = [(a, b) for a, targets in (graph or {}).items() for b in targets]
    edges_by_id = pairs if graph is None else graph_edges
    all_to_all = connectivity.get("fullyConnected") or (graph is None and not pairs)
    labels = labels | set(graph or {}) | {q for edge in edges_by_id for q in edge}
    index = {label: int(label) for label in labels}
    num_qubits = max([paradigm.get("qubitCount") or 0, *(i + 1 for i in index.values())])
    if all_to_all:
        return index, num_qubits, "all_to_all"
    if not edges_by_id:
        notes.append(
            "no qubit pair is connected, because Braket's connectivity graph has no edges and is"
            " not fully connected"
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
            "NoiseVault left out the native gates that are not a known one- or two-qubit gate:"
            f" {sorted(left_out)}"
        )
    if not natives[1]:
        natives[1] = ["r"]
        notes.append(
            "the device lists no one-qubit native gate, so the one-qubit fidelity goes to r"
        )
    defs: dict[str, dict[str, Any]] = {"rz": {"virtual": True}}
    defs |= {g: {"assumption": _ASSUMPTION} for g in natives[1]}
    defs |= {g: {"qubits": 2, "assumption": _ASSUMPTION} for g in natives[2]}
    return natives[1], defs


def _preferred(
    entries: list[Mapping[str, Any]], arity: int, skipped: set[str]
) -> tuple[Mapping[str, Any], str] | None:
    """The entry of the most preferred known fidelity type, with that type.

    Readout entries are not gate fidelities. A v3 entry with no type counts as RB.
    """
    ranked = []
    order = list(_RB_TYPES[arity])
    for entry in entries:
        fidelity_type = entry.get("fidelityType")
        kind = fidelity_type["name"] if fidelity_type else "RANDOMIZED_BENCHMARKING"
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

    Braket reads an entry without a direction as bidirectional. A symmetric gate needs only one
    record for such an entry.
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
        notes.append(
            f"NoiseVault skipped the fidelity types with no known meaning: {sorted(skipped)}"
        )


def _error(fidelity: float) -> float:
    """1 - fidelity without the binary round-off of decimal input (1 - 0.9991 -> 0.0009)."""
    return float(f"{1 - fidelity:.12g}")


def _canonical(name: str) -> str:
    return _GATES.get(name.lower(), name.lower())


def _us(time: Mapping[str, Any]) -> float:
    return units.convert(float(time["value"]), time["unit"], "us")


def _origin(source: str) -> Origin:
    return Origin(source, hint=f"correct that value in {source}")


def _key_problem(node: Any, path: tuple[str | int, ...] = ()) -> str | None:
    """Where ``node`` holds a key that is not a string, or None."""
    if isinstance(node, Mapping):
        for key in node:
            if not isinstance(key, str):
                where = f"a key of {json_path(path)}" if path else "a top-level key"
                return f"{where} is {key!r}, not a string"
        children = node.items()
    elif isinstance(node, list):
        children = enumerate(node)
    else:
        return None
    found = (_key_problem(value, (*path, key)) for key, value in children)
    return next(filter(None, found), None)


def _pair_problem(std: Mapping[str, Any], prefix: tuple[str, ...]) -> str | None:
    """Why a two-qubit key or direction does not name one pair of qubits, or None."""
    for key, props in (std.get("twoQubitProperties") or {}).items():
        pair = key.split("-")
        if len(pair) != 2 or not all(re.fullmatch(_QUBIT_ID, q) for q in pair):
            where = json_path([*prefix, "twoQubitProperties"])
            return f"{where} has the key {key!r}, not a pair of qubit ids such as '0-1'"
        for i, entry in enumerate(props.get("twoQubitGateFidelity") or []):
            direction = entry.get("direction")
            if direction and {str(direction["control"]), str(direction["target"])} != set(pair):
                where = json_path([*prefix, "twoQubitProperties", key, "twoQubitGateFidelity", i])
                return (
                    f"{where}.direction goes from qubit {direction['control']} to qubit"
                    f" {direction['target']}, not between the qubits of {key!r}"
                )
    return None


_EXPECTED = {
    "bool_type": "true or false",
    "finite_number": "a finite number",
    "float_type": "a number",
    "int_type": "an integer",
    "string_pattern_mismatch": "a qubit id such as '0'",
    "string_type": "a string",
}


def _shape_error(
    exc: ValidationError, prefix: tuple[str, ...], source: str, origin: Origin
) -> SourceDataError:
    error = exc.errors()[0]
    *parents, last = (*prefix, *error["loc"])
    if error["type"] in _EXPECTED:
        where = json_path([*parents, last])
        if last == "[key]":
            where = f"a key of {json_path(parents[:-1])}"
        return origin.refuse(f"{where} is {error['input']!r}, not {_EXPECTED[error['type']]}")
    if error["type"] == "missing":
        problem = f"{json_path(parents)} has no {last!r}"
    elif error["type"] == "literal_error":
        where, expected = json_path([*parents, last]), error["ctx"]["expected"]
        problem = f"{where} is {error['input']!r}, not {expected}"
    else:
        kind = "array" if error["type"] == "list_type" else "object"
        problem = f"{json_path([*parents, last])} is not a JSON {kind}"
    return SourceDataError(f"{source}: {problem}", hint=_SAVE)


def _vendor(caps: Mapping[str, Any]) -> str | None:
    """``iqm`` from a header name such as braket.device_schema.iqm.iqm_device_capabilities."""
    parts = ((caps.get("braketSchemaHeader") or {}).get("name") or "").split(".")
    return parts[2] if len(parts) > 3 and parts[:2] == ["braket", "device_schema"] else None
