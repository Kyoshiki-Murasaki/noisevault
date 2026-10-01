"""Reader for NoiseVault 0.1 snapshots: a pure dict-to-dict upgrade to format 1.0.

0.1 files were IBM-shaped: one record per (gate, qubits) with an average gate error, per-qubit
T1/T2 and readout. The upgrade keeps every number and turns conventions into explicit fields:
``error >= 1`` (IBM's dead-gate sentinel) or ``operational: false`` become ``disabled``, an
``rz`` with usable records whose errors and durations are all explicitly zero becomes
``virtual`` (its disabled records stay), a missing value stays missing (so an ``rz`` without
that evidence stays a calibrated or uncalibrated native), and the per-qubit readout pair wins
over the averaged ``readout_error``. A ``qiskit_fake`` file keeps the package's Apache-2.0
license only when its ``raw_hash`` is a verified package snapshot.
"""

from __future__ import annotations

from typing import Any

from . import gates
from .errors import NoiseVaultError

_IBM_PROVIDERS = {"ibm", "qiskit_fake"}
_KINDS = {  # 0.1 provider -> (data_kind, source_kind)
    "qiskit_fake": ("measured", "package_snapshot"),
    "ibm": ("measured", "account_api"),
    "demo": ("hypothetical", "hand_written"),
}
# 0.1 raw_hash values (canonical JSON of configuration + properties) checked against the snapshot
# qiskit-ibm-runtime ships (FakeManilaV2 in 0.49.0); any other qiskit_fake file may hold
# refreshed IBM service data.
_PACKAGE_SNAPSHOTS = {"sha256:f79216df928d9dba24b98d2f65f3bbf4b918c46e61e7a6455f0c8218ae4179a0"}
_UNVERIFIED_FAKE = (
    "The 0.1 file did not prove its data was the snapshot qiskit-ibm-runtime ships (a refreshed"
    " fake holds IBM Quantum service data), so no license is claimed."
)


def is_v01(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema_version") == "0.1"


# Fields the upgrade reads, and the JSON type each must have; None means any type.
_FIELDS: dict[str, type | None] = {
    "provider": str,
    "backend_name": str,
    "num_qubits": None,
    "captured_at": None,
    "basis_gates": list,
    "coupling_map": list,
    "gates": list,
    "qubits": list,
    "provenance": dict,
}
_JSON_TYPES = {dict: "an object", list: "a list", str: "a string"}


def upgrade_v01(old: dict[str, Any]) -> dict[str, Any]:
    _check_shape(old)
    provider = old["provider"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for gate in old["gates"]:
        by_name.setdefault(gate["name"], []).append(gate)

    definitions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for name in dict.fromkeys([*old["basis_gates"], *by_name]):
        entries = by_name.get(name, [])
        converted = [_record(e) for e in entries]
        live = [e for e, r in zip(entries, converted, strict=True) if not r.get("disabled")]
        if name == "rz" and live and all(_explicitly_free(e) for e in live):
            definitions[name] = {"virtual": True}
            records += [
                {"gate": name, "qubits": r["qubits"], "disabled": True}
                for r in converted
                if r.get("disabled")
            ]
            continue
        arity = len(entries[0]["qubits"]) if entries else None
        definitions[name] = {} if gates.lookup(name) else {"qubits": arity or 1}
        if provider in _IBM_PROVIDERS and any(e.get("error") is not None for e in entries):
            definitions[name]["method"] = "rb"
        records += converted

    data_kind, source_kind = _KINDS.get(provider, ("unknown", "other"))
    prov = old["provenance"]
    licensing = _licensing(provider, prov.get("raw_hash"))
    notes = [*prov.get("notes", []), "Upgraded from NoiseVault format 0.1."]
    if provider == "qiskit_fake" and licensing["redistributable"] != "yes":
        source_kind = "other"
        notes.append(_UNVERIFIED_FAKE)
    extra = {"migrated_from": "0.1", "v01_captured_at": old["captured_at"]}
    if prov.get("harvester_version"):
        extra["v01_harvester_version"] = prov["harvester_version"]
    extensions = dict(old.get("extensions") or {})
    frequencies = {
        str(q["index"]): q["frequency_ghz"] for q in old["qubits"] if q.get("frequency_ghz")
    }
    if frequencies:
        extensions["v01_qubit_frequency_ghz"] = frequencies
    metadata = {str(q["index"]): q["metadata"] for q in old["qubits"] if q.get("metadata")}
    if metadata:
        extensions["v01_qubit_metadata"] = metadata

    return {
        "noisevault": "1.0",
        "device": {
            "name": _device_name(provider, old["backend_name"]),
            "vendor": "ibm" if provider in _IBM_PROVIDERS else provider,
            "technology": "superconducting",
            "num_qubits": old["num_qubits"],
            "calibrated_at": prov.get("source_timestamp") or old["captured_at"],
        },
        "connectivity": {"edges": [list(e) for e in old["coupling_map"]], "directed": True},
        "gates": definitions,
        "idle": {"t2_kind": "echo"} if provider in _IBM_PROVIDERS else None,
        "qubits": [_qubit(q) for q in old["qubits"]],
        "calibrations": records,
        "provenance": {
            "data_kind": data_kind,
            "source_kind": source_kind,
            "source": prov.get("source"),
            "source_url": prov.get("source_url"),
            "source_hash": prov.get("raw_hash"),
            "tool": f"noisevault {prov.get('harvester_version', '0.1')}",
            "notes": notes,
            "extra": extra,
            **licensing,
        },
        "extensions": extensions,
    }


class _InvalidV01(NoiseVaultError, ValueError):
    """A 0.1 file the upgrade cannot read."""


def _check_shape(old: dict[str, Any]) -> None:
    """Raise ValueError naming the first field the upgrade cannot read."""

    def fail(problem: str) -> None:
        raise _InvalidV01(
            f"not a valid NoiseVault 0.1 file: {problem}",
            hint="fix that field or pull the device again",
        )

    for key, kind in _FIELDS.items():
        if key not in old:
            fail(f"{key} is missing")
        if kind is not None and not isinstance(old[key], kind):
            fail(f"{key} should be {_JSON_TYPES[kind]}, not {_json_type(old[key])}")
    for i, gate in enumerate(old["gates"]):
        if not (
            isinstance(gate, dict)
            and isinstance(gate.get("name"), str)
            and isinstance(gate.get("qubits"), list)
        ):
            fail(f"gates[{i}] needs a name and a list of qubits")
        if not isinstance(gate.get("error"), int | float | None):
            fail(f"gates[{i}]: error should be a number, not {_json_type(gate['error'])}")
    for i, qubit in enumerate(old["qubits"]):
        if not (isinstance(qubit, dict) and isinstance(qubit.get("index"), int)):
            fail(f"qubits[{i}] needs an integer index")
    for i, edge in enumerate(old["coupling_map"]):
        if not isinstance(edge, list):
            fail(f"coupling_map[{i}] should be a list of two qubits, not {_json_type(edge)}")
    if not isinstance(old["provenance"].get("notes", []), list):
        fail("provenance.notes should be a list")
    if not isinstance(old["provenance"].get("raw_hash"), str | None):
        fail("provenance.raw_hash should be a string")
    if not isinstance(old.get("extensions"), dict | None):
        fail("extensions should be an object")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, int | float):
        return "a number"
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _device_name(provider: str, backend_name: str) -> str:
    if provider == "qiskit_fake" and backend_name.startswith("fake_"):
        return "ibm_" + backend_name.removeprefix("fake_")
    return backend_name


def _licensing(provider: str, raw_hash: Any) -> dict[str, Any]:
    if provider == "qiskit_fake" and raw_hash in _PACKAGE_SNAPSHOTS:
        return {
            "license": "Apache-2.0",
            "attribution": "IBM Quantum (via qiskit-ibm-runtime)",
            "redistributable": "yes",
        }
    if provider == "demo":
        return {"redistributable": "yes"}
    if provider in _IBM_PROVIDERS:
        return {"attribution": "IBM Quantum", "redistributable": "unknown"}
    return {"redistributable": "unknown"}


def _explicitly_free(entry: dict[str, Any]) -> bool:
    return entry.get("error") == 0 and entry.get("duration_ns") == 0


def _record(entry: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"gate": entry["name"], "qubits": entry["qubits"]}
    error = entry.get("error")
    if not entry.get("operational", True) or (error is not None and error >= 1):
        record["disabled"] = True
    elif error is not None:
        record["avg_infidelity"] = error
    if entry.get("duration_ns") is not None:
        record["duration_ns"] = entry["duration_ns"]
    return record


def _qubit(q: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"index": q["index"]}
    for key in ("t1_us", "t2_us"):
        if q.get(key) is not None:
            out[key] = q[key]
    a, b = q.get("prob_meas1_prep0"), q.get("prob_meas0_prep1")
    if a is not None and b is not None:
        out["readout"] = {"p1_given_0": a, "p0_given_1": b}
    elif q.get("readout_error") is not None:
        out["readout"] = {"error": q["readout_error"]}
    if not q.get("operational", True):
        out["disabled"] = True
    return out
