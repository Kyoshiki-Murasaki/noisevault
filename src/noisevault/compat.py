"""Reader for NoiseVault 0.1 snapshots: a pure dict-to-dict upgrade to format 1.0.

0.1 files were IBM-shaped: one record per (gate, qubits) with an average gate error, per-qubit
T1/T2 and readout. The upgrade keeps every number and turns conventions into explicit fields:
``error >= 1`` (IBM's dead-gate sentinel) or ``operational: false`` become ``disabled``, an
``rz`` whose usable errors and durations are all zero becomes ``virtual`` (its disabled records
stay), a missing value stays missing, and the per-qubit readout pair wins over the averaged
``readout_error``.
"""

from __future__ import annotations

from typing import Any

from . import gates

_IBM_PROVIDERS = {"ibm", "qiskit_fake"}
_KINDS = {  # 0.1 provider -> (data_kind, source_kind)
    "qiskit_fake": ("measured", "package_snapshot"),
    "ibm": ("measured", "account_api"),
    "demo": ("hypothetical", "hand_written"),
}


def is_v01(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema_version") == "0.1"


def upgrade_v01(old: dict[str, Any]) -> dict[str, Any]:
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
        if name == "rz" and all(not e.get("error") and not e.get("duration_ns") for e in live):
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
            "notes": [*prov.get("notes", []), "Upgraded from NoiseVault format 0.1."],
            "extra": extra,
            **_licensing(provider),
        },
        "extensions": extensions,
    }


def _device_name(provider: str, backend_name: str) -> str:
    if provider == "qiskit_fake" and backend_name.startswith("fake_"):
        return "ibm_" + backend_name.removeprefix("fake_")
    return backend_name


def _licensing(provider: str) -> dict[str, Any]:
    if provider == "qiskit_fake":
        return {
            "license": "Apache-2.0",
            "attribution": "IBM Quantum (via qiskit-ibm-runtime)",
            "redistributable": "yes",
        }
    if provider == "demo":
        return {"redistributable": "yes"}
    return {"redistributable": "unknown"}


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
