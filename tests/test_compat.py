from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import MANILA_V01, V01, migrated, require
from typer.testing import CliRunner

from noisevault.channels import gate_channels, readout_matrix, superoperator
from noisevault.cli import app
from noisevault.errors import MigrationWarning, NoiseVaultError
from noisevault.profile import Profile, load_file
from noisevault.table import Unavailable

BASELINE = np.load(V01 / "baseline_superops.npz")
META = json.loads(str(BASELINE["_meta"]))
KEYS = sorted(k for k in BASELINE.files if k != "_meta")
PROFILES = {path.name: migrated(path) for path in sorted(V01.glob("*.json"))}


def test_baseline_covers_every_snapshot() -> None:
    assert {key.split("|")[0] for key in KEYS} == set(PROFILES)
    assert sum("|readout|" not in key for key in KEYS) == 99


@pytest.mark.parametrize("key", [k for k in KEYS if "|readout|" not in k])
def test_gate_superoperators_match_0_1(key: str) -> None:
    file, op, qubits_text = key.split("|")
    qubits = tuple(int(q) for q in qubits_text.split(","))
    table = PROFILES[file].table
    # 0.1 borrowed calibrations through aliases (h -> sx); 1.0 names the calibration it uses.
    found = table.gate(META[key]["calibration"], qubits)
    built = gate_channels(found, [table.qubit(q) for q in qubits])
    assert np.abs(superoperator(built.channels, qubits) - BASELINE[key]).max() <= 1e-12


@pytest.mark.parametrize("key", [k for k in KEYS if "|readout|" in k])
def test_readout_matrices_match_0_1(key: str) -> None:
    file, _, qubit = key.split("|")
    assert np.array_equal(readout_matrix(PROFILES[file].table.qubit(int(qubit))), BASELINE[key])


def test_aliases_are_gone() -> None:
    manila = PROFILES[MANILA_V01.name]
    assert isinstance(manila.table.gate("h", (0,)), Unavailable)


def test_loading_0_1_warns() -> None:
    with pytest.warns(MigrationWarning, match="0.1"):
        load_file(MANILA_V01)


def test_migrated_manila_fields() -> None:
    manila = PROFILES[MANILA_V01.name]
    assert manila.id == "ibm_manila"
    assert manila.device.calibrated_at.isoformat() == "2024-05-27T18:27:23+00:00"
    assert manila.gates["rz"].virtual
    assert manila.table.gate("reset", (0,)).state == "uncalibrated"
    assert manila.idle.t2_kind == "echo"
    assert manila.provenance.redistributable == "yes"
    assert manila.provenance.license == "Apache-2.0"
    assert manila.provenance.extra["migrated_from"] == "0.1"
    assert manila.connectivity.directed


def _v01(**changes) -> dict:
    data = json.loads(MANILA_V01.read_text())
    data.update(changes)
    return data


def _upgrade(data: dict) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MigrationWarning)
        return Profile.from_dict(data)


def test_ibm_sentinel_error_disables_the_gate() -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "cx" and gate["qubits"] == [0, 1]:
            gate["error"] = 1.0
        if gate["name"] == "sx" and gate["qubits"] == [3]:
            gate["operational"] = False
    table = _upgrade(data).table
    assert table.gate("cx", (0, 1)).state == "disabled"
    assert table.gate("sx", (3,)).state == "disabled"
    assert table.gate("cx", (1, 0)).state == "calibrated"


def test_rz_stays_calibrated_when_it_has_an_error() -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz" and gate["qubits"] == [2]:
            gate["error"] = 1e-4
    profile = _upgrade(data)
    assert not profile.gates["rz"].virtual
    assert profile.table.gate("rz", (2,)).avg_infidelity == 1e-4


@pytest.mark.parametrize("dead", [{"operational": False}, {"error": 1.0}])
def test_disabled_rz_stays_disabled_when_rz_becomes_virtual(dead: dict) -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz" and gate["qubits"] == [0]:
            gate.update(dead)
    profile = _upgrade(data)
    assert profile.gates["rz"].virtual
    assert profile.table.gate("rz", (0,)).state == "disabled"
    assert profile.table.gate("rz", (1,)).state == "ideal"


def test_missing_values_stay_missing() -> None:
    data = _v01()
    data["qubits"][1].update(t2_us=None, prob_meas0_prep1=None, readout_error=0.03)
    data["qubits"][2].update(prob_meas0_prep1=None, prob_meas1_prep0=None, readout_error=None)
    table = _upgrade(data).table
    assert table.qubit(1).t2_ns is None
    assert table.qubit(1).readout == (0.03, 0.03)
    assert table.qubit(2).readout is None


def _first_gate(**changes) -> dict:
    data = _v01()
    data["gates"][0].update(changes)
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"schema_version": "0.1"}, "provider is missing"),
        (_v01(gates=None), "gates should be a list, not null"),
        (_v01(provenance=[]), "provenance should be an object, not a list"),
        (_v01(qubits=[{"t1_us": 1.0}]), "qubits[0] needs an integer index"),
        (_v01(gates=[{"name": "sx"}]), "gates[0] needs a name and a list of qubits"),
        (_first_gate(error="high"), "gates[0]: error should be a number"),
        (_v01(provenance={"raw_hash": ["x"]}), "provenance.raw_hash should be a string"),
    ],
    ids=[
        "empty",
        "null-gates",
        "list-provenance",
        "qubit-index",
        "gate-qubits",
        "gate-error",
        "raw-hash",
    ],
)
def test_a_malformed_0_1_file_is_refused_with_the_field_to_fix(data: dict, message: str) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"not a valid NoiseVault 0.1 file: {message}")
    ) as info:
        _upgrade(data)
    assert isinstance(info.value, NoiseVaultError)
    assert info.value.hint == "fix that field or pull the device again"


def test_nv_show_prints_the_fix_for_a_malformed_0_1_file_on_a_hint_line(tmp_path: Path) -> None:
    data = _v01()
    del data["coupling_map"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    result = CliRunner().invoke(app, ["show", str(path)])
    assert result.stderr.splitlines() == [
        f"error: {path}: not a valid NoiseVault 0.1 file: coupling_map is missing",
        "hint: fix that field or pull the device again",
    ]
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "missing", [{"error": None, "duration_ns": None}, {"error": None}], ids=["both", "error"]
)
def test_rz_with_missing_calibration_stays_uncalibrated(missing: dict) -> None:
    data = _v01()
    for gate in data["gates"]:
        if gate["name"] == "rz":
            gate.update(missing)
    table = _upgrade(data).table
    assert not table.profile.gates["rz"].virtual
    assert table.gate("rz", (0,)).state == "uncalibrated"


def test_rz_without_records_stays_uncalibrated() -> None:
    data = _v01()
    data["gates"] = [g for g in data["gates"] if g["name"] != "rz"]
    profile = _upgrade(data)
    assert "rz" in data["basis_gates"] and not profile.gates["rz"].virtual
    assert profile.table.gate("rz", (0,)).state == "uncalibrated"


@pytest.mark.parametrize("raw_hash", ["sha256:" + "0" * 64, None], ids=["other-data", "no-hash"])
def test_an_unverified_fake_snapshot_claims_no_license(raw_hash: str | None) -> None:
    data = _v01()
    data["provenance"]["raw_hash"] = raw_hash
    prov = _upgrade(data).provenance
    assert (prov.license, prov.attribution, prov.redistributable) == (
        None,
        "IBM Quantum",
        "unknown",
    )
    assert prov.source_kind != "package_snapshot"
    assert any("did not prove" in note for note in prov.notes)


def test_the_verified_hash_is_the_0_1_payload_of_the_shipped_fake_manila() -> None:
    runtime = require("qiskit_ibm_runtime")
    if runtime.__version__ != "0.49.0":
        pytest.skip(f"the hash was taken from qiskit-ibm-runtime 0.49.0, not {runtime.__version__}")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    backend = FakeManilaV2()
    raw = {
        "configuration": backend.configuration().to_dict(),
        "properties": backend.properties().to_dict(),
    }
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    payload_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert payload_hash == json.loads(MANILA_V01.read_text())["provenance"]["raw_hash"]
    assert PROFILES[MANILA_V01.name].provenance.redistributable == "yes"
