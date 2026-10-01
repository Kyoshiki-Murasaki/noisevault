from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from conftest import MANILA_V01, V01, migrated

from noisevault.channels import gate_channels, readout_matrix, superoperator
from noisevault.errors import MigrationWarning
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
