"""Profiles from Amazon Braket standardized device properties saved by the user."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault.reference import Op, probabilities
from noisevault.sources.braket import from_braket

FIXTURES = Path(__file__).parent / "fixtures" / "braket"
IQM = FIXTURES / "iqm_capabilities_v1.json"
RIGETTI = FIXTURES / "rigetti_standardized_v1.json"
IONQ = FIXTURES / "ionq_capabilities_v3.json"


def test_fixtures_are_valid_braket_documents() -> None:
    require("braket.device_schema")
    from braket.device_schema.ionq.ionq_device_capabilities_v1 import IonqDeviceCapabilities
    from braket.device_schema.iqm.iqm_device_capabilities_v1 import IqmDeviceCapabilities
    from braket.device_schema.standardized_gate_model_qpu_device_properties_v1 import (
        StandardizedGateModelQpuDeviceProperties,
    )

    IqmDeviceCapabilities.parse_raw(IQM.read_text())
    IonqDeviceCapabilities.parse_raw(IONQ.read_text())
    StandardizedGateModelQpuDeviceProperties.parse_raw(RIGETTI.read_text())


def test_iqm_capabilities_keep_braket_qubit_numbers() -> None:
    profile = from_braket(IQM, device="garnet")
    assert (profile.id, profile.device.technology) == ("iqm_garnet", "superconducting")
    assert profile.device.calibrated_at.isoformat() == "2026-09-15T08:30:00+00:00"
    assert profile.device.num_qubits == 5  # ids 1..4 stay 1..4; index 0 has no qubit
    assert profile.table.qubit(0).disabled
    assert [q.label for q in profile.qubits if not q.disabled] == ["1", "2", "3", "4"]
    assert profile.connectivity.edges == ((1, 2), (2, 3), (3, 4))
    first = profile.table.qubit(1)
    assert (first.t1_ns, first.t2_ns) == pytest.approx((41_000, 22_000))
    assert first.readout == pytest.approx((0.038, 0.038))


def test_fidelity_types_map_to_methods_in_preference_order() -> None:
    profile = from_braket(IQM, device="garnet")
    one = {r.qubits[0]: r for r in profile.calibrations if r.gate == "r"}
    assert (one[1].avg_infidelity, one[1].method, one[1].stderr) == (0.0009, "rb", 0.0001)
    assert (one[3].method, one[3].measured) == ("srb", "simultaneous")  # only SRB given
    cz = {r.qubits: r for r in profile.calibrations if r.gate == "cz"}
    assert (cz[(1, 2)].avg_infidelity, cz[(1, 2)].method) == (0.009, "irb")  # IRB over RB
    assert "Braket gives a fidelity" in profile.gates["cz"].assumption


def test_standardized_only_file_uses_directions_and_pair_keys() -> None:
    profile = from_braket(RIGETTI)
    cx = {r.qubits: r.avg_infidelity for r in profile.calibrations if r.gate == "cx"}
    assert cx == {(1, 0): 0.045, (1, 3): 0.038}  # control first, from direction
    assert profile.connectivity.edges == ((0, 1), (1, 3))
    assert profile.table.qubit(2).disabled
    notes = " ".join(profile.provenance.notes)
    assert "PURITY_BENCHMARKING" in notes
    assert "no one-qubit native gate is listed" in notes


def test_v3_device_level_values() -> None:
    profile = from_braket(IONQ)
    assert (profile.device.technology, profile.device.num_qubits) == ("trapped_ion", 4)
    assert profile.connectivity == "all_to_all"
    one, two = profile.gates["r"], profile.gates["ms"]
    assert (one.avg_infidelity, one.duration_ns) == (0.0004, 120_000)
    assert (two.avg_infidelity, two.duration_ns, two.stderr) == (0.0079, 600_000, 0.0004)
    assert (profile.readout.error, profile.readout.duration_ns) == (0.0058, 200_000)
    assert (profile.idle.t1_us, profile.idle.t2_us) == (1e7, 5e5)
    # the characterization time, not the later service refresh (2026-09-20)
    assert profile.device.calibrated_at.isoformat() == "2026-09-19T12:00:00+00:00"
    assert any("standardized updatedAt" in note for note in profile.provenance.notes)


def test_timestamp_without_time_zone_is_read_as_utc() -> None:
    require("braket.device_schema")
    from braket.device_schema.iqm.iqm_device_capabilities_v1 import IqmDeviceCapabilities

    data = json.loads(IQM.read_text())
    data["service"]["updatedAt"] = "2020-06-16T19:28:02.869136"
    saved = IqmDeviceCapabilities.parse_raw(json.dumps(data)).json()  # as AwsDevice saves it
    profile = from_braket(json.loads(saved), device="garnet")
    assert profile.device.calibrated_at.isoformat() == "2020-06-16T19:28:02.869136+00:00"
    assert "the service updatedAt had no time zone; read as UTC" in profile.provenance.notes


def test_provenance_hashes_what_was_read() -> None:
    profile = from_braket(IQM)
    prov = profile.provenance
    assert prov.source_hash == "sha256:" + hashlib.sha256(IQM.read_bytes()).hexdigest()
    assert (prov.source_kind, prov.redistributable) == ("user_file", "no")
    as_dict = from_braket(json.loads(IQM.read_text()), device=IQM.stem)
    assert as_dict.fingerprint == profile.fingerprint


@pytest.mark.parametrize(
    ("path", "layout", "ops"),
    [
        (IQM, [1, 2], [Op("r", (0,), (np.pi / 2, 0.0)), Op("cz", (0, 1))]),
        (RIGETTI, [1, 0], [Op("r", (0,), (np.pi / 2, 0.0)), Op("cx", (0, 1))]),
        (IONQ, [0, 1], [Op("r", (0,), (np.pi / 2, 0.0)), Op("ms", (0, 1), (0.0, 0.0))]),
    ],
)
def test_profiles_convert_on_their_natives(path: Path, layout: list[int], ops: list[Op]) -> None:
    profile = from_braket(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert probabilities(
            profile, [*ops, Op("rz", (1,), (0.2,))], 2, layout=layout
        ).sum() == pytest.approx(1)


def test_errors_say_what_to_save(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"braketSchemaHeader": {"name": "something.else", "version": "1"}}))
    with pytest.raises(ValueError, match=r"AwsDevice\(arn\).properties.json\(\)"):
        from_braket(other)
    v3_alone = json.loads(IONQ.read_text())["standardized"]
    with pytest.raises(ValueError, match="qubit count and native gates"):
        from_braket(v3_alone)


def test_public_api() -> None:
    assert nv.from_braket(IQM).fingerprint == from_braket(IQM).fingerprint
    named = nv.from_braket(json.loads(IQM.read_text()), device="garnet")
    assert named.id == "iqm_garnet"
