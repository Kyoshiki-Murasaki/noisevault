"""Profiles from Amazon Braket standardized device properties saved by the user."""

from __future__ import annotations

import contextlib
import hashlib
import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import deeper_than_the_parser_takes, require

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


def _ionq_with_ids(ids: list[int], *, graph: bool = False) -> dict:
    data = json.loads(IONQ.read_text())
    if graph:
        edges = {str(a): [str(b)] for a, b in zip(ids, ids[1:], strict=False)}
        data["paradigm"]["connectivity"] = {"fullyConnected": False, "connectivityGraph": edges}
    else:
        rb = {"name": "RANDOMIZED_BENCHMARKING"}
        entry = {"oneQubitFidelity": [{"fidelityType": rb, "fidelity": 0.9991, "unit": "fraction"}]}
        data["standardized"]["oneQubitProperties"] = {str(i): entry for i in ids}
    return data


@pytest.mark.parametrize(
    ("ids", "graph", "num_qubits", "disabled"),
    [
        ([1, 2, 3, 4], False, 5, [0]),
        ([1, 2, 3, 4], True, 5, [0]),
        ([0, 1, 3], False, 4, [2]),
        ([0, 1, 2, 3], False, 4, []),
    ],
)
def test_v3_indices_with_no_braket_id_are_disabled(
    ids: list[int], graph: bool, num_qubits: int, disabled: list[int]
) -> None:
    profile = from_braket(_ionq_with_ids(ids, graph=graph), device="ids")
    assert profile.device.num_qubits == num_qubits
    assert [q.index for q in profile.qubits if q.disabled] == disabled
    usable = [q for q in range(num_qubits) if not profile.table.qubit(q).disabled]
    assert usable == ids
    notes = [n for n in profile.provenance.notes if "no Braket id" in n]
    assert notes == ([f"qubits {disabled} have no Braket id and are disabled"] if disabled else [])


def test_v3_from_one_never_places_a_circuit_on_index_zero() -> None:
    profile = from_braket(_ionq_with_ids([1, 2, 3, 4], graph=True), device="from_one")
    assert profile.suggest_layout(1) == {0: 1}
    with pytest.raises(nv.LayoutError, match="qubit 0, which ionq_from_one marks disabled"):
        probabilities(profile, [Op("r", (0,), (np.pi / 2, 0.0))], 1, layout=[0])


def test_v3_without_qubit_ids_numbers_the_qubits_from_zero() -> None:
    profile = from_braket(IONQ)
    assert profile.qubits == ()
    assert profile.suggest_layout(4) == {0: 0, 1: 1, 2: 2, 3: 3}


_EDGELESS = (
    "no qubit pair is connected; Braket's connectivity graph has no edges and is not"
    " fully connected"
)


def _with_graph(path: Path, graph: dict[str, list[str]]) -> dict:
    data = json.loads(path.read_text())
    data["paradigm"]["connectivity"] = {"fullyConnected": False, "connectivityGraph": graph}
    return data


@pytest.mark.parametrize(
    ("path", "graph"),
    [(IONQ, {}), (IONQ, {"0": [], "1": [], "2": [], "3": []}), (IQM, {})],
)
def test_a_graph_with_no_edges_connects_no_pair(path: Path, graph: dict[str, list[str]]) -> None:
    profile = from_braket(_with_graph(path, graph), device="edgeless")
    assert profile.to_dict()["connectivity"] == {"directed": False, "edges": []}
    assert _EDGELESS in profile.provenance.notes


def test_v3_with_no_edges_refuses_a_two_qubit_gate() -> None:
    profile = from_braket(_with_graph(IONQ, {"0": [], "1": [], "2": [], "3": []}), device="none")
    with pytest.raises(nv.MissingCalibrationError, match="connectivity does not allow it"):
        probabilities(profile, [Op("ms", (0, 1), (0.0, 0.0))], 2, layout=[0, 1])


def test_v1_with_no_edges_keeps_the_calibrated_pairs() -> None:
    profile = from_braket(_with_graph(IQM, {}), device="garnet")
    cz = profile.table.gate("cz", (1, 2))
    assert (cz.origin, cz.avg_infidelity) == ("record", 0.009)


@pytest.mark.parametrize(
    ("path", "graph", "count", "usable"),
    [
        (IONQ, {"1": ["2"], "2": [], "3": [], "4": []}, 4, [1, 2, 3, 4]),
        (IQM, {"1": ["2"], "2": ["3"], "3": ["4"], "5": []}, 5, [1, 2, 3, 4, 5]),
    ],
)
def test_graph_keys_with_no_edges_are_qubit_ids(
    path: Path, graph: dict[str, list[str]], count: int, usable: list[int]
) -> None:
    data = _with_graph(path, graph)
    data["paradigm"]["qubitCount"] = count
    profile = from_braket(data, device="keys")
    n = profile.device.num_qubits
    assert [q for q in range(n) if not profile.table.qubit(q).disabled] == usable


def test_with_no_graph_and_no_calibrated_pair_every_pair_is_allowed() -> None:
    data = json.loads(RIGETTI.read_text())
    data["twoQubitProperties"] = {}
    assert from_braket(data, device="rig").connectivity == "all_to_all"


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
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(other)
    assert info.value.message == "other.json is not Braket standardized gate-model properties"
    assert info.value.hint == "save AwsDevice(arn).properties.json() and pass that file"
    v3_alone = json.loads(IONQ.read_text())["standardized"]
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(v3_alone)
    assert info.value.message == (
        "Braket v3 standardized properties hold device-level values only, so the qubit count"
        " and native gates are missing"
    )
    assert info.value.hint == (
        "pass the whole AwsDevice(arn).properties.json(), not only its standardized part"
    )


_SAVE = "save AwsDevice(arn).properties.json() and pass that file"


def test_values_it_cannot_read_are_named() -> None:
    data = json.loads(RIGETTI.read_text())
    newer = {**data, "braketSchemaHeader": {**data["braketSchemaHeader"], "version": "4"}}
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(newer)
    assert str(info.value) == (
        "the dict passed in: Braket standardized properties version 4 is not supported"
    )
    data["oneQubitProperties"]["0"]["T1"]["unit"] = "min"
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(data)
    assert str(info.value) == (
        "the dict passed in: oneQubitProperties['0'].T1.unit is 'min', not 'ns', 'us', 'ms'"
        f" or 's'; {_SAVE}"
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            b'{"oneQubitProperties": {',
            "is not JSON (expecting property name enclosed in double quotes at line 1, column 25)",
        ),
        (
            b'{"braketSchemaHeader": {"name": "braket',
            "is not JSON (unterminated string starting at line 1, column 33)",
        ),
        (b'{\n  "braketSchemaHeader": "\xb5"}', "is not UTF-8 text (byte 0xb5 on line 2)"),
        (b"[]", "is not Braket standardized gate-model properties"),
        (b'{"standardized": [1]}', "is not Braket standardized gate-model properties"),
    ],
    ids=["truncated", "cut-in-a-string", "latin-1", "array", "standardized-array"],
)
def test_a_file_that_is_not_braket_json_names_the_file(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    path = tmp_path / "saved.json"
    path.write_bytes(raw)
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(path)
    assert str(info.value) == f"saved.json {message}; {_SAVE}"


def test_a_file_nested_deeper_than_the_parser_takes_names_the_file(tmp_path: Path) -> None:
    nested = deeper_than_the_parser_takes()
    path = tmp_path / "saved.json"
    path.write_text(nested)
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(path)
    depth = len(nested) // 2
    assert str(info.value) == (
        f"saved.json is not JSON (nested {depth} levels deep at line 1, column {depth}); {_SAVE}"
    )


_DELETE = object()


def _parent(doc: Any, path: tuple[str | int, ...]) -> Any:
    for key in path[:-1]:
        doc = doc[key]
    return doc


@pytest.mark.parametrize(
    ("fixture", "path", "value", "shown"),
    [
        (
            IQM,
            ("standardized", "oneQubitProperties", "1", "T1", "value"),
            _DELETE,
            "standardized.oneQubitProperties['1'].T1 has no 'value'",
        ),
        (
            IQM,
            ("standardized", "oneQubitProperties", "2", "oneQubitFidelity", 1, "fidelityType"),
            _DELETE,
            "standardized.oneQubitProperties['2'].oneQubitFidelity[1] has no 'fidelityType'",
        ),
        (
            RIGETTI,
            ("twoQubitProperties", "0-1", "twoQubitGateFidelity", 0, "gateName"),
            _DELETE,
            "twoQubitProperties['0-1'].twoQubitGateFidelity[0] has no 'gateName'",
        ),
        (
            RIGETTI,
            ("twoQubitProperties", "0-1", "twoQubitGateFidelity", 0, "direction", "target"),
            _DELETE,
            "twoQubitProperties['0-1'].twoQubitGateFidelity[0].direction has no 'target'",
        ),
        (
            IONQ,
            ("standardized", "readoutFidelity", 0, "fidelity"),
            _DELETE,
            "standardized.readoutFidelity[0] has no 'fidelity'",
        ),
        (
            IONQ,
            ("standardized", "twoQubitGateDuration", "value"),
            _DELETE,
            "standardized.twoQubitGateDuration has no 'value'",
        ),
        (
            RIGETTI,
            ("oneQubitProperties", "0", "T1", "unit"),
            _DELETE,
            "oneQubitProperties['0'].T1 has no 'unit'",
        ),
        (
            IONQ,
            ("standardized", "twoQubitGateDuration", "unit"),
            _DELETE,
            "standardized.twoQubitGateDuration has no 'unit'",
        ),
        (
            RIGETTI,
            ("twoQubitProperties", "0-1", "twoQubitGateFidelity", 0, "fidelityType"),
            _DELETE,
            "twoQubitProperties['0-1'].twoQubitGateFidelity[0] has no 'fidelityType'",
        ),
        (
            IQM,
            ("standardized", "oneQubitProperties"),
            [{"T1": {"value": 4.1e-05, "unit": "s"}}],
            "standardized.oneQubitProperties is not a JSON object",
        ),
        (
            RIGETTI,
            ("oneQubitProperties", "0", "oneQubitFidelity"),
            {"fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}, "fidelity": 0.99},
            "oneQubitProperties['0'].oneQubitFidelity is not a JSON array",
        ),
    ],
    ids=[
        "T1-value",
        "fidelity-type",
        "gate-name",
        "direction-target",
        "v3-readout-fidelity",
        "v3-duration-value",
        "T1-unit",
        "v3-duration-unit",
        "two-qubit-fidelity-type",
        "object",
        "array",
    ],
)
def test_a_file_missing_a_value_the_reader_needs_names_where(
    tmp_path: Path, fixture: Path, path: tuple[str | int, ...], value: object, shown: str
) -> None:
    doc = json.loads(fixture.read_text())
    if value is _DELETE:
        del _parent(doc, path)[path[-1]]
    else:
        _parent(doc, path)[path[-1]] = value
    damaged = tmp_path / fixture.name
    damaged.write_text(json.dumps(doc))
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(damaged)
    assert str(info.value) == f"{fixture.name}: {shown}; {_SAVE}"


def _key_paths(node: Any, path: tuple[str | int, ...] = ()) -> Iterator[tuple[str | int, ...]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield (*path, key)
            yield from _key_paths(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _key_paths(value, (*path, index))


@pytest.mark.parametrize(
    ("fixture", "path"),
    [
        pytest.param(fixture, path, id=f"{fixture.stem}:{'.'.join(map(str, path))}")
        for fixture in (IQM, RIGETTI, IONQ)
        for path in _key_paths(json.loads(fixture.read_text()))
    ],
)
def test_a_file_missing_any_key_imports_or_raises_source_data_error(
    tmp_path: Path, fixture: Path, path: tuple[str | int, ...]
) -> None:
    doc = json.loads(fixture.read_text())
    del _parent(doc, path)[path[-1]]
    damaged = tmp_path / fixture.name
    damaged.write_text(json.dumps(doc))
    with contextlib.suppress(nv.SourceDataError):
        from_braket(damaged)


def test_v3_fidelity_with_no_type_is_read_as_randomized_benchmarking() -> None:
    data = json.loads(IONQ.read_text())
    del data["standardized"]["singleQubitFidelity"][0]["fidelityType"]
    data["standardized"]["twoQubitGateFidelity"][0]["fidelityType"] = None
    profile = from_braket(data, device="untyped")
    one, two = profile.gates["r"], profile.gates["ms"]
    assert (one.avg_infidelity, one.method) == (0.0004, "rb")
    assert (two.avg_infidelity, two.method) == (0.0079, "rb")


def test_public_api() -> None:
    assert nv.from_braket(IQM).fingerprint == from_braket(IQM).fingerprint
    named = nv.from_braket(json.loads(IQM.read_text()), device="garnet")
    assert named.id == "iqm_garnet"


def _rigetti_with(pair: str, entries: list[dict]) -> dict:
    data = json.loads(RIGETTI.read_text())
    data["twoQubitProperties"][pair]["twoQubitGateFidelity"] = entries
    return data


def test_each_direction_of_a_directed_gate_keeps_its_own_fidelity() -> None:
    irb = {"name": "INTERLEAVED_RANDOMIZED_BENCHMARKING"}
    data = _rigetti_with(
        "0-1",
        [
            {
                "direction": {"control": 1, "target": 0},
                "gateName": "CNOT",
                "fidelity": 0.955,
                "fidelityType": irb,
            },
            {
                "direction": {"control": 0, "target": 1},
                "gateName": "CNOT",
                "fidelity": 0.98,
                "fidelityType": irb,
            },
        ],
    )
    profile = from_braket(data, device="rig")
    cx = {r.qubits: r.avg_infidelity for r in profile.calibrations if r.gate == "cx"}
    assert cx == {(1, 0): 0.045, (0, 1): 0.02, (1, 3): 0.038}
    assert profile.table.typical(2, (0, 1)).avg_infidelity == 0.02


def test_directionless_fidelity_of_a_directed_gate_covers_both_orders() -> None:
    irb = {"name": "INTERLEAVED_RANDOMIZED_BENCHMARKING"}
    data = _rigetti_with("1-3", [{"gateName": "CNOT", "fidelity": 0.962, "fidelityType": irb}])
    profile = from_braket(data, device="rig")
    cx = {r.qubits: r.avg_infidelity for r in profile.calibrations if r.gate == "cx"}
    assert cx == {(1, 0): 0.045, (1, 3): 0.038, (3, 1): 0.038}
    assert profile.table.gate("cx", (3, 1)).origin == "record"


def _ionq_natives(natives: list[str]) -> dict:
    data = json.loads(IONQ.read_text())
    data["paradigm"]["nativeGateSet"] = natives
    return data


def test_v3_xx_native_keeps_the_two_qubit_calibration() -> None:
    profile = from_braket(_ionq_natives(["RX", "RZ", "XX"]), device="xx")
    assert sorted(profile.gates) == ["rx", "rxx", "rz"]
    xx = profile.gates["rxx"]
    assert (xx.avg_infidelity, xx.duration_ns, xx.stderr) == (0.0079, 600_000, 0.0004)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ops = [Op("rx", (0,), (np.pi / 2,)), Op("rxx", (0, 1), (np.pi / 2,))]
        assert probabilities(profile, ops, 2, layout=[0, 1]).sum() == pytest.approx(1)


@pytest.mark.parametrize(
    ("native", "gate"),
    [("XX", "rxx"), ("YY", "ryy"), ("ZZ", "rzz"), ("MS", "ms"), ("ECR", "ecr"), ("CNOT", "cx")],
)
def test_two_qubit_natives_get_their_canonical_gate(native: str, gate: str) -> None:
    profile = from_braket(_ionq_natives(["GPI", "GPI2", native]), device="two")
    assert profile.gates[gate].avg_infidelity == 0.0079


@pytest.mark.parametrize(
    ("native", "gate"), [("V", "sx"), ("VI", "sxdg"), ("I", "id"), ("PRX", "r")]
)
def test_one_qubit_natives_get_their_canonical_gate(native: str, gate: str) -> None:
    profile = from_braket(_ionq_natives([native, "MS"]), device="one")
    assert profile.gates[gate].avg_infidelity == 0.0004


def test_natives_with_no_equivalent_are_named_in_a_note() -> None:
    profile = from_braket(_ionq_natives(["GPI", "GPI2", "MS", "XY", "CPhaseShift"]), device="odd")
    assert sorted(profile.gates) == ["ms", "r", "rz"]
    left_out = "native gates that are not a known one- or two-qubit gate were left out:"
    assert f"{left_out} ['cphaseshift', 'xy']" in profile.provenance.notes


def test_a_value_a_profile_cannot_hold_names_the_file_and_the_value(tmp_path: Path) -> None:
    data = json.loads(RIGETTI.read_text())
    [readout] = [
        f
        for f in data["oneQubitProperties"]["3"]["oneQubitFidelity"]
        if f["fidelityType"]["name"] == "READOUT"
    ]
    readout["fidelity"] = 1.5
    path = tmp_path / "rigetti.json"
    path.write_text(json.dumps(data))
    with pytest.raises(nv.SourceDataError) as info:
        from_braket(path)
    assert (info.value.message, info.value.hint) == (
        "rigetti.json: readout.error of qubit 3: Input should be greater than or equal to 0,"
        " got -0.5",
        "correct that value in rigetti.json",
    )
