from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import deeper_than_the_parser_takes

import noisevault as nv
from noisevault.sources import ibm_public

FIXTURES = Path(__file__).parent / "fixtures" / "ibm"
# FakeManilaV2's packaged BackendProperties (Apache-2.0), the shape the endpoint returns
PROPERTIES = (FIXTURES / "manila_properties.json").read_bytes()
CONFIGURATION = (FIXTURES / "manila_configuration.json").read_bytes()
LISTING = json.dumps([{"name": "ibm_fez"}, {"name": "ibm_kingston"}]).encode()

network = pytest.mark.skipif(
    os.environ.get("NOISEVAULT_NETWORK") != "1", reason="set NOISEVAULT_NETWORK=1 to run"
)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the fixture for ibm_manila at any time; every other device is a 404."""
    requested: list[str] = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        if url == ibm_public.BASE_URL:
            return LISTING
        if url.startswith(f"{ibm_public.BASE_URL}/ibm_manila/properties"):
            return PROPERTIES
        if url == f"{ibm_public.BASE_URL}/ibm_manila/configuration":
            return CONFIGURATION
        raise ibm_public._NotFound(url)

    monkeypatch.setattr(ibm_public, "fetch", fetch)
    return requested


def test_properties_json_gives_the_same_physics_as_the_target(served: list[str]) -> None:
    pulled = ibm_public.pull("ibm_manila")
    assert pulled.fingerprint == nv.load("ibm_manila").fingerprint  # bundled, from the Target
    assert pulled.device.processor == "Falcon r5.11"
    assert pulled.table.qubit(0).readout == (0.0158, 0.05479999999999996)


def test_provenance_pins_the_exact_response(served: list[str]) -> None:
    prov = ibm_public.pull("ibm_manila").provenance
    assert prov.source_hash == "sha256:" + hashlib.sha256(PROPERTIES).hexdigest()
    assert (prov.data_kind, prov.source_kind, prov.redistributable) == (
        "measured",
        "public_api",
        "unknown",
    )
    assert prov.source_url == f"{ibm_public.BASE_URL}/ibm_manila/properties"
    assert prov.retrieved_at is not None


def test_at_becomes_updated_before_in_utc(served: list[str]) -> None:
    ibm_public.pull("ibm_manila", at="2025-06-01")
    assert served[0].endswith("/ibm_manila/properties?updated_before=2025-06-01T00%3A00%3A00Z")
    served.clear()
    ibm_public.pull("ibm_manila", at="2025-06-01T02:00:00+02:00")
    assert served[0].endswith("updated_before=2025-06-01T00%3A00%3A00Z")


def test_unlisted_device_points_to_the_bundled_snapshot(served: list[str]) -> None:
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_torino")
    assert info.value.message == (
        "ibm_torino is not listed on the public endpoint (the endpoint lists ibm_fez,"
        " ibm_kingston); did you mean 'ibm_kingston'? if not, the device may be retired"
    )
    assert info.value.hint == "ibm_torino is bundled, so nv.load('ibm_torino') loads it offline"


def test_unlisted_device_close_to_a_listed_one_names_it(served: list[str]) -> None:
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fezz")
    assert info.value.message == (
        "ibm_fezz is not listed on the public endpoint (the endpoint lists ibm_fez, ibm_kingston);"
        " did you mean 'ibm_fez'? if not, the device may be retired"
    )
    assert info.value.hint == "if your IBM account can see it, pull with source='ibm-account'"


def test_listed_device_with_no_history_that_far_back(served: list[str]) -> None:
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fez", at="2019-01-01")
    assert info.value.message == (
        "IBM's public endpoint has no ibm_fez calibration older than 2019-01-01"
    )
    assert info.value.hint == (
        "pick a later date, or pull through your IBM account with source='ibm-account'"
    )


def test_network_failure_is_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(ibm_public.urllib.request, "urlopen", refuse)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fez")
    assert info.value.message == "could not reach IBM's public endpoint (Name or service not known)"
    assert info.value.hint == (
        "check the network connection, or run nv list to see every profile you can load offline"
    )


def test_an_http_error_says_to_try_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(ibm_public.urllib.request, "urlopen", unavailable)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    assert info.value.message == f"IBM's public endpoint answered HTTP 503 for {url}"
    assert info.value.hint == (
        "try again later, or pull through your IBM account with source='ibm-account'"
    )


_NOT_JSON = {
    "html": b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>",
    "cut short": PROPERTIES[:300],
    "latin-1": "<html>Accès refusé</html>".encode("latin-1"),
}


@pytest.mark.parametrize("body", [*_NOT_JSON, "too deep"])
def test_a_reply_that_is_not_json_says_to_try_again(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    raw = deeper_than_the_parser_takes().encode() if body == "too deep" else _NOT_JSON[body]
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint answered {url} with something other than JSON",
        "try again later, or pull through your IBM account with source='ibm-account'",
    )


def _properties(edit: Callable[[dict[str, Any]], object]) -> bytes:
    props = json.loads(PROPERTIES)
    edit(props)
    return json.dumps(props).encode()


_WRONG_SHAPE = {
    "a list": (b"[]", ""),
    "a string": (b'"maintenance"', ""),
    "qubits as text": (_properties(lambda p: p.update(qubits="x")), " at qubits"),
    "a parameter without a name": (
        _properties(lambda p: p["qubits"][0][0].pop("name")),
        " at qubits[0][0].name",
    ),
    "an error as text": (
        _properties(lambda p: p["gates"][0]["parameters"][0].update(value="0.000155")),
        " at gates[0].parameters[0].value",
    ),
    "a unit as a number": (
        _properties(lambda p: p["qubits"][0][0].update(unit=1)),
        " at qubits[0][0].unit",
    ),
    "a gate without qubits": (
        _properties(lambda p: p["gates"][0].pop("qubits")),
        " at gates[0].qubits",
    ),
    "a gate name as a number": (
        _properties(lambda p: p["gates"][0].update(gate=5)),
        " at gates[0].gate",
    ),
    "parameters as text": (
        _properties(lambda p: p["gates"][0].update(parameters="x")),
        " at gates[0].parameters",
    ),
    "a date that is not a date": (
        _properties(lambda p: p.update(last_update_date="yesterday")),
        " at last_update_date",
    ),
}


@pytest.mark.parametrize("body", list(_WRONG_SHAPE))
def test_a_reply_of_the_wrong_shape_says_where_and_to_try_again(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    raw, where = _WRONG_SHAPE[body]
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw)
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint answered {url} with JSON of the wrong shape{where}",
        "try again later, or pull through your IBM account with source='ibm-account'",
    )


_UNREADABLE = {
    "too deep": deeper_than_the_parser_takes().encode(),
    "a string": b'"maintenance"',
    "a name as a number": b'[{"name": 5}]',
    "a processor type as text": b'{"processor_type": "Heron"}',
}


@pytest.mark.parametrize("body", list(_UNREADABLE))
def test_a_listing_or_configuration_it_cannot_read_only_loses_its_detail(
    served: list[str], monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    serve = ibm_public.fetch

    def fetch(url: str) -> bytes:
        if url == ibm_public.BASE_URL or url.endswith("/configuration"):
            return _UNREADABLE[body]
        return serve(url)

    monkeypatch.setattr(ibm_public, "fetch", fetch)
    assert ibm_public.pull("ibm_manila").device.processor is None
    with pytest.raises(nv.SourceUnavailable) as info:
        ibm_public.pull("ibm_torino")
    assert (
        info.value.message
        == "ibm_torino is not listed on the public endpoint; the device may be retired"
    )


def test_nv_pull_saves_one_file_per_calibration(served: list[str], vault: Path) -> None:
    first = nv.pull("ibm_manila")
    second = nv.pull("ibm_manila")
    assert first.fingerprint == second.fingerprint
    assert [p.name for p in vault.glob("*.json.gz")] == ["ibm_manila@2024-05-27T182723Z.json.gz"]
    [info] = [i for i in nv.profiles() if i.id == "ibm_manila"]
    assert info.location == "vault" and info.data_kind == "measured"


def test_repull_of_a_reconverted_calibration_replaces_the_older_file(
    served: list[str], vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = nv.pull("ibm_manila")
    changed = json.loads(PROPERTIES)
    changed["qubits"][0][0]["value"] = 99.0  # T1 of qubit 0, same last_update_date
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(changed).encode())
    with pytest.warns(nv.NoiseVaultWarning, match=r"replaced ibm_manila@2024-05-27T182723Z"):
        newer = nv.pull("ibm_manila")
    assert newer.fingerprint != first.fingerprint
    assert [p.name for p in vault.iterdir() if p.name.endswith(".gz")] == [
        "ibm_manila@2024-05-27T182723Z.json.gz"
    ]
    loaded = nv.load("ibm_manila")
    assert loaded.fingerprint == newer.fingerprint
    assert loaded.table.qubit(0).t1_ns == 99_000


def test_pull_of_an_id_no_source_serves_says_what_pull_takes(vault: Path) -> None:
    with pytest.raises(nv.SourceUnavailable, match="no source pulls 'quantinuum_h2-1'") as info:
        nv.pull("quantinuum_h2-1")
    assert "nv.load('quantinuum_h2-1') loads it offline" in str(info.value)
    with pytest.raises(nv.SourceUnavailable) as info:
        nv.pull("rigetti_ankaa-3")
    assert str(info.value) == (
        "no source pulls 'rigetti_ankaa-3'; nv pull takes IBM devices (ibm_fez) and IonQ devices"
        " (ionq_forte-1), and nv list shows every profile you can load offline"
    )


def test_at_accepts_a_date_and_explains_a_bad_string(served: list[str]) -> None:
    ibm_public.pull("ibm_manila", at=date(2025, 6, 1))
    assert served[0].endswith("updated_before=2025-06-01T00%3A00%3A00Z")
    with pytest.raises(ValueError, match=r"at='June 2025' is not an ISO 8601 date or time"):
        nv.pull("ibm_manila", at="June 2025")


def test_nv_pull_output_writes_there(served: list[str], tmp_path: Path, vault: Path) -> None:
    target = tmp_path / "manila.json"
    nv.pull("ibm_manila", output=target)
    assert nv.load(target).id == "ibm_manila" and not vault.exists()


@pytest.mark.network
@network
def test_live_pull_of_ibm_fez(vault: Path) -> None:
    now = nv.pull("ibm_fez")
    assert now.id == "ibm_fez" and now.device.num_qubits == 156
    assert now.provenance.source_kind == "public_api"
    past = nv.pull("ibm_fez", at="2025-06-01")
    assert past.device.calibrated_at.isoformat() < "2025-06-01"
    assert past.fingerprint != now.fingerprint
    assert len(list(vault.glob("ibm_fez@*.json.gz"))) == 2


def test_a_gate_the_response_omits_on_a_qubit_takes_the_median_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    props = json.loads(PROPERTIES)
    props["gates"] = [e for e in props["gates"] if (e["gate"], e["qubits"]) != ("sx", [0])]
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    errors = [
        p["value"]
        for e in props["gates"]
        if e["gate"] == "sx"
        for p in e["parameters"]
        if p["name"] == "gate_error"
    ]
    assert len(errors) == 4
    profile = ibm_public.pull("ibm_manila")
    sx = profile.table.gate("sx", (0,))
    assert (sx.state, sx.origin, sx.avg_infidelity) == (
        "calibrated",
        "default",
        statistics.median(errors),
    )
    assert profile.provenance.notes == (
        "Qubits [0] have no sx error; the device median applies to them.",
    )


def test_a_qubit_that_is_not_operational_stays_out_of_the_device_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    props = json.loads(PROPERTIES)
    props["gates"] = [e for e in props["gates"] if (e["gate"], e["qubits"]) != ("sx", [0])]
    stamp = props["last_update_date"]
    for qubit, prep in zip(props["qubits"], (0.01, 0.02, 0.03, 0.5, 0.04), strict=True):
        qubit.append({"name": "init_error", "value": prep, "unit": "", "date": stamp})
    props["qubits"][3].append({"name": "operational", "value": 0, "unit": "", "date": stamp})
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    qubits = [{p["name"]: p["value"] for p in qubit} for qubit in props["qubits"]]
    sx = {
        e["qubits"][0]: p["value"]
        for e in props["gates"]
        if e["gate"] == "sx"
        for p in e["parameters"]
        if p["name"] == "gate_error"
    }

    def median(key: str) -> float:
        return statistics.median(qubits[q][key] for q in (0, 1, 2, 4))

    profile = ibm_public.pull("ibm_manila")
    sx_0 = profile.table.gate("sx", (0,)).avg_infidelity
    assert sx_0 == statistics.median(sx[q] for q in (1, 2, 4))
    readout = profile.readout
    assert (readout.p1_given_0, readout.p0_given_1) == (
        median("prob_meas1_prep0"),
        median("prob_meas0_prep1"),
    )
    assert profile.prep.error == median("init_error")
    assert (profile.idle.t1_us, profile.idle.t2_us) == pytest.approx((median("T1"), median("T2")))


def test_invalid_coherence_in_the_response_takes_the_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    props = json.loads(PROPERTIES)
    t2 = [next(p for p in q if p["name"] == "T2") for q in props["qubits"]]
    t2[0]["value"] = 0
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    profile = ibm_public.pull("ibm_manila")
    others_us = statistics.median(p["value"] for p in t2[1:])
    assert profile.table.qubit(0).t2_ns == pytest.approx(others_us * 1000)
    note = (
        "Qubit 0 reported T2 = 0 us. NoiseVault treats the value as missing, so the device median"
        " applies."
    )
    assert note in profile.provenance.notes


def _disabled_with_zero(disabled: int, zeroed: set[int], label: str) -> dict:
    props = json.loads(PROPERTIES)
    stamp = props["last_update_date"]
    props["qubits"][disabled].append({"name": "operational", "value": 0, "unit": "", "date": stamp})
    for index, qubit in enumerate(props["qubits"]):
        value = next(p for p in qubit if p["name"] == label)
        assert value["value"] > 0
        if index in zeroed:
            value["value"] = 0
    return props


@pytest.mark.parametrize("label", ["T1", "T2"])
def test_valid_coherence_only_on_a_disabled_qubit_is_an_error(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    props = _disabled_with_zero(3, {0, 1, 2, 4}, label)
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    message = (
        f"ibm_manila reports no valid {label} on any working qubit"
        f" (for example, qubit 0: {label} = 0 us). {label} must be a positive number of"
        " microseconds"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ibm_public.pull("ibm_manila")


@pytest.mark.parametrize("label", ["T1", "T2"])
def test_invalid_coherence_only_on_a_disabled_qubit_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    def pull(kept: set[int]) -> nv.Profile:
        props = _disabled_with_zero(3, {3}, label)
        props["qubits"] = [
            [p for p in qubit if p["name"] != label or index in kept]
            for index, qubit in enumerate(props["qubits"])
        ]
        monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
        return ibm_public.pull("ibm_manila")

    zero, omitted = pull({3}), pull(set())
    assert getattr(zero.idle, f"{label.lower()}_us") is None
    assert (zero.fingerprint, zero.provenance.notes) == (
        omitted.fingerprint,
        omitted.provenance.notes,
    )


def test_no_valid_coherence_names_a_working_qubit(monkeypatch: pytest.MonkeyPatch) -> None:
    props = _disabled_with_zero(0, {0, 1, 2, 3, 4}, "T1")
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    message = (
        "ibm_manila reports no valid T1 on any working qubit (for example, qubit 1: T1 = 0 us)"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ibm_public.pull("ibm_manila")


def _gate(props: dict[str, Any], gate: str, qubits: list[int]) -> list[dict[str, Any]]:
    return next(e for e in props["gates"] if (e["gate"], e["qubits"]) == (gate, qubits))[
        "parameters"
    ]


def _param(params: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(p for p in params if p["name"] == name)


_IN_MINUTES = {
    "T1 of qubit 0": lambda p: _param(p["qubits"][0], "T1").update(unit="min"),
    "readout_length of qubit 4": lambda p: _param(p["qubits"][4], "readout_length").update(
        unit="min"
    ),
    "gate_length of cx on qubits 3-4": lambda p: _param(
        _gate(p, "cx", [3, 4]), "gate_length"
    ).update(unit="min"),
}


@pytest.mark.parametrize("field", list(_IN_MINUTES))
def test_a_time_in_a_unit_it_does_not_know_names_the_field_and_the_unit(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    raw = _properties(_IN_MINUTES[field])
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw)
    with pytest.raises(nv.SourceDataError) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint ({url}): {field} has the unknown time unit 'min', not one of"
        " ns, us, µs, ms or s",
        "pass an earlier at= to use an older calibration",
    )


def _cx_3_4(name: str, value: float) -> Callable[[dict[str, Any]], object]:
    return lambda p: _param(_gate(p, "cx", [3, 4]), name).update(value=value)


_OUT_OF_RANGE = {
    "a readout error above 1": (
        lambda p: _param(p["qubits"][0], "prob_meas0_prep1").update(value=1.5),
        "readout.p0_given_1 of qubit 0: Input should be less than or equal to 1, got 1.5",
    ),
    "a readout error above 1 after a qubit with no data": (
        lambda p: (
            p["qubits"].__setitem__(1, []),
            _param(p["qubits"][3], "prob_meas0_prep1").update(value=1.5),
        ),
        "readout.p0_given_1 of qubit 3: Input should be less than or equal to 1, got 1.5",
    ),
    "a gate error that is not a number": (
        _cx_3_4("gate_error", math.nan),
        "avg_infidelity of cx on qubits 3-4: Input should be a finite number, got nan",
    ),
    "a negative gate length": (
        _cx_3_4("gate_length", -5),
        "cx on qubits 3-4: duration_ns must not be negative, got -5.0",
    ),
}


@pytest.mark.parametrize("case", list(_OUT_OF_RANGE))
def test_a_value_a_profile_cannot_hold_names_the_reply_and_the_value(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    edit, problem = _OUT_OF_RANGE[case]
    raw = _properties(edit)
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw)
    with pytest.raises(nv.SourceDataError) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint ({url}): {problem}",
        "pass an earlier at= to use an older calibration",
    )


_REPEATED = {
    "T1 of qubit 0": (lambda p: p["qubits"][0], "T1", 99, "131.5286444531517 us", "99 us"),
    "gate_error of cx on qubits 3-4": (
        lambda p: _gate(p, "cx", [3, 4]),
        "gate_error",
        0.5,
        "0.005696275468624307",
        "0.5",
    ),
}


@pytest.mark.parametrize("field", list(_REPEATED))
@pytest.mark.parametrize("reverse", [False, True], ids=["appended", "reversed"])
def test_a_parameter_given_twice_with_two_values_names_both(
    monkeypatch: pytest.MonkeyPatch, field: str, reverse: bool
) -> None:
    params_of, name, value, given, added = _REPEATED[field]

    def repeat(props: dict[str, Any]) -> None:
        params = params_of(props)
        params.append({**_param(params, name), "value": value})
        if reverse:
            params.reverse()

    raw = _properties(repeat)
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw)
    with pytest.raises(nv.SourceDataError) as info:
        ibm_public.pull("ibm_fez")
    url = f"{ibm_public.BASE_URL}/ibm_fez/properties"
    first, second = (added, given) if reverse else (given, added)
    assert info.value.message == (
        f"IBM's public endpoint ({url}): {field} has two different values, {first} and {second}"
    )


@pytest.mark.parametrize("field", list(_REPEATED))
def test_a_parameter_given_twice_with_one_value_counts_once(
    served: list[str], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    params_of, name, *_ = _REPEATED[field]
    raw = _properties(lambda p: params_of(p).append(dict(_param(params_of(p), name))))
    serve = ibm_public.fetch
    monkeypatch.setattr(ibm_public, "fetch", lambda url: raw if "properties" in url else serve(url))
    assert ibm_public.pull("ibm_manila").fingerprint == nv.load("ibm_manila").fingerprint


def _moved_rz(props: dict[str, Any]) -> None:
    next(g for g in props["gates"] if g["gate"] == "rz")["qubits"] = [99]


def _skipped(gate: str) -> Callable[[dict[str, Any]], None]:
    return lambda props: props["gates"].append({"gate": gate, "qubits": [5], "parameters": []})


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_moved_rz, "gate rz is on [99]; the calibration has 5 qubits"),
        (_skipped("measure"), "gate measure is on [5]; the calibration has 5 qubits"),
        (_skipped("delay"), "gate delay is on [5]; the calibration has 5 qubits"),
    ],
    ids=["virtual-rz", "measure", "delay"],
)
def test_a_gate_on_a_qubit_the_calibration_does_not_have_is_named(
    monkeypatch: pytest.MonkeyPatch, edit: Callable[[dict[str, Any]], None], problem: str
) -> None:
    props = json.loads(PROPERTIES)
    edit(props)
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    with pytest.raises(nv.SourceDataError) as info:
        ibm_public.pull("ibm_manila")
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint ({ibm_public.properties_url('ibm_manila')}): {problem}",
        "pass an earlier at= to use an older calibration",
    )


_ERROR = [{"name": "gate_error", "value": 0.001}]


@pytest.mark.parametrize(
    ("entries", "problem"),
    [
        ([("sx", [0, 1], [])], "gate sx is on 2 qubits [0, 1]; sx acts on 1 qubit"),
        (
            [("foo", [0], _ERROR), ("foo", [0, 1], [])],
            "gate foo is on 2 qubits [0, 1]; the first foo entry is on 1 qubit",
        ),
    ],
    ids=["known-gate", "unknown-gate"],
)
def test_a_gate_on_the_wrong_number_of_qubits_names_the_gate_and_its_qubits(
    monkeypatch: pytest.MonkeyPatch, entries: list[tuple[str, list[int], list]], problem: str
) -> None:
    props = json.loads(PROPERTIES)
    props["gates"] += [{"gate": g, "qubits": q, "parameters": p} for g, q, p in entries]
    monkeypatch.setattr(ibm_public, "fetch", lambda url: json.dumps(props).encode())
    with pytest.raises(nv.SourceDataError) as info:
        ibm_public.pull("ibm_manila")
    assert (info.value.message, info.value.hint) == (
        f"IBM's public endpoint ({ibm_public.properties_url('ibm_manila')}): {problem}",
        "pass an earlier at= to use an older calibration",
    )
