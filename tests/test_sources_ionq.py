"""IonQ profiles from the public characterization endpoint.

Offline tests serve ``fixtures/ionq/responses.json`` (made-up numbers in the real response
shape) in place of the network.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pytest

from noisevault.errors import SourceUnavailable
from noisevault.reference import Op, probabilities
from noisevault.sources import ionq

RESPONSES = json.loads((Path(__file__).parent / "fixtures" / "ionq" / "responses.json").read_text())


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Serve the fixture; every URL the module asks for must be one of them."""
    bodies = {url: json.dumps(body).encode() for url, body in RESPONSES.items()}

    def get(url: str) -> bytes:
        assert url in bodies, f"unexpected request {url}"
        return bodies[url]

    monkeypatch.setattr(ionq, "_get", get)
    return bodies


@pytest.mark.parametrize("name", ["ionq_forte-1", "forte-1", "qpu.forte-1", " QPU.Forte-1 "])
def test_device_names_accept_every_spelling(name: str) -> None:
    assert ionq.backend_name(name) == "qpu.forte-1"


def test_newest_record_with_its_own_fidelities_is_chosen(served) -> None:
    # The newest record lacks 1Q; the next lacks 2Q, but on the 10-record page the API has
    # filled it from the newer record. Its own one-record page shows it missing, so skip it.
    profile = ionq.pull("forte-1")
    assert profile.device.calibrated_at.isoformat() == "2026-09-27T00:00:00+00:00"
    assert "skipped 2 newer record(s) without 1Q/2Q fidelities" in profile.provenance.notes
    url = ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")
    assert profile.provenance.source_url == url
    assert profile.provenance.source_hash == "sha256:" + hashlib.sha256(served[url]).hexdigest()


def test_medians_become_device_defaults_with_stated_assumptions(served) -> None:
    profile = ionq.pull("qpu.forte-1")
    assert (profile.id, profile.device.num_qubits) == ("ionq_forte-1", 4)
    assert profile.connectivity == "all_to_all"
    one, two = profile.gates["r"], profile.gates["rzz"]
    assert (one.avg_infidelity, one.duration_ns) == (0.0002, 130_000)
    assert (two.avg_infidelity, two.duration_ns) == (0.0049, 970_000)
    for spec in (one, two):
        assert (spec.statistic, spec.stderr) == ("median", None)  # IonQ's stderr 0 is dropped
        assert "IonQ does not state the fidelity metric" in spec.assumption
    assert (profile.readout.error, profile.readout.duration_ns) == (0.0088, 150_000)
    assert (profile.idle.t1_us, profile.idle.t2_us) == (1e8, 1e6)
    assert any("nominal" in note for note in profile.provenance.notes)
    assert profile.prep is None
    prov = profile.provenance
    assert (prov.source_kind, prov.redistributable, prov.attribution) == (
        "public_api",
        "no",
        "IonQ",
    )


def test_placeholder_spam_is_missing_not_data(served) -> None:
    profile = ionq.pull("ionq_forte-1", at="2026-09-01")
    assert profile.device.calibrated_at.date().isoformat() == "2026-09-01"
    assert profile.readout is None
    assert profile.table.qubit(0).readout is None


def test_ms_backends_and_partial_connectivity(served) -> None:
    profile = ionq.pull("aria-1")
    assert "ms" in profile.gates and "rzz" not in profile.gates
    assert profile.connectivity.edges == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    ("device", "two_qubit", "params"), [("forte-1", "rzz", (0.8,)), ("aria-1", "ms", (0.0, 0.0))]
)
def test_profiles_convert_on_their_natives(served, device: str, two_qubit: str, params) -> None:
    profile = ionq.pull(device)
    ops = [Op("r", (0,), (np.pi / 2, 0.0)), Op(two_qubit, (0, 1), params), Op("rz", (1,), (0.3,))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert probabilities(profile, ops, 2).sum() == pytest.approx(1)


def test_implausible_medians_are_skipped_and_named(served) -> None:
    # aria-1's newest record has 1Q fidelity 0.79 next to a normal 2Q: corrupt, not data.
    profile = ionq.pull("aria-1")
    assert profile.device.calibrated_at.date().isoformat() == "2026-02-18"
    assert profile.gates["r"].avg_infidelity == pytest.approx(0.0068)
    [note] = [n for n in profile.provenance.notes if "implausible" in n]
    assert (
        "skipped 1 newer record(s)" in note and "corrupt: 2026-03-01 (1Q 0.79, 2Q 0.9861)" in note
    )


@pytest.mark.parametrize(
    ("one", "two", "usable"),
    [(0.9998, 0.995, True), (0.79, 0.995, False), (0.9998, 0.75, False), (0.9706, 0.9882, False)],
)
def test_gate_medians_below_floor_or_inverted_are_rejected(one, two, usable) -> None:
    record = {
        "date": "2025-01-01T00:00:00Z",
        "fidelity": {"1q": {"median": one}, "2q": {"median": two}},
    }
    assert (ionq._rejection(record) is None) == usable


def test_implausible_spam_leaves_readout_unknown(served) -> None:
    body = json.loads(served[ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")])
    record = body["characterizations"][0]
    record["fidelity"]["spam"]["median"] = 0.7261
    listing = RESPONSES["https://api.ionq.co/v0.4/backends"][0]
    profile = ionq.to_profile(record, listing, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.readout is None
    assert any("SPAM fidelity median 0.7261" in note for note in profile.provenance.notes)


def test_a_dated_record_keeps_its_own_qubit_count(served) -> None:
    body = json.loads(served[ionq._page_url("qpu.forte-1", limit=1, end="2026-09-27T00:00:00Z")])
    record = body["characterizations"][0]
    grown = {**RESPONSES["https://api.ionq.co/v0.4/backends"][0], "qubits": 8}
    profile = ionq.to_profile(record, grown, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.device.num_qubits == record["qubits"] == 4
    assert profile.connectivity == "all_to_all"

    del record["qubits"]
    profile = ionq.to_profile(record, grown, source_url="u", source_hash="sha256:" + "0" * 64)
    assert profile.device.num_qubits == 8
    assert any("qubit count (8)" in note for note in profile.provenance.notes)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ([2, 4], r"edge \[2, 4\] is outside 0..3"),
        ([2, 2], r"edge \[2, 2\] joins a qubit to itself"),
    ],
)
def test_a_bad_edge_is_rejected_even_among_as_many_edges_as_a_complete_graph(
    served, bad: list[int], error: str
) -> None:
    for url in [u for u in served if "qpu.forte-1/characterizations" in u]:
        body = json.loads(served[url])
        for record in body["characterizations"] or []:
            record["connectivity"] = [bad if e == [2, 3] else e for e in record["connectivity"]]
        served[url] = json.dumps(body).encode()
    with pytest.raises(ValueError, match=error):
        ionq.pull("forte-1")


def test_errors_say_what_to_do(served) -> None:
    with pytest.raises(SourceUnavailable, match="qpu.aria-1, qpu.forte-1, qpu.harmony"):
        ionq.pull("forte-9")
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1", at="2025-01-01")
    assert info.value.message == (
        "IonQ publishes no qpu.forte-1 characterization at or before 2025-01-01 with plausible"
        " 1Q and 2Q fidelities of its own (0 record(s) checked)"
    )
    assert info.value.hint == "pass a later at= or none"
    # every harmony record lacks a 1Q fidelity, so no date can help
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("harmony")
    assert info.value.message == (
        "IonQ publishes no qpu.harmony characterization with plausible 1Q and 2Q fidelities of"
        " its own (2 record(s) checked)"
    )
    assert info.value.hint == "use another backend"


def test_a_search_that_runs_out_of_probes_says_to_search_older(
    served, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ionq, "_MAX_PROBES", 1)
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert info.value.message == (
        "none of the 4 newest qpu.forte-1 characterizations has plausible 1Q and 2Q fidelities"
        " of its own"
    )
    assert info.value.hint == "pass an earlier at= to search older records"


def test_a_record_without_fidelities_cannot_make_a_profile() -> None:
    record = {"backend": "qpu.forte-1", "id": "r1", "qubits": 4, "fidelity": {}}
    with pytest.raises(SourceUnavailable) as info:
        ionq.to_profile(record, {"qubits": 4}, source_url="u", source_hash="h")
    assert info.value.message == "IonQ published qpu.forte-1 record r1 without 1Q/2Q fidelities"
    assert info.value.hint == "use a record from another date"


def _refuse(request: urllib.request.Request, timeout: float) -> None:
    raise urllib.error.URLError("timed out")


def _throttle(request: urllib.request.Request, timeout: float) -> None:
    raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)


@pytest.mark.parametrize(
    ("urlopen", "message", "hint"),
    [
        (
            _refuse,
            "could not reach IonQ's API (timed out)",
            "check the network connection, or run nv list to see every profile you can load"
            " offline",
        ),
        (_throttle, f"IonQ's API answered HTTP 429 for {ionq.API}/backends", "try again later"),
    ],
)
def test_a_failed_request_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, urlopen, message: str, hint: str
) -> None:
    monkeypatch.setattr(ionq.urllib.request, "urlopen", urlopen)
    with pytest.raises(SourceUnavailable) as info:
        ionq.pull("forte-1")
    assert (info.value.message, info.value.hint) == (message, hint)


def test_nothing_is_bundled() -> None:
    assert ionq.bundled_profiles() == []


@pytest.mark.network
def test_live_forte_1() -> None:
    profile = ionq.pull("qpu.forte-1")
    assert profile.device.num_qubits == 36
    assert profile.connectivity == "all_to_all"
    assert profile.gates["rzz"].statistic == "median"
    assert 0 < profile.gates["rzz"].avg_infidelity < 0.1
