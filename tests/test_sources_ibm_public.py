from __future__ import annotations

import hashlib
import json
import os
import statistics
import urllib.error
from datetime import date
from pathlib import Path

import pytest

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
    message = str(info.value)
    assert "ibm_torino is not listed on the public endpoint" in message
    assert 'nv.load("ibm_torino")' in message and "ibm_fez, ibm_kingston" in message


def test_listed_device_with_no_history_that_far_back(served: list[str]) -> None:
    with pytest.raises(nv.SourceUnavailable, match="no ibm_fez calibration older than 2019"):
        ibm_public.pull("ibm_fez", at="2019-01-01")


def test_network_failure_is_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(ibm_public.urllib.request, "urlopen", refuse)
    with pytest.raises(nv.SourceUnavailable, match="could not reach IBM's public endpoint"):
        ibm_public.pull("ibm_fez")


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


def test_pull_of_an_id_no_source_serves_names_the_sources(vault: Path) -> None:
    with pytest.raises(
        nv.SourceUnavailable, match="no live source pulls 'quantinuum_h2-1'"
    ) as info:
        nv.pull("quantinuum_h2-1")
    assert "nv.load('quantinuum_h2-1') loads it offline" in str(info.value)
    with pytest.raises(nv.SourceUnavailable, match="ionq") as info:
        nv.pull("rigetti_ankaa-3")
    assert "nv.load" not in str(info.value)


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
    note = "Qubit 0 reported T2 = 0 us; treated as missing, the device median applies."
    assert note in profile.provenance.notes
