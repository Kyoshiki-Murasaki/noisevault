from __future__ import annotations

import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from conftest import MANILA_V01, migrated, toy

import noisevault as nv
from noisevault.catalog import bundled_profiles, vault_dir, vault_path
from noisevault.errors import AmbiguousRef, FingerprintMismatch, ProfileNotFound
from noisevault.profile import Profile

FRAMEWORKS = {
    "qiskit",
    "qiskit_aer",
    "qiskit_ibm_runtime",
    "cirq",
    "cirq_google",
    "pennylane",
    "stim",
}


def _dated(stamp: str, error: float = 1e-3) -> Profile:
    data = toy(
        device={
            "name": "toy",
            "vendor": "test",
            "technology": "superconducting",
            "num_qubits": 3,
            "calibrated_at": stamp,
        }
    )
    data["gates"]["sx"]["avg_infidelity"] = error
    return Profile.model_validate(data)


def test_bundled_manila_loads_by_id_and_date() -> None:
    by_id = nv.load("ibm_manila")
    assert by_id.id == "ibm_manila"
    assert nv.load("ibm_manila@2024-05-27").fingerprint == by_id.fingerprint
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z").fingerprint == by_id.fingerprint
    old = migrated(MANILA_V01)  # the 0.1 snapshot of the same calibration keeps its numbers
    for record in old.calibrations:
        now = by_id.table.gate(record.gate, record.qubits)
        assert now.avg_infidelity == record.avg_infidelity
        assert now.duration_ns == pytest.approx(record.duration_ns, rel=1e-11)
    assert [q.readout.pair for q in by_id.qubits] == [q.readout.pair for q in old.qubits]
    [info] = [i for i in bundled_profiles() if i.id == "ibm_manila"]
    assert info.fingerprint == by_id.fingerprint and info.location == "bundled"


def test_load_by_path(tmp_path: Path) -> None:
    path = _dated("2025-01-01T00:00:00Z").save(tmp_path / "mine.json.gz")
    assert nv.load(path).id == "test_toy"
    assert nv.load(str(path)).id == "test_toy"
    with pytest.raises(ProfileNotFound):
        nv.load(tmp_path / "missing.json")


def test_expect_pins_the_fingerprint() -> None:
    profile = nv.load("ibm_manila")
    assert nv.load("ibm_manila", expect=profile.short_fingerprint) == profile
    assert nv.load("ibm_manila", expect="sha256:" + profile.fingerprint) == profile
    other = "0" * 64
    with pytest.raises(FingerprintMismatch, match=profile.short_fingerprint):
        nv.load("ibm_manila", expect=other)
    with pytest.raises(FingerprintMismatch):
        nv.load("ibm_manila", expect="nv:000000000000")
    with pytest.raises(ValueError, match="full sha256"):
        nv.load("ibm_manila", expect="nv:abc")


def test_vault_refs_newest_date_and_ambiguity(vault: Path) -> None:
    for stamp, error in (
        ("2025-01-01T08:00:00Z", 1e-3),
        ("2025-01-01T20:00:00Z", 2e-3),
        ("2025-02-01T08:00:00Z", 3e-3),
    ):
        _dated(stamp, error).save(vault_path(_dated(stamp, error)))
    assert vault_dir() == vault
    assert nv.load("test_toy").gates["sx"].avg_infidelity == 3e-3
    assert nv.load("test_toy@2025-02-01").gates["sx"].avg_infidelity == 3e-3
    assert nv.load("test_toy@2025-01-01T20:00:00Z").gates["sx"].avg_infidelity == 2e-3
    with pytest.raises(AmbiguousRef) as info:
        nv.load("test_toy@2025-01-01")
    assert "2025-01-01T08:00:00Z" in str(info.value) and "2025-01-01T20:00:00Z" in str(info.value)
    with pytest.raises(ProfileNotFound, match="available"):
        nv.load("test_toy@2024-12-31")


def test_unknown_id_suggests_close_matches() -> None:
    with pytest.raises(ProfileNotFound, match="ibm_manila"):
        nv.load("ibm_manilla")


def test_vault_copy_of_a_bundled_profile_is_not_ambiguous(vault: Path) -> None:
    manila = nv.load("ibm_manila")
    manila.save(vault_path(manila))
    assert nv.load("ibm_manila@2024-05-27") == manila
    infos = [i for i in nv.profiles() if i.id == "ibm_manila"]
    assert [i.location for i in infos] == ["vault"]


def test_profiles_filters(vault: Path) -> None:
    _dated("2025-01-01T00:00:00Z").save(vault / "toy.json")
    assert {i.id for i in nv.profiles()} >= {"ibm_manila", "test_toy"}
    assert {i.id for i in nv.profiles(vendor="test")} == {"test_toy"}
    ions = nv.profiles(technology="trapped_ion")
    assert {i.technology for i in ions} <= {"trapped_ion"}
    assert not {"ibm_manila", "test_toy"} & {i.id for i in ions}


def test_unreadable_vault_file_is_skipped_with_a_warning(vault: Path) -> None:
    vault.mkdir(parents=True)
    (vault / "broken.json").write_text("{}")
    with pytest.warns(UserWarning, match="broken.json"):
        assert "ibm_manila" in {i.id for i in nv.profiles()}


def test_import_loads_no_framework() -> None:
    code = (
        "import sys, noisevault as nv; nv.load('ibm_manila').summary();"
        f" print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_stim_helpers_are_reachable_as_nv_stim() -> None:
    import noisevault.frameworks.stim as stim_module

    assert nv.stim is stim_module


def _changed_manila(error: float) -> Profile:
    data = nv.load("ibm_manila").to_dict()
    data["gates"]["sx"] = {**data["gates"]["sx"], "avg_infidelity": error}
    return Profile.model_validate(data)


def test_vault_profile_shadows_a_bundled_one_of_the_same_time(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    mine = _changed_manila(5e-4)
    mine.save(vault_path(mine))
    for ref in ("ibm_manila", "ibm_manila@2024-05-27", "ibm_manila@2024-05-27T18:27:23Z"):
        assert nv.load(ref).fingerprint == mine.fingerprint, ref
    assert nv.load("ibm_manila", expect=bundled.short_fingerprint) == bundled
    with pytest.raises(FingerprintMismatch, match=mine.short_fingerprint):
        nv.load("ibm_manila", expect="nv:000000000000")


def test_same_time_profiles_in_the_vault_are_told_apart_by_expect(vault: Path) -> None:
    first, second = _changed_manila(5e-4), _changed_manila(6e-4)
    first.save(vault / "first.json.gz")
    second.save(vault / "second.json.gz")
    with pytest.raises(AmbiguousRef, match="expect='nv:...'") as info:
        nv.load("ibm_manila@2024-05-27T18:27:23Z")
    assert "first.json.gz" in str(info.value) and "second.json.gz" in str(info.value)
    assert nv.load("ibm_manila", expect=second.fingerprint) == second


def test_profiles_are_sorted_by_id_then_date(vault: Path) -> None:
    for stamp in ("2025-02-01T08:00:00Z", "2025-01-01T08:00:00Z"):
        _dated(stamp).save(vault_path(_dated(stamp)))
    infos = nv.profiles()
    assert [i.id for i in infos] == sorted(i.id for i in infos)
    toys = [i.ref for i in infos if i.id == "test_toy"]
    assert toys == ["test_toy@2025-01-01T08:00:00Z", "test_toy@2025-02-01T08:00:00Z"]


def test_vault_listing_parses_each_file_once(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import noisevault.catalog as catalog

    first = _dated("2025-01-01T00:00:00Z")
    path = first.save(vault_path(first))
    parsed: list[Path] = []
    real = catalog.load_file
    monkeypatch.setattr(catalog, "load_file", lambda p: parsed.append(p) or real(p))
    assert [i.fingerprint for i in catalog.vault_profiles()] == [first.fingerprint]
    assert [i.fingerprint for i in catalog.vault_profiles()] == [first.fingerprint]
    assert parsed == [path]  # the second listing came from the index
    changed = _dated("2025-01-01T00:00:00Z", error=4.25e-3)  # another size, too
    changed.save(path)
    assert [i.fingerprint for i in catalog.vault_profiles()] == [changed.fingerprint]
    assert parsed == [path, path]


def test_listing_carries_data_kind_and_license() -> None:
    [info] = [i for i in nv.profiles() if i.id == "ibm_manila"]
    assert (info.data_kind, info.license) == ("measured", "Apache-2.0")


def test_readme_fingerprint_pin_loads() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    pins = re.findall(r'nv\.load\("([^"]+)", expect="(nv:[0-9a-f]{12})"\)', readme)
    assert pins, "README no longer shows a pinned load"
    for ref, expect in pins:
        assert nv.load(ref, expect=expect).short_fingerprint == expect


def test_dot_files_in_the_vault_are_not_profiles(vault: Path) -> None:
    vault.mkdir(parents=True)
    (vault / ".index.json.4242.tmp").write_text("{\x01")  # a cache write cut short
    (vault / ".ibm_x@2025.json.gz.4242.tmp.json.gz").write_bytes(b"\x1f\x8b")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "ibm_manila" in {i.id for i in nv.profiles()}
