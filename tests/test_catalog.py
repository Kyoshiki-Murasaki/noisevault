from __future__ import annotations

import errno
import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
import zlib
from pathlib import Path

import pytest
from conftest import MANILA_V01, migrated, require, toy
from typer.testing import CliRunner

import noisevault as nv
from noisevault import catalog
from noisevault.catalog import bundled_profiles, vault_dir, vault_path
from noisevault.cli import app
from noisevault.errors import AmbiguousRef, FingerprintMismatch, ProfileNotFound, SourceUnavailable
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
    assert "2025-01-01T08:00:00Z" in info.value.message
    assert "2025-01-01T20:00:00Z" in info.value.message
    assert info.value.hint == "use a full timestamp"
    with pytest.raises(ProfileNotFound, match="2024-12-31 UTC; you have test_toy@2025-01-01T08"):
        nv.load("test_toy@2024-12-31")


def test_a_ref_date_that_misses_says_it_names_the_calibration_day() -> None:
    misses = {
        "ibm_fez@2025-03-01": (
            "no ibm_fez profile calibrated on 2025-03-01 UTC;"
            " you have ibm_fez@2025-02-26T20:16:25Z",
            "to fetch the calibration in effect at 2025-03-01, run nv pull ibm_fez --at 2025-03-01",
        ),
        "quantinuum_h2-1@2020-01-01": (
            "no quantinuum_h2-1 profile calibrated on 2020-01-01 UTC;"
            " you have quantinuum_h2-1@2025-04-30T00:00:00Z",
            None,
        ),
        "ibm_fez@2025-02-26T00:00:00Z": (
            "no ibm_fez profile calibrated at 2025-02-26T00:00:00Z;"
            " you have ibm_fez@2025-02-26T20:16:25Z",
            None,
        ),
    }
    for ref, (message, hint) in misses.items():
        with pytest.raises(ProfileNotFound) as info:
            nv.load(ref)
        assert (info.value.message, info.value.hint) == (message, hint)
        assert str(info.value) == (f"{message}; {hint}" if hint else message)


def test_an_unknown_id_says_where_to_see_the_ids() -> None:
    with pytest.raises(ProfileNotFound) as info:
        nv.load("xyzzy")
    assert info.value.message == "no profile with id 'xyzzy'"
    assert info.value.hint == "run nv list to see every profile you can load offline"


def test_unknown_id_suggests_close_matches() -> None:
    with pytest.raises(ProfileNotFound, match="ibm_manila"):
        nv.load("ibm_manilla")


def test_an_id_that_also_names_a_folder_here_loads_the_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ibm_manila").mkdir()
    assert nv.load("ibm_manila") == nv.load("ibm_manila@2024-05-27")
    assert nv.catalog.resolve("ibm_manila").location == "bundled"


def test_pulling_a_bundled_device_no_source_serves_says_how_to_load_it() -> None:
    with pytest.raises(SourceUnavailable) as info:
        nv.pull("google_weber")
    assert info.value.hint == "google_weber is bundled, so nv.load('google_weber') loads it offline"
    assert str(info.value).endswith(f"; {info.value.hint}")


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
    require("stim")
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


def _later_manila() -> Profile:
    data = _changed_manila(5e-4).to_dict()
    data["device"]["calibrated_at"] = "2024-06-03T10:00:00Z"
    return Profile.model_validate(data)


def test_a_fingerprint_mismatch_names_the_calibration_that_has_the_pin(vault: Path) -> None:
    bundled = nv.load("ibm_manila")
    later = _later_manila()
    path = later.save(vault_path(later))
    pin, held = bundled.short_fingerprint, later.short_fingerprint
    heads = {
        "ibm_manila": f"ibm_manila loads ibm_manila@2024-06-03T10:00:00Z ({held})",
        "ibm_manila@2024-06-03": f"ibm_manila@2024-06-03 is {held}",
        str(path): f"{path} holds ibm_manila@2024-06-03T10:00:00Z ({held})",
    }
    hint = "load ibm_manila@2024-05-27T18:27:23Z, which has that fingerprint"
    for ref, head in heads.items():
        with pytest.raises(FingerprintMismatch) as info:
            nv.load(ref, expect=pin)
        assert info.value.message == f"{head}, not the expected {pin}"
        assert info.value.hint == hint and str(info.value).endswith(f"; {hint}")
    assert nv.load("ibm_manila@2024-05-27T18:27:23Z", expect=pin) == bundled


def test_a_fingerprint_no_profile_has_says_how_to_get_the_file() -> None:
    nowhere, held = "nv:000000000000", nv.load("ibm_manila").short_fingerprint
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("ibm_manila", expect=nowhere)
    assert info.value.message == (
        f"ibm_manila loads ibm_manila@2024-05-27T18:27:23Z ({held}), not the expected"
        f" {nowhere}, and no profile you have has that fingerprint"
    )
    assert info.value.hint == (
        "ask whoever pinned it for the profile file, or, if the source still serves that"
        " calibration, run nv pull ibm_manila --at <a time it was in effect>"
    )
    with pytest.raises(FingerprintMismatch) as info:
        nv.load("quantinuum_h2-1", expect=nowhere)
    assert info.value.hint == "ask whoever pinned it for the profile file"


def test_same_time_profiles_in_the_vault_are_told_apart_by_expect(vault: Path) -> None:
    first, second = _changed_manila(5e-4), _changed_manila(6e-4)
    first.save(vault / "first.json.gz")
    second.save(vault / "second.json.gz")
    with pytest.raises(AmbiguousRef) as info:
        nv.load("ibm_manila@2024-05-27T18:27:23Z")
    assert info.value.message.startswith("ibm_manila matches 2 profiles that share a calibration")
    assert info.value.hint == (
        "pass expect='nv:...' or load one of their files:"
        f" {vault / 'first.json.gz'}, {vault / 'second.json.gz'}"
    )
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


def _serve(monkeypatch: pytest.MonkeyPatch, profile: Profile) -> None:
    from noisevault.sources import ibm_public

    monkeypatch.setattr(ibm_public, "pull", lambda device, at=None: profile)


def _pull_quietly(**options) -> catalog.Pulled:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return catalog.pull_and_save("ibm_toy", source="ibm", **options)


def test_pulls_a_fraction_of_a_second_apart_keep_both_calibrations(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    first = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    second = _dated("2024-05-27T18:27:23.900000Z", 2e-3)
    for profile in (first, second):
        _serve(monkeypatch, profile)
        assert _pull_quietly().written
    assert len(list(vault.glob("*.json.gz"))) == 2
    assert nv.load("test_toy@2024-05-27T18:27:23.1Z") == first
    assert nv.load("test_toy@2024-05-27T18:27:23.9Z") == second


def test_vault_files_named_by_whole_seconds_still_load_and_are_reused(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    held = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    legacy = held.save(vault / "test_toy@2024-05-27T182723Z.json.gz")
    assert [i.ref for i in nv.profiles() if i.id == "test_toy"] == [
        "test_toy@2024-05-27T18:27:23.100000Z"
    ]
    _serve(monkeypatch, held)
    assert _pull_quietly() == (held, legacy, False)
    reconverted = _dated("2024-05-27T18:27:23.100000Z", 1.5e-3)
    _serve(monkeypatch, reconverted)
    with pytest.warns(UserWarning, match="converted differently"):
        pulled = catalog.pull_and_save("ibm_toy", source="ibm")
    assert pulled.path == legacy and list(vault.glob("*.json.gz")) == [legacy]
    assert nv.load(legacy) == reconverted


def test_a_pull_never_replaces_a_vault_file_of_another_calibration(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    held = _dated("2024-05-27T18:27:23.100000Z", 1e-3)
    legacy = held.save(vault / "test_toy@2024-05-27T182723Z.json.gz")
    _serve(monkeypatch, _dated("2024-05-27T18:27:23Z", 2e-3))
    with pytest.raises(FileExistsError, match="2024-05-27T18:27:23.100000Z"):
        _pull_quietly()
    assert nv.load(legacy) == held


def test_a_failed_pull_to_a_file_leaves_the_existing_file_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _dated("2025-01-01T00:00:00Z").save(tmp_path / "existing.json.gz")
    before = target.read_bytes()
    newer = _dated("2025-02-01T00:00:00Z")
    _serve(monkeypatch, newer)

    def disk_full(self: Path, data: bytes) -> int:
        with open(self, "wb") as handle:
            handle.write(data[:20])
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", disk_full)
        with pytest.raises(OSError, match="No space"):
            _pull_quietly(output=target)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
    assert _pull_quietly(output=target) == (newer, target, True)
    assert nv.load(target) == newer and target.read_bytes()[:2] == b"\x1f\x8b"


@pytest.mark.parametrize(
    "damage",
    [
        lambda entry: [],
        lambda entry: "stale",
        lambda entry: {k: v for k, v in entry.items() if k != "id"},
        lambda entry: {**entry, "calibrated_at": 5},
        lambda entry: {**entry, "id": []},
        lambda entry: {**entry, "num_qubits": "3"},
        lambda entry: {**entry, "num_qubits": True},
        lambda entry: {**entry, "fingerprint": None},
        lambda entry: {**entry, "fingerprint": "nv:1234"},
        lambda entry: {**entry, "vendor": 7},
        lambda entry: {**entry, "license": ["MIT"]},
        lambda entry: {**entry, "calibrated_at": "2025-01-01T00:00:00"},
    ],
    ids=[
        "list",
        "string",
        "missing id",
        "bad timestamp",
        "list id",
        "string qubits",
        "bool qubits",
        "no fingerprint",
        "short fingerprint",
        "int vendor",
        "list license",
        "naive timestamp",
    ],
)
def test_a_damaged_index_entry_is_rebuilt_from_its_file(vault: Path, damage) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    index = vault / ".index.json"
    entry = json.loads(index.read_text())[path.name]
    index.write_text(json.dumps({path.name: damage(entry)}))
    assert catalog.vault_profiles() == [catalog.ProfileInfo.of(profile, "vault", path)]
    assert nv.load("test_toy") == profile
    assert nv.load("ibm_manila").id == "ibm_manila"
    assert json.loads(index.read_text())[path.name] == entry


def test_odd_vault_entries_are_skipped_with_one_line_each(vault: Path) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    (vault / "gone.json.gz").symlink_to(vault.parent / "moved_away.json.gz")
    (vault / "folder.json").mkdir()
    os.mkfifo(vault / "pipe.json")
    (vault / "empty.json").write_text("{}")
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        found = {i.id for i in nv.profiles()}
        assert nv.load("test_toy") == profile
    assert {"test_toy", "ibm_manila"} <= found
    messages = sorted({str(w.message) for w in caught})
    assert [m.split(":")[0] for m in messages] == [
        f"skipped {vault / name}"
        for name in ("empty.json", "folder.json", "gone.json.gz", "pipe.json")
    ]
    assert all("\n" not in m for m in messages)
    assert f"links to {vault.parent / 'moved_away.json.gz'}, which does not exist" in messages[2]
    assert messages[0].endswith(f"; run nv validate {vault / 'empty.json'}")


def test_a_link_to_an_unreadable_file_is_skipped_and_the_rest_still_list(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    hidden = vault.parent / "locked" / "toy.json.gz"
    hidden.parent.mkdir()
    profile.save(hidden)
    link = vault / "locked.json.gz"
    link.symlink_to(hidden)
    real_stat = Path.stat

    def denied(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == link and follow_symlinks:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        assert nv.load("ibm_manila").id == "ibm_manila"
        assert nv.load("test_toy") == profile
    assert {str(w.message) for w in caught} == {f"skipped {link}: {os.strerror(errno.EACCES)}"}

    result = CliRunner().invoke(app, ["list"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "test_toy" in result.stdout and "ibm_manila" in result.stdout
    assert f"warning: skipped {link}: {os.strerror(errno.EACCES)}" in result.stderr


def _lock(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """What ``chmod a-r`` does to an indexed file, also as root: the inode change time moves
    (size and modification time stay) and reading it fails."""
    before = path.stat().st_ctime_ns
    deadline = time.monotonic() + 5
    while path.stat().st_ctime_ns == before:  # file systems with a coarse clock need a retry
        assert time.monotonic() < deadline, "chmod never moved the change time"
        time.sleep(0.01)
        path.chmod(path.stat().st_mode)
    real_read = Path.read_bytes

    def denied(self: Path) -> bytes:
        if self == path:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", denied)


def test_an_indexed_file_made_unreadable_is_skipped_and_the_rest_still_list(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    catalog.vault_profiles()
    _lock(monkeypatch, path)
    skipped = f"skipped {path}: {os.strerror(errno.EACCES)}"
    with pytest.warns(nv.NoiseVaultWarning) as caught:
        found = {i.id for i in nv.profiles()}
    assert "ibm_manila" in found and "test_toy" not in found
    assert {str(w.message) for w in caught} == {skipped}

    result = CliRunner().invoke(app, ["list"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "ibm_manila" in result.stdout and "test_toy" not in result.stdout
    assert f"warning: {skipped}" in result.stderr


def test_a_pull_onto_an_unreadable_vault_file_does_not_call_it_saved(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    path = profile.save(vault_path(profile))
    before = path.read_bytes()
    catalog.vault_profiles()
    _serve(monkeypatch, profile)
    _lock(monkeypatch, path)
    with pytest.warns(nv.NoiseVaultWarning, match="skipped"):
        with pytest.raises(FileExistsError, match="exists but cannot be read"):
            catalog.pull_and_save("ibm_toy", source="ibm")
    monkeypatch.undo()
    assert path.read_bytes() == before


def test_listing_reads_no_profile_once_the_vault_is_indexed(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    profile.save(vault_path(profile))
    catalog.vault_profiles()
    manila = nv.load("ibm_manila")
    expected = {
        "test_toy": (None, None, "unknown"),
        "ibm_manila": ("Falcon r5.11", "package_snapshot", "yes"),
    }
    assert manila.device.processor == "Falcon r5.11"

    def no_parse(*args: object) -> None:
        raise AssertionError("listing parsed a profile file")

    monkeypatch.setattr(catalog, "load_file", no_parse)
    monkeypatch.setattr(catalog, "load_bytes", no_parse)
    runner = CliRunner()
    assert runner.invoke(app, ["list"]).exit_code == 0
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["id"]: row for row in json.loads(result.stdout)}
    for name, values in expected.items():
        row = rows[name]
        assert (row["processor"], row["source_kind"], row["redistributable"]) == values
    assert rows["ibm_manila"]["calibrated_at"] == "2024-05-27T18:27:23Z"


def test_vault_files_have_one_gzip_header_on_every_python(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    """Before Python 3.13, gzip.compress(mtime=0) let zlib write its platform's OS byte."""
    monkeypatch.setattr(
        gzip, "compress", lambda data, level=9, *, mtime=None: zlib.compress(data, level, wbits=31)
    )
    profile = _dated("2025-01-01T00:00:00Z")
    raw = profile.save(vault_path(profile)).read_bytes()
    assert raw[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"  # no time, no name, OS 255
    assert nv.load("test_toy") == profile


def test_concurrent_pulls_of_one_calibration_both_succeed(
    monkeypatch: pytest.MonkeyPatch, vault: Path
) -> None:
    profile = _dated("2025-01-01T00:00:00Z")
    _serve(monkeypatch, profile)
    vault.mkdir(parents=True)
    both_staged = threading.Barrier(2, timeout=10)
    real_replace = os.replace

    def replace_once_both_are_staged(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == vault_path(profile):
            both_staged.wait()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_once_both_are_staged)
    failures: list[BaseException] = []

    def pull() -> None:
        try:
            catalog.pull_and_save("ibm_toy", source="ibm")
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=pull) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert [p.name for p in vault.iterdir()] == [vault_path(profile).name]
    assert nv.load("test_toy") == profile


@pytest.mark.parametrize("source", ["ibm", "ionq"])
def test_an_impossible_date_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    def no_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("the source was asked before the date was checked")

    for module in ("noisevault.sources.ibm_public", "noisevault.sources.ionq"):
        monkeypatch.setattr(f"{module}.pull", no_request)
    with pytest.raises(ValueError, match="not an ISO 8601 date"):
        catalog.pull_and_save("ibm_fez", at="2025-02-30", source=source)
