from __future__ import annotations

import gzip
import json
import re
from importlib.resources import files
from pathlib import Path

import pytest
from conftest import require

from noisevault.catalog import bundled_profiles
from noisevault.profile import canonical_json, load_bytes

ROOT = Path(__file__).resolve().parents[1]
DATA = files("noisevault") / "data" / "profiles"
INDEX = json.loads((DATA / "index.json").read_text(encoding="utf-8"))["profiles"]
BUDGET_BYTES = 2_500_000


@pytest.mark.parametrize("entry", INDEX, ids=[e["file"] for e in INDEX])
def test_bundled_file_matches_its_index_line(entry: dict) -> None:
    profile = load_bytes((DATA / entry["file"]).read_bytes())
    assert profile.fingerprint == entry["fingerprint"]
    assert profile.artifact_hash == entry["artifact_hash"]
    assert entry["file"] == f"{profile.id}@{profile.device.calibrated_at.date()}.json.gz"
    assert (entry["id"], entry["num_qubits"], entry["technology"]) == (
        profile.id,
        profile.device.num_qubits,
        profile.device.technology,
    )
    prov = profile.provenance
    assert prov.redistributable == "yes"
    assert prov.license and prov.source_url and prov.attribution
    assert entry["license"] == prov.license and entry["data_kind"] == prov.data_kind


def test_index_lists_every_file_and_nothing_else() -> None:
    on_disk = {p.name for p in DATA.iterdir() if p.name.endswith(".json.gz")}
    assert on_disk == {e["file"] for e in INDEX}
    assert [(e["id"], e["calibrated_at"]) for e in INDEX] == sorted(
        (e["id"], e["calibrated_at"]) for e in INDEX
    )
    assert {i.id for i in bundled_profiles()} >= {"ibm_fez", "ibm_manila", "ibm_sherbrooke"}


def test_bundle_fits_the_size_budget() -> None:
    total = sum(p.stat().st_size for p in Path(str(DATA)).iterdir())
    assert total < BUDGET_BYTES, f"bundle is {total / 1e6:.2f} MB"


def test_notice_credits_every_bundled_file() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for entry in INDEX:
        assert re.search(rf"^\s+{re.escape(entry['file'])}\s", notice, re.MULTILINE), entry["file"]


def test_bundled_ibm_files_are_what_the_source_produces_today() -> None:
    """The committed IBM files equal a fresh conversion of the installed snapshots.

    ``scripts/build_catalog.py --check`` does the same for every source at once.
    """
    require("qiskit_ibm_runtime")
    from noisevault.sources.qiskit_backend import bundled_profiles as ibm_profiles

    by_id = {e["id"]: e for e in INDEX}
    for profile in ibm_profiles():
        entry = by_id[profile.id]
        raw = gzip.decompress((DATA / entry["file"]).read_bytes())
        assert raw == canonical_json(profile.to_dict()).encode("utf-8"), entry["file"]
        assert entry["fingerprint"] == profile.fingerprint
