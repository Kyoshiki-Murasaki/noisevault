from __future__ import annotations

import gzip
import json
import re
import runpy
import zlib
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

    Each file names the qiskit-ibm-runtime release it was converted from, and other releases
    ship other snapshots, so the bytes are compared only under that release. CI's bundle job
    installs it. ``scripts/build_catalog.py --check`` does the same for every source at once.
    """
    runtime = require("qiskit_ibm_runtime")
    from noisevault.sources.qiskit_backend import bundled_profiles as ibm_profiles

    ibm = [e for e in INDEX if e["vendor"] == "ibm"]
    raw = {e["id"]: gzip.decompress((DATA / e["file"]).read_bytes()) for e in ibm}
    sources = {json.loads(r)["provenance"]["source"] for r in raw.values()}
    built_with = {re.match(r"qiskit-ibm-runtime (\S+) ", s)[1] for s in sources}
    assert len(built_with) == 1, f"the IBM files come from several releases: {built_with}"
    [release] = built_with
    if runtime.__version__ != release:
        pytest.skip(
            f"the IBM files were converted from qiskit-ibm-runtime {release}, not the installed"
            f" {runtime.__version__}; install {release} to compare them, or rebuild the bundle"
            " with python scripts/build_catalog.py"
        )
    fresh = ibm_profiles()
    assert {p.id for p in fresh} == set(raw)
    by_id = {e["id"]: e for e in ibm}
    for profile in fresh:
        assert raw[profile.id] == canonical_json(profile.to_dict()).encode("utf-8"), profile.id
        assert by_id[profile.id]["fingerprint"] == profile.fingerprint


def _gzip_of_python_3_12(data: bytes, level: int = 9, *, mtime: float | None = None) -> bytes:
    """gzip.compress(data, mtime=0) before Python 3.13: zlib's header, whose OS byte names the
    platform zlib was built for instead of 255."""
    return zlib.compress(data, level, wbits=31)


def test_bundle_bytes_do_not_depend_on_the_pythons_gzip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = runpy.run_path(str(ROOT / "scripts" / "build_catalog.py"), run_name="build_catalog")
    committed = (DATA / "ibm_manila@2024-05-27.json.gz").read_bytes()
    monkeypatch.setattr(gzip, "compress", _gzip_of_python_3_12)
    built = build["write_bundle"]([load_bytes(committed)], tmp_path)
    assert built["ibm_manila@2024-05-27.json.gz"] == committed
