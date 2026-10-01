"""Rebuild the bundled profiles in src/noisevault/data/profiles/ and the NOTICE data section.

Deterministic: the same installed source packages give byte-identical files (gzip with mtime 0,
sorted keys, names ``<id>@<YYYY-MM-DD>.json.gz``). Run from the repository root:

    python scripts/build_catalog.py           # rewrite the bundle and NOTICE
    python scripts/build_catalog.py --check   # exit 1 if the committed bundle is out of date
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisevault.catalog import index_entry  # noqa: E402
from noisevault.profile import Profile, canonical_json  # noqa: E402

DATA = ROOT / "src" / "noisevault" / "data" / "profiles"
NOTICE = ROOT / "NOTICE"
SOURCES = ("qiskit_backend", "quantinuum", "google")
# Copyright lines of upstream projects whose data is bundled, by repository URL.
COPYRIGHT = {
    "https://github.com/Qiskit/qiskit-ibm-runtime": "Copyright 2021 IBM and its contributors.",
    "https://github.com/Quantinuum/quantinuum-hardware-specifications": (
        "Copyright 2025 Quantinuum (www.quantinuum.com)."
    ),
    "https://github.com/quantumlib/Cirq": "Copyright 2018 The Cirq Developers.",
}
NOTICE_HEAD = """NoiseVault
Copyright 2026 Dev D. Goyal

Licensed under the Apache License, Version 2.0 (see LICENSE).

Bundled calibration data
========================

NoiseVault ships device calibration profiles under src/noisevault/data/profiles/. Each
profile records its source, license and attribution in its "provenance" block. The numbers
are the source's; NoiseVault only converts them to its profile format.
"""


def collect(load: Callable[[str], list[Profile]]) -> list[Profile]:
    """Profiles of every source that implements bundled_profiles(), sorted by id and date."""
    out: list[Profile] = []
    for name in SOURCES:
        try:
            found = load(name)
        except NotImplementedError as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        print(f"{name}: {len(found)} profiles", file=sys.stderr)
        out += found
    for profile in out:
        if profile.provenance.redistributable != "yes":
            raise SystemExit(f"{profile.id} is not redistributable; it cannot be bundled")
    # Bundled data is pinned by package version or source commit, not by download time, so a
    # rebuild from the same sources must produce the same bytes.
    out = [_without_retrieval_time(p) for p in out]
    return sorted(out, key=lambda p: (p.id, p.device.calibrated_at is None, p.device.calibrated_at))


def _without_retrieval_time(profile: Profile) -> Profile:
    if profile.provenance.retrieved_at is None:
        return profile
    provenance = profile.provenance.model_copy(update={"retrieved_at": None})
    return profile.model_copy(update={"provenance": provenance})


def _bundled(name: str) -> list[Profile]:
    return importlib.import_module(f"noisevault.sources.{name}").bundled_profiles()


def file_name(profile: Profile) -> str:
    when = profile.device.calibrated_at
    return f"{profile.id}@{when.date().isoformat() if when else 'undated'}.json.gz"


def write_bundle(profiles: list[Profile], folder: Path) -> dict[str, bytes]:
    """Write every profile and index.json into ``folder``; returns name -> bytes written."""
    files: dict[str, bytes] = {}
    entries = []
    for profile in profiles:
        name = file_name(profile)
        if name in files:
            raise SystemExit(f"two bundled profiles would share the file name {name}")
        files[name] = gzip.compress(canonical_json(profile.to_dict()).encode("utf-8"), mtime=0)
        entries.append(
            {**index_entry(profile), "file": name, "artifact_hash": profile.artifact_hash}
        )
    index = {"format": 1, "profiles": entries}
    files["index.json"] = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*"):
        if stale.is_file() and stale.name not in files:
            stale.unlink()
    for name, raw in files.items():
        (folder / name).write_bytes(raw)
    return files


NOTICE_TAIL = """
Test fixtures
=============

tests/fixtures/ibm/manila_properties.json is taken from the FakeManilaV2 snapshot in
qiskit-ibm-runtime (https://github.com/Qiskit/qiskit-ibm-runtime), and
tests/fixtures/ibm/manila_configuration.json keeps seven fields of that snapshot's configuration
(backend_name, backend_version, basis_gates, coupling_map, dt, n_qubits, processor_type).
  License: Apache-2.0.
  Copyright 2021 IBM and its contributors.
"""


def notice(profiles: list[Profile]) -> str:
    """The NOTICE file: one block per upstream source, listing the files taken from it."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for profile in profiles:
        prov = profile.provenance
        key = (prov.attribution or "unknown", prov.license or "unknown", _repo(prov.source_url))
        groups.setdefault(key, []).append(f"{file_name(profile)}  ({prov.source})")
    blocks = []
    for (attribution, license_, url), lines in sorted(groups.items()):
        head = [f"From {url}" if url else "From an unnamed source", f"  License: {license_}."]
        if url in COPYRIGHT:
            head.append(f"  {COPYRIGHT[url]}")
        head.append(f"  Attribution: {attribution}.")
        blocks.append("\n".join([*head, "  Files:", *(f"    {line}" for line in lines)]))
    return NOTICE_HEAD + "\n" + "\n\n".join(blocks) + "\n" + NOTICE_TAIL


def _repo(url: str | None) -> str:
    """A GitHub URL's repository root, so files from one repository share one NOTICE block."""
    if not url:
        return ""
    parts = url.split("/")
    return "/".join(parts[:5]) if url.startswith("https://github.com/") else url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    profiles = collect(_bundled)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            built = write_bundle(profiles, Path(tmp))
        current = {p.name: p.read_bytes() for p in DATA.glob("*") if p.is_file()}
        stale = sorted(set(built) ^ set(current) | {n for n in built if built[n] != current.get(n)})
        if NOTICE.read_text(encoding="utf-8") != notice(profiles):
            stale.append("NOTICE")
        print("up to date" if not stale else f"out of date: {', '.join(stale)}")
        return 1 if stale else 0
    files = write_bundle(profiles, DATA)
    NOTICE.write_text(notice(profiles), encoding="utf-8")
    total = sum(len(raw) for raw in files.values())
    print(f"wrote {len(files) - 1} profiles + index.json, {total / 1e6:.2f} MB, to {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
