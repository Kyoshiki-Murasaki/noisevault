"""Where profiles live (the local vault, then the bundled set) and how refs resolve.

The vault is ``$NOISEVAULT_HOME/profiles``, by default ``~/.noisevault/profiles``. Loading never
touches the network; only :func:`pull` does.
"""

from __future__ import annotations

import difflib
import importlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal

from .errors import AmbiguousRef, FingerprintMismatch, NoiseVaultWarning, ProfileNotFound
from .profile import Profile, Ref, load_bytes, load_file, parse_ref

_PULL_SOURCES = {
    "ibm": "noisevault.sources.ibm_public",
    "ibm-account": "noisevault.sources.ibm_account",
    "ionq": "noisevault.sources.ionq",
}


@dataclass(frozen=True)
class ProfileInfo:
    id: str
    calibrated_at: datetime | None
    technology: str
    vendor: str | None
    num_qubits: int
    fingerprint: str
    location: Literal["vault", "bundled"]
    path: Path | Traversable

    @property
    def ref(self) -> str:
        return f"{self.id}@{_stamp(self.calibrated_at)}" if self.calibrated_at else self.id

    def load(self) -> Profile:
        return load_bytes(self.path.read_bytes())

    @classmethod
    def of(cls, profile: Profile, location: Literal["vault", "bundled"], path: Path) -> ProfileInfo:
        dev = profile.device
        return cls(
            profile.id,
            dev.calibrated_at,
            dev.technology,
            dev.vendor,
            dev.num_qubits,
            profile.fingerprint,
            location,
            path,
        )


def vault_dir() -> Path:
    home = os.environ.get("NOISEVAULT_HOME")
    return (Path(home) if home else Path.home() / ".noisevault") / "profiles"


def bundled_dir() -> Traversable:
    return files("noisevault") / "data" / "profiles"


def vault_path(profile: Profile) -> Path:
    """Default vault file for a profile: ``<id>@<UTC timestamp>.json.gz``."""
    when = profile.device.calibrated_at
    stamp = when.strftime("%Y-%m-%dT%H%M%SZ") if when else "undated"
    return vault_dir() / f"{profile.id}@{stamp}.json.gz"


def bundled_profiles() -> list[ProfileInfo]:
    index = json.loads((bundled_dir() / "index.json").read_text(encoding="utf-8"))
    return [
        ProfileInfo(
            id=entry["id"],
            calibrated_at=datetime.fromisoformat(entry["calibrated_at"])
            if entry.get("calibrated_at")
            else None,
            technology=entry["technology"],
            vendor=entry.get("vendor"),
            num_qubits=entry["num_qubits"],
            fingerprint=entry["fingerprint"],
            location="bundled",
            path=bundled_dir() / entry["file"],
        )
        for entry in index["profiles"]
    ]


def vault_profiles() -> list[ProfileInfo]:
    out = []
    for path in sorted(vault_dir().glob("*.json*")):
        try:
            profile = load_file(path)
        except Exception as exc:  # one unreadable file must not hide the rest
            warnings.warn(f"skipping {path}: {exc}", NoiseVaultWarning, stacklevel=2)
            continue
        out.append(ProfileInfo.of(profile, "vault", path))
    return out


def profiles(*, technology: str | None = None, vendor: str | None = None) -> list[ProfileInfo]:
    """Every known profile, vault first, then bundled; newest first within an id."""
    found = _dedupe(vault_profiles() + bundled_profiles())
    found = [
        info
        for info in found
        if (technology is None or info.technology == technology)
        and (vendor is None or info.vendor == vendor)
    ]
    return sorted(found, key=lambda i: (i.id, -_epoch(i.calibrated_at)))


def resolve(ref: str | Ref, *, expect: str | None = None) -> ProfileInfo:
    """The single catalog profile a ``Ref`` names, or ProfileNotFound / AmbiguousRef.

    ``expect`` keeps only the profiles with that fingerprint. Of profiles with the same id and
    calibration time, a vault copy shadows a bundled one.
    """
    if isinstance(ref, str):
        parsed = parse_ref(ref)
        if isinstance(parsed, Path):
            raise ValueError(f"{ref!r} is a path, not a catalog ref")
        ref = parsed
    known = _dedupe(vault_profiles() + bundled_profiles())
    candidates = [info for info in known if info.id == ref.id]
    if not candidates:
        close = difflib.get_close_matches(ref.id, sorted({i.id for i in known}), n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else "; run `nv list`"
        raise ProfileNotFound(f"no profile with id {ref.id!r}{hint}")
    if ref.timestamp is not None:
        matches = [i for i in candidates if i.calibrated_at == ref.timestamp]
    elif ref.date is not None:
        matches = [i for i in candidates if i.calibrated_at and i.calibrated_at.date() == ref.date]
    else:
        newest = max(_epoch(i.calibrated_at) for i in candidates)
        matches = [i for i in candidates if _epoch(i.calibrated_at) == newest]
    if not matches:
        dates = ", ".join(sorted(i.ref for i in candidates))
        raise ProfileNotFound(f"no {ref.id} profile at that time; available: {dates}")
    if expect is not None:
        want = _expect_prefix(expect)
        pinned = [i for i in matches if i.fingerprint.startswith(want)]
        if not pinned:
            found = ", ".join(f"nv:{i.fingerprint[:12]}" for i in matches)
            raise FingerprintMismatch(
                f"{ref.id} at that time is {found}, not the expected {expect}"
            )
        matches = pinned
    shadowed = {_epoch(i.calibrated_at) for i in matches if i.location == "vault"}
    matches = [
        i for i in matches if i.location == "vault" or _epoch(i.calibrated_at) not in shadowed
    ]
    if len(matches) > 1:
        listing = "; ".join(f"{i.ref} ({i.location}, nv:{i.fingerprint[:12]})" for i in matches)
        if len({i.calibrated_at for i in matches}) > 1:
            fix = "use a full timestamp"
        else:
            files = ", ".join(str(i.path) for i in matches)
            fix = f"they share a calibration time, so pass expect='nv:...' or load a file: {files}"
        raise AmbiguousRef(f"{ref.id} matches {len(matches)} profiles: {listing}; {fix}")
    return matches[0]


def load(ref: str | Path, *, expect: str | None = None) -> Profile:
    """Load a profile by path, ``id``, ``id@YYYY-MM-DD`` or ``id@<timestamp>``, offline.

    ``expect`` pins the fingerprint (full hex, ``sha256:<hex>`` or ``nv:<12 hex>``); a different
    profile raises FingerprintMismatch.
    """
    target = parse_ref(ref)
    if isinstance(target, Path):
        if not target.exists():
            raise ProfileNotFound(f"no file {target}")
        profile = load_file(target)
    else:
        profile = resolve(target, expect=expect).load()
    if expect is not None:
        _check_expect(profile, expect)
    return profile


def pull(
    device: str,
    *,
    at: str | datetime | None = None,
    source: str | None = None,
    output: str | Path | None = None,
) -> Profile:
    """Fetch a calibration from a live source and save it (to the vault unless ``output``)."""
    source = source or ("ionq" if device.lower().startswith("ionq") else "ibm")
    if source not in _PULL_SOURCES:
        raise ValueError(f"unknown source {source!r}; choose one of {', '.join(_PULL_SOURCES)}")
    module = importlib.import_module(_PULL_SOURCES[source])
    profile = module.pull(device, at=at)
    profile.save(output if output is not None else vault_path(profile))
    return profile


def _expect_prefix(expect: str) -> str:
    """The fingerprint hex an ``expect`` value pins: 64 digits, or 12 after ``nv:``."""
    want = expect.strip().lower().removeprefix("sha256:")
    short = want.startswith("nv:")
    want = want.removeprefix("nv:")
    if not re.fullmatch(r"[0-9a-f]{12}" if short else r"[0-9a-f]{64}", want):
        raise ValueError(f"expect={expect!r}: give a full sha256 fingerprint or nv:<12 hex>")
    return want


def _check_expect(profile: Profile, expect: str) -> None:
    if not profile.fingerprint.startswith(_expect_prefix(expect)):
        raise FingerprintMismatch(
            f"{profile.id} has fingerprint {profile.short_fingerprint}"
            f" ({profile.fingerprint}), not the expected {expect}"
        )


def _dedupe(infos: list[ProfileInfo]) -> list[ProfileInfo]:
    """Keep the first of profiles with the same id, time and fingerprint (vault wins)."""
    seen: set[tuple[str, float, str]] = set()
    out = []
    for info in infos:
        key = (info.id, _epoch(info.calibrated_at), info.fingerprint)
        if key not in seen:
            seen.add(key)
            out.append(info)
    return out


def _epoch(when: datetime | None) -> float:
    return when.timestamp() if when else float("-inf")


def _stamp(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")
