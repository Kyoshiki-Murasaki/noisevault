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
import shutil
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from .errors import (
    AmbiguousRef,
    FingerprintMismatch,
    NoiseVaultWarning,
    ProfileNotFound,
    SourceUnavailable,
)
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
    data_kind: str = "unknown"
    license: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.id}@{_stamp(self.calibrated_at)}" if self.calibrated_at else self.id

    def load(self) -> Profile:
        return load_bytes(self.path.read_bytes())

    @classmethod
    def of(cls, profile: Profile, location: Literal["vault", "bundled"], path: Path) -> ProfileInfo:
        return cls.from_entry(index_entry(profile), location, path)

    @classmethod
    def from_entry(
        cls, entry: dict[str, Any], location: Literal["vault", "bundled"], path: Path | Traversable
    ) -> ProfileInfo:
        """From a line of a catalog index (see :func:`index_entry`)."""
        stamp = entry.get("calibrated_at")
        return cls(
            id=entry["id"],
            calibrated_at=datetime.fromisoformat(stamp) if stamp else None,
            technology=entry["technology"],
            vendor=entry.get("vendor"),
            num_qubits=entry["num_qubits"],
            fingerprint=entry["fingerprint"],
            location=location,
            path=path,
            data_kind=entry.get("data_kind", "unknown"),
            license=entry.get("license"),
        )


def index_entry(profile: Profile) -> dict[str, Any]:
    """What a catalog index records about a profile, so listing never parses the file."""
    dev, prov = profile.device, profile.provenance
    return {
        "id": profile.id,
        "date": dev.calibrated_at.date().isoformat() if dev.calibrated_at else None,
        "calibrated_at": _stamp(dev.calibrated_at) if dev.calibrated_at else None,
        "vendor": dev.vendor,
        "technology": dev.technology,
        "num_qubits": dev.num_qubits,
        "processor": dev.processor,
        "data_kind": prov.data_kind,
        "license": prov.license,
        "fingerprint": profile.fingerprint,
    }


def vault_dir() -> Path:
    home = os.environ.get("NOISEVAULT_HOME")
    return (Path(home) if home else Path.home() / ".noisevault") / "profiles"


def bundled_dir() -> Traversable:
    return files("noisevault") / "data" / "profiles"


def vault_path(profile: Profile) -> Path:
    """Default vault file for a profile: ``<id>@<UTC timestamp>.json.gz``.

    Microseconds appear only when nonzero, so whole-second names match older vault files.
    """
    when = profile.device.calibrated_at
    if when is None:
        stamp = "undated"
    else:
        stamp = when.strftime("%Y-%m-%dT%H%M%S.%fZ" if when.microsecond else "%Y-%m-%dT%H%M%SZ")
    return vault_dir() / f"{profile.id}@{stamp}.json.gz"


def bundled_profiles() -> list[ProfileInfo]:
    index = json.loads((bundled_dir() / "index.json").read_text(encoding="utf-8"))
    return [
        ProfileInfo.from_entry(entry, "bundled", bundled_dir() / entry["file"])
        for entry in index["profiles"]
    ]


_VAULT_INDEX = ".index.json"


def vault_profiles() -> list[ProfileInfo]:
    """Profiles in the vault, indexed by file name so unchanged files are not parsed again.

    The index (``.index.json`` in the vault) is only a cache: it is rebuilt from the files
    whenever a file's size or modification time changes, and losing it costs one re-read.
    """
    folder = vault_dir()
    cached = _read_vault_index(folder)
    index: dict[str, dict[str, Any]] = {}
    out = []
    for path in sorted(folder.glob("*.json*")):
        if path.name.startswith("."):  # the index cache and in-flight temporary files
            continue
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        entry = cached.get(path.name)
        info = _cached_info(entry, signature, path)
        if info is None:
            try:
                profile = load_file(path)
            except Exception as exc:  # one unreadable file must not hide the rest
                warnings.warn(f"skipping {path}: {exc}", NoiseVaultWarning, stacklevel=2)
                continue
            entry = {"signature": signature, **index_entry(profile)}
            info = ProfileInfo.from_entry(entry, "vault", path)
        index[path.name] = entry
        out.append(info)
    if index != cached:
        _write_vault_index(folder, index)
    return out


def _cached_info(entry: Any, signature: list[int], path: Path) -> ProfileInfo | None:
    """The cached listing of an unchanged file, or None when the entry is stale or damaged."""
    if not isinstance(entry, dict) or entry.get("signature") != signature:
        return None
    try:
        return ProfileInfo.from_entry(entry, "vault", path)
    except (KeyError, TypeError, ValueError):
        return None


def _read_vault_index(folder: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((folder / _VAULT_INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_vault_index(folder: Path, index: dict[str, dict[str, Any]]) -> None:
    """Best effort and atomic: a read-only vault or a concurrent writer only loses the cache."""
    text = json.dumps(index, sort_keys=True)
    try:
        _write_atomically(folder / _VAULT_INDEX, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    except OSError:
        pass


def profiles(*, technology: str | None = None, vendor: str | None = None) -> list[ProfileInfo]:
    """Every known profile (a vault copy hides an identical bundled one), by id then date."""
    found = _dedupe(vault_profiles() + bundled_profiles())
    found = [
        info
        for info in found
        if (technology is None or info.technology == technology)
        and (vendor is None or info.vendor == vendor)
    ]
    return sorted(found, key=lambda i: (i.id, _epoch(i.calibrated_at), i.location))


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


class Pulled(NamedTuple):
    profile: Profile
    path: Path
    written: bool  # False when the vault already held this calibration


def pull(
    device: str,
    *,
    at: str | date | datetime | None = None,
    source: str | None = None,
    output: str | Path | None = None,
) -> Profile:
    """Fetch a calibration from a live source and save it (to the vault unless ``output``).

    Pulling a calibration the vault already holds (same id and fingerprint) writes nothing, so
    repeated pulls leave one file per calibration. A pull that converts the same calibration
    differently (say, after a NoiseVault upgrade) replaces the older file, with a warning.
    """
    return pull_and_save(device, at=at, source=source, output=output).profile


def pull_and_save(
    device: str,
    *,
    at: str | date | datetime | None = None,
    source: str | None = None,
    output: str | Path | None = None,
) -> Pulled:
    """:func:`pull`, also saying where the profile is and whether this call wrote it."""
    source = source or _default_source(device)
    if source not in _PULL_SOURCES:
        raise ValueError(f"unknown source {source!r}; choose one of {', '.join(_PULL_SOURCES)}")
    module = importlib.import_module(_PULL_SOURCES[source])
    profile = module.pull(device, at=at)
    if output is not None:
        _write_atomically(Path(output), profile.save)
        return Pulled(profile, Path(output), written=True)
    listed = vault_profiles()
    held = [i for i in listed if i.id == profile.id]
    for info in held:
        if info.fingerprint == profile.fingerprint:
            return Pulled(profile, Path(str(info.path)), written=False)
    same_time = [i for i in held if i.calibrated_at == profile.device.calibrated_at]
    if same_time:
        old = same_time[0]
        path = Path(str(old.path))  # may carry an older naming scheme; replace it in place
        warnings.warn(
            f"replaced {path.name} (nv:{old.fingerprint[:12]}) with this pull"
            f" ({profile.short_fingerprint}): the same calibration, converted differently;"
            " update any expect= pins",
            NoiseVaultWarning,
            stacklevel=3,
        )
    else:
        path = vault_path(profile)
        if path.exists():
            occupant = next((i.ref for i in listed if i.path == path), "an unreadable profile")
            raise FileExistsError(
                f"{path} already holds {occupant}, another calibration; move that file out of"
                f" {path.parent} and pull again"
            )
    _write_atomically(path, profile.save)
    return Pulled(profile, path, written=True)


_DEFAULT_SOURCES = {"ibm_": "ibm", "ionq": "ionq"}  # id prefix -> source that pulls it


def _default_source(device: str) -> str:
    for prefix, source in _DEFAULT_SOURCES.items():
        if device.lower().startswith(prefix):
            return source
    bundled = any(i.id == device for i in bundled_profiles())
    hint = f"; {device} is bundled, so nv.load({device!r}) loads it offline" if bundled else ""
    raise SourceUnavailable(
        f"no live source pulls {device!r}: pull reads IBM devices (ibm_..., source='ibm' or"
        f" 'ibm-account') and IonQ devices (ionq..., source='ionq'){hint}"
    )


def _write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """A failed write leaves the old file whole; the dot name keeps listings from seeing it.

    The temporary file ends in ``path``'s suffix, so it is written in the same format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=f".tmp{path.suffix}", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        if path.exists():
            shutil.copymode(path, tmp)
        else:  # NamedTemporaryFile creates 0600; a new profile should get the usual mode
            mask = os.umask(0)
            os.umask(mask)
            tmp.chmod(0o666 & ~mask)
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
