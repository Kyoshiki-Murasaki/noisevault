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
from datetime import date, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal, NamedTuple

from pydantic import ValidationError

from .errors import (
    AmbiguousRef,
    FingerprintMismatch,
    NoiseVaultWarning,
    ProfileNotFound,
    SourceUnavailable,
    did_you_mean,
)
from .profile import Profile, Ref, load_bytes, load_file, parse_ref, write_atomically
from .sources.qiskit_backend import as_utc

_PULL_SOURCES = {
    "ibm": "noisevault.sources.ibm_public",
    "ibm-account": "noisevault.sources.ibm_account",
    "ionq": "noisevault.sources.ionq",
}

# The ProfileInfo fields an index entry carries; whatever else it holds is ignored.
_ENTRY_TYPES: dict[str, Any] = {
    "id": str,
    "calibrated_at": str | None,
    "technology": str,
    "vendor": str | None,
    "num_qubits": int,
    "fingerprint": str,
    "data_kind": str,
    "license": str | None,
    "processor": str | None,
    "source_kind": str | None,
    "redistributable": str,
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
    processor: str | None = None
    source_kind: str | None = None
    redistributable: str = "unknown"

    @property
    def ref(self) -> str:
        return _ref(self.id, self.calibrated_at)

    def load(self) -> Profile:
        return load_bytes(self.path.read_bytes())

    @classmethod
    def of(cls, profile: Profile, location: Literal["vault", "bundled"], path: Path) -> ProfileInfo:
        return cls.from_entry(index_entry(profile), location, path)

    @classmethod
    def from_entry(
        cls, entry: dict[str, Any], location: Literal["vault", "bundled"], path: Path | Traversable
    ) -> ProfileInfo:
        """From a line of a catalog index (see :func:`index_entry`); ValueError if it is damaged."""
        fields = {
            "calibrated_at": None,
            "vendor": None,
            "data_kind": "unknown",
            "license": None,
            "processor": None,
            "source_kind": None,
        }
        fields |= {key: entry[key] for key in _ENTRY_TYPES if key in entry}
        for key, kind in _ENTRY_TYPES.items():
            value = fields.get(key)
            if isinstance(value, bool) or not isinstance(value, kind):
                raise ValueError(f"index entry {key} is {value!r}")
        stamp = fields.pop("calibrated_at")
        when = datetime.fromisoformat(stamp) if stamp else None
        if when is not None and when.tzinfo is None:
            raise ValueError(f"index entry time {stamp} has no time zone")
        if not re.fullmatch(r"[0-9a-f]{64}", fields["fingerprint"]):
            raise ValueError(f"index entry fingerprint {fields['fingerprint']!r} is not sha256 hex")
        return cls(**fields, calibrated_at=when, location=location, path=path)


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
        "source_kind": prov.source_kind,
        "license": prov.license,
        "redistributable": prov.redistributable,
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
    whenever a file's size, modification time or change time (moved by chmod and chown, which
    can make it unreadable) changes, and losing it costs one re-read.
    """
    folder = vault_dir()
    cached = _read_vault_index(folder)
    index: dict[str, dict[str, Any]] = {}
    out = []
    for path in sorted(folder.glob("*.json*")):
        if path.name.startswith("."):  # the index cache and in-flight temporary files
            continue
        try:
            entry, info = _vault_entry(path, cached.get(path.name))
        except Exception as exc:  # one damaged or odd entry must not hide the rest
            warnings.warn(_skipped(path, exc), NoiseVaultWarning, stacklevel=2)
            continue
        index[path.name] = entry
        out.append(info)
    if index != cached:
        _write_vault_index(folder, index)
    return out


def _vault_entry(path: Path, cached: Any) -> tuple[dict[str, Any], ProfileInfo]:
    """The index entry and listing of one vault file, from the cache while the file is unchanged."""
    stat = path.stat()
    if not S_ISREG(stat.st_mode):  # opening a pipe would block the listing
        raise OSError("not a regular file")
    signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if isinstance(cached, dict) and cached.get("signature") == signature:
        try:
            return cached, ProfileInfo.from_entry(cached, "vault", path)
        except ValueError:
            pass
    entry = {"signature": signature, **index_entry(load_file(path))}
    return entry, ProfileInfo.from_entry(entry, "vault", path)


def _skipped(path: Path, exc: Exception) -> str:
    """One line naming a vault entry that cannot be listed, and why."""
    try:
        dangling = path.readlink() if path.is_symlink() and not path.exists() else None
    except OSError:  # an unreachable target fails these checks as it failed the read
        dangling = None
    if dangling is not None:
        why = f"it links to {dangling}, which does not exist; remove the link"
    elif isinstance(exc, ValidationError):
        n = exc.error_count()
        why = f"not a valid profile ({n} problem{'s' * (n != 1)}); run nv validate {path}"
    elif isinstance(exc, OSError):
        why = exc.strerror or str(exc)
    else:
        why = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"skipped {path}: {why}"


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
        write_atomically(folder / _VAULT_INDEX, text.encode("utf-8"))
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
        parsed = target(ref)
        if isinstance(parsed, Path):
            raise ValueError(f"{ref!r} is a path, not a catalog ref")
        ref = parsed
    known = _dedupe(vault_profiles() + bundled_profiles())
    candidates = [info for info in known if info.id == ref.id]
    if not candidates:
        close = difflib.get_close_matches(ref.id, sorted({i.id for i in known}), n=3)
        if close:
            raise ProfileNotFound(
                f"no profile with id {ref.id!r}; did you mean {', '.join(close)}?"
            )
        raise ProfileNotFound(
            f"no profile with id {ref.id!r}",
            hint="run nv list to see every profile you can load offline",
        )
    if ref.timestamp is not None:
        matches = [i for i in candidates if i.calibrated_at == ref.timestamp]
    elif ref.date is not None:
        matches = [i for i in candidates if i.calibrated_at and i.calibrated_at.date() == ref.date]
    else:
        newest = max(_epoch(i.calibrated_at) for i in candidates)
        matches = [i for i in candidates if _epoch(i.calibrated_at) == newest]
    if not matches:
        raise _no_calibration(ref, candidates)
    if expect is not None:
        want = _expect_prefix(expect)
        pinned = [i for i in matches if i.fingerprint.startswith(want)]
        if not pinned:
            raise _mismatch(_loaded(ref, _unshadowed(matches)), expect, ref.id, known)
        matches = pinned
    matches = _unshadowed(matches)
    if len(matches) > 1:
        listing = "; ".join(f"{i.ref} ({i.location}, nv:{i.fingerprint[:12]})" for i in matches)
        if len({i.calibrated_at for i in matches}) > 1:
            raise AmbiguousRef(
                f"{ref.id} matches {len(matches)} profiles: {listing}", hint="use a full timestamp"
            )
        files = ", ".join(str(i.path) for i in matches)
        raise AmbiguousRef(
            f"{ref.id} matches {len(matches)} profiles that share a calibration time: {listing}",
            hint=f"pass expect='nv:...' or load one of their files: {files}",
        )
    return matches[0]


def target(ref: str | Path) -> Path | Ref:
    """What a ref names, as :func:`parse_ref` reads it, except that a bare id that also names a
    folder here is still the id."""
    parsed = parse_ref(ref)
    text = ref.strip() if isinstance(ref, str) else ""
    if isinstance(parsed, Path) and parsed.is_dir() and text == parsed.name and "@" not in text:
        return Ref(text.lower())
    return parsed


def load(ref: str | Path, *, expect: str | None = None) -> Profile:
    """Load a profile by path, ``id``, ``id@YYYY-MM-DD`` or ``id@<timestamp>``, offline.

    ``expect`` pins the fingerprint (full hex, ``sha256:<hex>`` or ``nv:<12 hex>``); a different
    profile raises FingerprintMismatch.
    """
    named = target(ref)
    if isinstance(named, Path):
        if not named.exists():
            raise ProfileNotFound(f"no file {named}")
        path, profile = named, load_file(named)
    else:
        info = resolve(named, expect=expect)
        path, profile = info.path, info.load()
    if expect is not None:
        _check_expect(profile, expect, path)
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
    source = source.strip().lower() if source else _default_source(device)
    if source not in _PULL_SOURCES:
        raise ValueError(_unknown_source(source))
    when = None if at is None else as_utc(at)
    module = importlib.import_module(_PULL_SOURCES[source])
    profile = module.pull(device, at=when)
    if output is not None:
        return Pulled(profile, profile.save(output), written=True)
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
            occupant = next((i.ref for i in listed if i.path == path), None)
            if occupant is None:  # skipped by the listing, with a warning saying why
                raise FileExistsError(
                    f"{path} exists but cannot be read; make it readable or move it out of"
                    f" {path.parent}, then pull again"
                )
            raise FileExistsError(
                f"{path} already holds {occupant}, another calibration; move that file out of"
                f" {path.parent} and pull again"
            )
    profile.save(path)
    return Pulled(profile, path, written=True)


_DEFAULT_SOURCES = {"ibm_": "ibm", "ionq": "ionq"}  # id prefix -> source that pulls it


def _pull_source(device: str) -> str | None:
    return next((s for p, s in _DEFAULT_SOURCES.items() if device.lower().startswith(p)), None)


def _default_source(device: str) -> str:
    source = _pull_source(device)
    if source is not None:
        return source
    bundled = any(i.id == device for i in bundled_profiles())
    raise SourceUnavailable(
        f"no live source pulls {device!r}: pull reads IBM devices (ibm_..., source='ibm' or"
        " 'ibm-account') and IonQ devices (ionq..., source='ionq')",
        hint=f"{device} is bundled, so nv.load({device!r}) loads it offline" if bundled else None,
    )


def _unknown_source(source: str) -> str:
    choices = ", ".join(_PULL_SOURCES)
    guess = did_you_mean(source, _PULL_SOURCES)
    offline = sorted({i.vendor for i in bundled_profiles() if i.vendor} - _PULL_SOURCES.keys())
    vendor = next((v for v in offline if did_you_mean(source, [v])), None)
    if vendor and not guess:
        return (
            f"unknown source {source!r}; no live source serves {vendor} devices; to see the"
            f" bundled ones, run nv list --vendor {vendor}"
        )
    return f"unknown source {source!r}; {guess}choose one of {choices}"


def _expect_prefix(expect: str) -> str:
    """The fingerprint hex an ``expect`` value pins: 64 digits, or 12 after ``nv:``."""
    want = expect.strip().lower().removeprefix("sha256:")
    short = want.startswith("nv:")
    want = want.removeprefix("nv:")
    if not re.fullmatch(r"[0-9a-f]{12}" if short else r"[0-9a-f]{64}", want):
        raise ValueError(f"expect={expect!r}: give a full sha256 fingerprint or nv:<12 hex>")
    return want


def _check_expect(profile: Profile, expect: str, path: Path | Traversable) -> None:
    if not profile.fingerprint.startswith(_expect_prefix(expect)):
        held = _ref(profile.id, profile.device.calibrated_at)
        loaded = f"{path} holds {held} ({profile.short_fingerprint})"
        raise _mismatch(loaded, expect, profile.id, profiles())


def _loaded(ref: Ref, held: list[ProfileInfo]) -> str:
    """The ref as given and the fingerprints it loads, with the newest ref when the id is bare."""
    said = f"{ref.id}@{ref.date}" if ref.date else _ref(ref.id, ref.timestamp)
    newest = said if ref.date or ref.timestamp else held[0].ref
    found = " or ".join(f"nv:{i.fingerprint[:12]}" for i in held)
    return f"{said} is {found}" if newest == said else f"{said} loads {newest} ({found})"


def _mismatch(
    loaded: str, expect: str, device: str, known: list[ProfileInfo]
) -> FingerprintMismatch:
    """The error for a profile that misses its pin. If you have a profile with that fingerprint,
    the hint is the nv.load call, with the same pin, that loads it. If not, the hint says how to
    get the file."""
    want = _expect_prefix(expect)
    missed = f"{loaded}, not the expected {expect}"
    match = next((i for i in known if i.fingerprint.startswith(want)), None)
    if match is not None:
        where = match.ref if match.calibrated_at else str(match.path)
        call = f"nv.load({where!r}, expect={expect!r})"
        return FingerprintMismatch(missed, hint=f"{call} loads the profile with that fingerprint")
    hint = "ask whoever pinned it for the profile file"
    if _pull_source(device):
        hint += (
            ", or, if the source still serves that calibration, run"
            f" nv pull {device} --at <a time it was in effect>"
        )
    return FingerprintMismatch(f"{missed}, and no profile you have has that fingerprint", hint=hint)


def _no_calibration(ref: Ref, candidates: list[ProfileInfo]) -> ProfileNotFound:
    """What a dated ref's date means when it names no profile, and what you have instead."""
    have = ", ".join(sorted({i.ref for i in candidates}))
    if ref.timestamp is not None:
        return ProfileNotFound(
            f"no {ref.id} profile calibrated at {_stamp(ref.timestamp)}; you have {have}"
        )
    fetch = (
        f"to fetch the calibration in effect at {ref.date}, run nv pull {ref.id} --at {ref.date}"
    )
    return ProfileNotFound(
        f"no {ref.id} profile calibrated on {ref.date} UTC; you have {have}",
        hint=fetch if _pull_source(ref.id) else None,
    )


def _unshadowed(infos: list[ProfileInfo]) -> list[ProfileInfo]:
    """Profiles of one id, without each bundled one that a vault profile of its time shadows."""
    shadowed = {_epoch(i.calibrated_at) for i in infos if i.location == "vault"}
    return [i for i in infos if i.location == "vault" or _epoch(i.calibrated_at) not in shadowed]


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


def _ref(device: str, when: datetime | None) -> str:
    return f"{device}@{_stamp(when)}" if when else device
