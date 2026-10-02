"""IBM Quantum's public calibration endpoint: current and past calibrations, no account needed."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, TypeAdapter, ValidationError

from .. import __version__
from ..errors import SourceUnavailable, did_you_mean, parse_json
from ..profile import Profile
from . import OFFLINE_HINT
from .qiskit_backend import (
    as_utc,
    calibration_from_properties,
    now_utc,
    processor_name,
    sha256_bytes,
    to_profile,
)

BASE_URL = "https://quantum.cloud.ibm.com/api/v1/public/backends"
TIMEOUT_S = 30.0
_TRY_LATER = "try again later, or pull through your IBM account with source='ibm-account'"


class _Parameter(BaseModel):
    name: str
    value: float | None = None
    unit: str | None = None


class _Gate(BaseModel):
    gate: str
    qubits: list[int]
    parameters: list[_Parameter] | None = None


class _Properties(BaseModel):
    """The keys calibration_from_properties needs from a properties reply, with their types."""

    qubits: list[list[_Parameter]] | None = None
    gates: list[_Gate] | None = None
    last_update_date: Annotated[str, AfterValidator(as_utc)] | None = None


class _Device(BaseModel):
    name: str


class _ProcessorType(BaseModel):
    family: str | None = None


class _Configuration(BaseModel):
    processor_type: _ProcessorType | None = None


_PROPERTIES = TypeAdapter(_Properties)
_LISTING = TypeAdapter(list[_Device])
_CONFIGURATION = TypeAdapter(_Configuration)


def pull(device: str, *, at: str | date | datetime | None = None) -> Profile:
    """The calibration of ``device`` now, or the newest one older than ``at``.

    ``at`` is a datetime or ISO 8601 string; a date or naive time is read as UTC.
    """
    name = device.strip().lower()
    url = properties_url(name, at)
    try:
        raw = fetch(url)
    except _NotFound:
        raise _not_found(name, at) from None
    props = _json(raw, url, _PROPERTIES)
    if not props.get("qubits"):
        raise SourceUnavailable(f"IBM's public endpoint returned no qubit data for {name} ({url})")
    cal = replace(calibration_from_properties(props), name=name, processor=_processor(name))
    return to_profile(
        cal,
        {
            "data_kind": "measured",
            "source_kind": "public_api",
            "source": "IBM Quantum public calibration endpoint",
            "source_url": url,
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "retrieved_at": now_utc(),
            "source_hash": sha256_bytes(raw),
        },
    )


def properties_url(name: str, at: str | date | datetime | None = None) -> str:
    url = f"{BASE_URL}/{urllib.parse.quote(name)}/properties"
    if at is None:
        return url
    stamp = as_utc(at).isoformat().replace("+00:00", "Z")
    return f"{url}?{urllib.parse.urlencode({'updated_before': stamp})}"


class _NotFound(Exception):
    pass


def fetch(url: str) -> bytes:
    """GET ``url``: its body, _NotFound on HTTP 404, SourceUnavailable on any other failure."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": f"noisevault/{__version__}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise _NotFound(url) from None
        raise SourceUnavailable(
            f"IBM's public endpoint answered HTTP {exc.code} for {url}", hint=_TRY_LATER
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceUnavailable(
            f"could not reach IBM's public endpoint ({reason})", hint=OFFLINE_HINT
        ) from None


def _json(raw: bytes, url: str, shape: TypeAdapter[Any]) -> Any:
    try:
        data = parse_json(raw)
    except ValueError:
        raise SourceUnavailable(
            f"IBM's public endpoint answered {url} with something other than JSON",
            hint=_TRY_LATER,
        ) from None
    try:
        shape.validate_python(data, strict=True)
    except ValidationError as exc:
        raise SourceUnavailable(
            f"IBM's public endpoint answered {url} with JSON of the wrong shape{_where(exc)}",
            hint=_TRY_LATER,
        ) from None
    return data


def _where(exc: ValidationError) -> str:
    """`` at gates[0].qubits``, where a reply first leaves its shape; '' for all of it."""
    loc = exc.errors()[0]["loc"]
    path = "".join(f"[{key}]" if isinstance(key, int) else f".{key}" for key in loc)
    return f" at {path.removeprefix('.')}" if path else ""


def listed_devices() -> list[str]:
    """Names of the devices the public endpoint lists right now."""
    return sorted(entry["name"] for entry in _json(fetch(BASE_URL), BASE_URL, _LISTING))


def _processor(name: str) -> str | None:
    """The processor type from the public configuration; None when it cannot be read."""
    url = f"{BASE_URL}/{urllib.parse.quote(name)}/configuration"
    try:
        config = _json(fetch(url), url, _CONFIGURATION)
    except (_NotFound, SourceUnavailable):
        return None
    return processor_name(config.get("processor_type"))


def _not_found(name: str, at: str | date | datetime | None) -> SourceUnavailable:
    try:
        listed = listed_devices()
    except (_NotFound, SourceUnavailable):
        listed = None
    if at is not None and listed is not None and name in listed:
        return SourceUnavailable(
            f"IBM's public endpoint has no {name} calibration older than {at}",
            hint="pick a later date, or pull through your IBM account with source='ibm-account'",
        )
    from ..catalog import bundled_profiles as bundled

    known = f" (it lists {', '.join(listed)})" if listed else ""
    guess = did_you_mean(name, listed or [])
    if any(info.id == name for info in bundled()):
        hint = f"{name} is bundled, so nv.load({name!r}) loads it offline"
    else:
        hint = "if your IBM account can see it, pull with source='ibm-account'"
    retired = "if not, it may be retired" if guess else "it may be retired"
    return SourceUnavailable(
        f"{name} is not listed on the public endpoint{known}; {guess}{retired}", hint=hint
    )


def bundled_profiles() -> list[Profile]:
    """Live pulls are never bundled: IBM's terms for this data are not an open license."""
    return []
