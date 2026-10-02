"""IonQ's public characterization endpoint (API v0.4): dated device medians, no account needed.

IonQ publishes one record per characterization with device-wide medians only: 1Q, 2Q and SPAM
fidelity, gate and readout times, T1 and T2. Every qubit and pair of the profile gets those
medians, and each interpretation of an unstated metric is written into the profile. IonQ's
EULA restricts redistribution, so nothing is bundled: profiles are pulled onto the user's
machine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from .. import __version__
from ..errors import SourceUnavailable, did_you_mean
from ..profile import Profile
from . import OFFLINE_HINT

API = "https://api.ionq.co/v0.4"
_TIMEOUT_S = 30.0
_PAGE = 10  # the largest page the endpoint serves
_MAX_PROBES = 30  # one-record reads per pull before giving up
# IonQ's history holds medians no trapped-ion device produces (1Q 0.69, 2Q 0.75, SPAM 0.73)
# next to normal ones, and chance-level SPAM placeholders (0.5, 0.501) from before SPAM was
# measured. Medians below these floors are read as corrupt, not as data.
_FLOORS = {"1q": 0.99, "2q": 0.9, "spam": 0.9}
_NO_FIDELITIES = "without 1Q/2Q fidelities"
_IMPLAUSIBLE = "with implausible medians on "
# IonQ native two-qubit gate -> canonical name; IonQ's zz takes any angle, like rzz.
_TWO_QUBIT = {"zz": "rzz", "ms": "ms"}
_FIDELITY_ASSUMPTION = "IonQ does not state the fidelity metric; read as average gate fidelity"


def bundled_profiles() -> list[Profile]:
    """None: IonQ's EULA restricts redistribution, so use :func:`pull` instead."""
    return []


def pull(device: str, *, at: str | date | datetime | None = None) -> Profile:
    """The newest usable characterization of ``device`` at or before ``at`` (default: now).

    ``device`` may be ``ionq_forte-1``, ``forte-1`` or ``qpu.forte-1``. ``at`` is a datetime,
    a date or an ISO 8601 string; a date or a naive time is read as UTC. Records without 1Q and
    2Q fidelities (IonQ backfills some) or with implausible ones (see :data:`_FLOORS`) are
    skipped for the next older one, and the reason is written to the provenance notes.
    """
    backend = backend_name(device)
    listing = _listing(backend)
    record, raw, url, skipped = _newest_usable(backend, at)
    return to_profile(
        record,
        listing,
        source_url=url,
        source_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
        skipped=skipped,
    )


def backend_name(device: str) -> str:
    """``ionq_forte-1``, ``forte-1`` or ``qpu.forte-1`` -> ``qpu.forte-1``."""
    name = device.strip().lower().removeprefix("ionq_").removeprefix("ionq.")
    return name if name.startswith("qpu.") else f"qpu.{name}"


def to_profile(
    record: Mapping[str, Any],
    listing: Mapping[str, Any],
    *,
    source_url: str,
    source_hash: str,
    skipped: Sequence[str] = (),
) -> Profile:
    """A profile from one characterization record and the backend's entry in the listing.

    ``skipped`` holds why each newer record was passed over.
    """
    backend = record["backend"]
    name = backend.removeprefix("qpu.")
    num_qubits = int(record.get("qubits") or listing["qubits"])
    fidelity = record.get("fidelity") or {}
    timing = record.get("timing") or {}
    reason = _rejection(record)
    if reason is not None:
        raise SourceUnavailable(
            f"IonQ published {backend} record {record.get('id')} {reason}",
            hint="use a record from another date",
        )
    one, two = _fidelity(fidelity, "1q"), _fidelity(fidelity, "2q")
    assert one is not None and two is not None  # _rejection() checked both
    natives = [g for g in listing.get("supported_native_gates", ()) if g in _TWO_QUBIT]
    if not natives:
        raise SourceUnavailable(
            f"IonQ lists no known two-qubit native gate for {backend}"
            f" ({listing.get('supported_native_gates')}); expected one of {', '.join(_TWO_QUBIT)}"
        )
    two_qubit = _TWO_QUBIT[natives[0]]
    two_assumption = _FIDELITY_ASSUMPTION
    if two_qubit == "rzz":
        two_assumption += "; IonQ's native zz gate takes any angle, and this error is used for all"
    notes = [
        "IonQ publishes device medians only: every qubit and pair gets the same values",
        "SPAM fidelity is combined state preparation and measurement; 1 - fidelity is used as"
        " a symmetric readout error and prep is left unknown",
        "t1, t2 and gate times are device-level values IonQ gives with no measurement"
        " statement (they are the same on every record of a backend), so read them as nominal",
        "a stderr of 0, as IonQ reports on every record, is a placeholder and is dropped",
        f"the two-qubit native ({natives[0]} -> {two_qubit}) comes from IonQ's current backend"
        " listing, also for older records",
    ]
    if not record.get("qubits"):
        notes.append(
            "the record gives no qubit count, so the current listing's qubit count"
            f" ({num_qubits}) is used"
        )
    notes += _skip_notes(skipped)
    spam = _fidelity(fidelity, "spam")
    readout: dict[str, float] | None = None
    if spam is not None:
        readout = {"error": _clean(1 - spam[0]), **_ns("duration_ns", timing.get("readout"))}
    elif _median(fidelity, "spam") is not None:
        notes.append(
            f"SPAM fidelity median {_median(fidelity, 'spam')} is below {_FLOORS['spam']}"
            " (a chance-level placeholder or corrupt), so readout error is left unknown"
        )
    idle = {
        key: _clean(seconds * 1e6)
        for key, seconds in (("t1_us", timing.get("t1")), ("t2_us", timing.get("t2")))
        if _positive(seconds)
    }
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "ionq",
                "name": name,
                "technology": "trapped_ion",
                "num_qubits": num_qubits,
                "processor": name.split("-")[0].capitalize(),
                "calibrated_at": record["date"],
            },
            "connectivity": _connectivity(record.get("connectivity"), num_qubits),
            "gates": {
                "rz": {"virtual": True},
                "r": _gate(one, timing.get("1q"), _FIDELITY_ASSUMPTION),
                two_qubit: _gate(two, timing.get("2q"), two_assumption),
            },
            "readout": readout,
            "idle": idle or None,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "public_api",
                "source": f"IonQ characterization API v0.4, {backend} record {record['id']}",
                "source_url": source_url,
                "license": "IonQ EULA (not an open license)",
                "attribution": "IonQ",
                "redistributable": "no",
                "retrieved_at": datetime.now(UTC),
                "source_hash": source_hash,
                "tool": f"noisevault {__version__}",
                "notes": notes,
            },
        }
    )


def _gate(fidelity: tuple[float, float | None], seconds: Any, assumption: str) -> dict[str, Any]:
    value, stderr = fidelity
    return {
        "avg_infidelity": _clean(1 - value),
        "stderr": stderr,
        **_ns("duration_ns", seconds),
        "method": "vendor",
        "statistic": "median",
        "assumption": assumption,
    }


def _rejection(record: Mapping[str, Any]) -> str | None:
    """Why ``record`` cannot pin a profile, or None when its 1Q and 2Q medians are usable."""
    fidelity = record.get("fidelity") or {}
    if _median(fidelity, "1q") is None or _median(fidelity, "2q") is None:
        return _NO_FIDELITIES
    one, two = _fidelity(fidelity, "1q"), _fidelity(fidelity, "2q")
    if one is not None and two is not None and one[0] >= two[0]:
        return None
    medians = f"1Q {_median(fidelity, '1q')}, 2Q {_median(fidelity, '2q')}"
    return f"{_IMPLAUSIBLE}{str(record.get('date'))[:10]} ({medians})"


def _median(fidelity: Mapping[str, Any], key: str) -> float | None:
    """The median IonQ gives for ``key``, or None for its null."""
    entry = fidelity.get(key)
    median = entry.get("median") if isinstance(entry, Mapping) else None
    ok = isinstance(median, int | float) and not isinstance(median, bool) and 0 < median <= 1
    return float(median) if ok else None


def _fidelity(fidelity: Mapping[str, Any], key: str) -> tuple[float, float | None] | None:
    """(median, stderr) when the median is plausible data; nulls and placeholders are None."""
    median = _median(fidelity, key)
    if median is None or median < _FLOORS[key]:
        return None
    stderr = fidelity[key].get("stderr")
    return median, float(stderr) if _positive(stderr) else None


def _skip_notes(skipped: Sequence[str]) -> list[str]:
    missing = sum(reason == _NO_FIDELITIES for reason in skipped)
    implausible = [reason for reason in skipped if reason != _NO_FIDELITIES]
    notes = [f"skipped {missing} newer record(s) {_NO_FIDELITIES}"] if missing else []
    if implausible:
        dated = [reason.removeprefix(_IMPLAUSIBLE) for reason in implausible]
        shown = "; ".join(dated[:3]) + ("; ..." if len(dated) > 3 else "")
        notes.append(
            f"skipped {len(implausible)} newer record(s) with implausible fidelity medians"
            f" (1Q below {_FLOORS['1q']}, 2Q below {_FLOORS['2q']}, or a 1Q error above the 2Q"
            f" error), read as corrupt: {shown}"
        )
    return notes


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _ns(key: str, seconds: Any) -> dict[str, float]:
    return {key: _clean(seconds * 1e9)} if _positive(seconds) else {}


def _clean(value: float) -> float:
    """Drop binary round-off from decimal inputs: 1 - 0.9998 -> 0.0002, 0.00013 s -> 130000 ns."""
    return float(f"{value:.12g}")


def _connectivity(pairs: Any, num_qubits: int) -> str | dict[str, Any]:
    if not pairs:
        return "all_to_all"
    edges = {tuple(sorted(pair)) for pair in pairs}
    if edges == set(itertools.combinations(range(num_qubits), 2)):
        return "all_to_all"
    return {"edges": sorted(edges), "directed": False}


# network ------------------------------------------------------------------------------------


def _listing(backend: str) -> Mapping[str, Any]:
    entries = json.loads(_get(f"{API}/backends"))
    for entry in entries:
        if entry.get("backend") == backend:
            return entry
    qpus = sorted(e["backend"] for e in entries if str(e.get("backend", "")).startswith("qpu."))
    raise SourceUnavailable(
        f"IonQ has no backend {backend!r}; {did_you_mean(backend, qpus)}it lists {', '.join(qpus)}"
    )


def _newest_usable(
    backend: str, at: str | date | datetime | None
) -> tuple[dict[str, Any], bytes, str, list[str]]:
    """The newest record at or before ``at`` with plausible 1Q and 2Q fidelities of its own.

    Returns the record, the bytes and URL of the one-record page it was read from, and the
    :func:`_rejection` of each newer record skipped. A page of several records fills a record's
    null fields from the next newer record on the page (seen on qpu.aria-1), so a record is
    judged on the page that holds it alone, which is also the page cited.
    """
    end = None if at is None else _utc(at).isoformat().replace("+00:00", "Z")
    skipped: list[str] = []
    probes = 0
    page = 1
    while True:
        body = json.loads(_get(_page_url(backend, limit=_PAGE, end=end, page=page)))
        records = body.get("characterizations") or []
        for listed in records:
            # a record rejected on the filled page is rejected alone too, so skip it unprobed
            reason = _rejection(listed)
            if reason is None and probes < _MAX_PROBES:
                probes += 1
                url = _page_url(backend, limit=1, end=listed["date"])
                raw = _get(url)
                alone = (json.loads(raw).get("characterizations") or [{}])[0]
                reason = _rejection(alone) if alone.get("id") == listed["id"] else _NO_FIDELITIES
                if reason is None:
                    return alone, raw, url, skipped
            skipped.append(reason or _NO_FIDELITIES)
        if probes >= _MAX_PROBES:
            raise SourceUnavailable(
                f"none of the {len(skipped)} newest {backend} characterizations{_before(at)} has"
                " plausible 1Q and 2Q fidelities of its own",
                hint="pass an earlier at= to search older records",
            )
        if not records or page >= int(body.get("pages") or 0):
            raise SourceUnavailable(
                f"IonQ publishes no {backend} characterization{_before(at)} with plausible 1Q and"
                f" 2Q fidelities of its own ({len(skipped)} record(s) checked)",
                hint="use another backend" if at is None else "pass a later at= or none",
            )
        page += 1


def _before(at: str | date | datetime | None) -> str:
    return "" if at is None else f" at or before {at}"


def _page_url(backend: str, *, limit: int, end: str | None = None, page: int = 1) -> str:
    query = {
        "limit": limit,
        **({"end": end} if end else {}),
        **({"page": page} if page > 1 else {}),
    }
    return f"{API}/backends/{backend}/characterizations?{urllib.parse.urlencode(query)}"


def _utc(at: str | date | datetime) -> datetime:
    if isinstance(at, str):
        at = datetime.fromisoformat(at.strip().replace("Z", "+00:00"))
    if not isinstance(at, datetime):
        at = datetime(at.year, at.month, at.day)
    return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": f"noisevault/{__version__}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SourceUnavailable(
            f"IonQ's API answered HTTP {exc.code} for {url}", hint="try again later"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceUnavailable(
            f"could not reach IonQ's API ({reason})", hint=OFFLINE_HINT
        ) from None
