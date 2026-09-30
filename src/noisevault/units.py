"""Time-unit aliases accepted in hand-written profiles, normalized before validation.

Saved profiles only use the canonical spellings: ``duration_ns``, ``t1_us``, ``t2_us``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_NS_PER = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}


@dataclass(frozen=True)
class TimeField:
    stem: str
    unit: str
    strictly_positive: bool

    @property
    def canonical(self) -> str:
        return f"{self.stem}_{self.unit}"

    def spellings(self) -> tuple[str, ...]:
        return tuple(f"{self.stem}_{unit}" for unit in _NS_PER)


DURATION = TimeField("duration", "ns", strictly_positive=False)
T1 = TimeField("t1", "us", strictly_positive=True)
T2 = TimeField("t2", "us", strictly_positive=True)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert with a single rounding step: the unit ratio is an exact power of ten."""
    src, dst = _NS_PER[from_unit], _NS_PER[to_unit]
    return value * (src // dst) if src >= dst else value / (dst // src)


def normalize_times(data: Any, fields: tuple[TimeField, ...]) -> Any:
    """Return ``data`` with every alias of ``fields`` rewritten to its canonical key.

    Raises ValueError for two spellings of one field, a non-numeric or nonfinite value, a
    negative duration, or a nonpositive coherence time. Non-dict input passes through so the
    model validator reports it.
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for field in fields:
        for key in field.spellings():
            if key in out and out[key] is None:
                del out[key]
        given = [key for key in field.spellings() if key in out]
        if not given:
            continue
        if len(given) > 1:
            raise ValueError(f"{field.stem} is given twice ({' and '.join(given)}); keep one")
        key = given[0]
        value = out.pop(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value!r}")
        if field.strictly_positive and value <= 0:
            raise ValueError(f"{key} must be positive, got {value!r}")
        if value < 0:
            raise ValueError(f"{key} must not be negative, got {value!r}")
        out[field.canonical] = convert(float(value), key.rsplit("_", 1)[1], field.unit)
    return out
