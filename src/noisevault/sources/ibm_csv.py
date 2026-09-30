"""IBM calibration CSV files as downloaded from the IBM Quantum platform."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, IBM sources (L1a)"


def from_ibm_csv(path: Any, *, device: str, calibrated_at: Any) -> Profile:
    raise NotImplementedError(f"from_ibm_csv is not available yet: {_PENDING}")


def bundled_profiles() -> list[Profile]:
    raise NotImplementedError(f"bundled_profiles is not available yet: {_PENDING}")
