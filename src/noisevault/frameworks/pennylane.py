"""PennyLane export: a NoiseModel carrying a report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, PennyLane export (L3b)"


def to_pennylane(profile: Profile, **options: Any) -> Any:
    raise NotImplementedError(f"to_pennylane is not available yet: {_PENDING}")
