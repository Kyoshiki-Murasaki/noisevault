"""IBM Quantum public calibration endpoint (no account needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, IBM sources (L1a)"


def pull(device: str, *, at: Any = None) -> Profile:
    raise NotImplementedError(f"pull is not available yet: {_PENDING}")


def bundled_profiles() -> list[Profile]:
    raise NotImplementedError(f"bundled_profiles is not available yet: {_PENDING}")
