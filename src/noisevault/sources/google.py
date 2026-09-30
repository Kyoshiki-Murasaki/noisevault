"""Google devices through cirq_google's calibration-to-noise conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, Quantinuum, Google, IonQ and Braket sources (L1b)"


def bundled_profiles() -> list[Profile]:
    raise NotImplementedError(f"bundled_profiles is not available yet: {_PENDING}")
