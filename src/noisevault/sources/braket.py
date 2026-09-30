"""Amazon Braket standardized device properties saved by the user."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, Quantinuum, Google, IonQ and Braket sources (L1b)"


def from_braket(path_or_dict: Any) -> Profile:
    raise NotImplementedError(f"from_braket is not available yet: {_PENDING}")


def bundled_profiles() -> list[Profile]:
    raise NotImplementedError(f"bundled_profiles is not available yet: {_PENDING}")
