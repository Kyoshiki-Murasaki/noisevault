"""Stim export: a noisy copy of a Stim circuit carrying a report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, Stim export (L4)"


def to_stim(profile: Profile, circuit: Any, **options: Any) -> Any:
    raise NotImplementedError(f"to_stim is not available yet: {_PENDING}")


def layout_from_coords(circuit: Any, profile: Profile) -> dict[int, int]:
    raise NotImplementedError(f"layout_from_coords is not available yet: {_PENDING}")
