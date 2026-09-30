"""Differences between two profiles of the same device."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profile import Profile

_PENDING = "implemented in wave 3, diff, check and CLI (L5)"


def diff(a: Profile, b: Profile, **options: Any) -> Any:
    raise NotImplementedError(f"diff is not available yet: {_PENDING}")
