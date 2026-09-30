"""Conversion checks: each framework export against the reference on small circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profile import Profile

_PENDING = "implemented in wave 3, diff, check and CLI (L5)"


def check(profile: Profile, **options: Any) -> Any:
    raise NotImplementedError(f"check is not available yet: {_PENDING}")
