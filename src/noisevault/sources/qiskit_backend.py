"""Profiles from any Qiskit BackendV2 (its Target) and the bundled IBM fake backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, IBM sources (L1a)"


def from_qiskit_backend(backend: Any) -> Profile:
    raise NotImplementedError(f"from_qiskit_backend is not available yet: {_PENDING}")


def bundled_profiles() -> list[Profile]:
    raise NotImplementedError(f"bundled_profiles is not available yet: {_PENDING}")
