"""Qiskit export: a noise-aware AerSimulator with a Target and a report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profile import Profile

_PENDING = "implemented in wave 2, Qiskit export (L2)"


def to_qiskit(profile: Profile, **options: Any) -> Any:
    raise NotImplementedError(f"to_qiskit is not available yet: {_PENDING}")
