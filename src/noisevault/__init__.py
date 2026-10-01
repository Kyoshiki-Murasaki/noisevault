"""NoiseVault: real device noise, pinned and portable.

Importing this package loads no quantum framework; each export imports its framework on use.
"""

from __future__ import annotations

__version__ = "0.2.0"

import importlib
from pathlib import Path
from typing import Any

from .catalog import ProfileInfo, load, profiles, pull
from .errors import (
    AmbiguousRef,
    DisabledGateError,
    FingerprintMismatch,
    LayoutError,
    MigrationWarning,
    MissingCalibrationError,
    NoiseApproximationWarning,
    NoiseVaultError,
    NoiseVaultWarning,
    ProfileNotFound,
    SourceUnavailable,
    UnsupportedEffect,
)
from .profile import Profile, json_schema
from .report import Report

_LAZY_MODULES = {"stim": "noisevault.frameworks.stim"}


def from_qiskit_backend(backend: Any) -> Profile:
    """A profile from a Qiskit BackendV2 (its Target), e.g. a qiskit-ibm-runtime fake backend."""
    from .sources.qiskit_backend import from_qiskit_backend as convert

    return convert(backend)


def from_ibm_csv(path: str | Path, *, device: str, calibrated_at: Any) -> Profile:
    """A profile from an IBM Quantum calibration CSV download."""
    from .sources.ibm_csv import from_ibm_csv as convert

    return convert(path, device=device, calibrated_at=calibrated_at)


def from_braket(path_or_dict: str | Path | dict[str, Any], *, device: str | None = None) -> Profile:
    """A profile from saved Amazon Braket standardized device properties.

    Braket properties do not name the device, so pass ``device`` (e.g. ``"garnet"``) to name the
    profile; by default it is the file name without its suffix.
    """
    from .sources.braket import from_braket as convert

    return convert(path_or_dict, device=device)


def from_cirq_google(processor_id: str) -> Profile:
    """A profile from a calibration cirq_google ships (rainbow, weber, willow_pink)."""
    from .sources.google import from_cirq_google as convert

    return convert(processor_id)


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module 'noisevault' has no attribute {name!r}")


__all__ = [
    "AmbiguousRef",
    "DisabledGateError",
    "FingerprintMismatch",
    "LayoutError",
    "MigrationWarning",
    "MissingCalibrationError",
    "NoiseApproximationWarning",
    "NoiseVaultError",
    "NoiseVaultWarning",
    "Profile",
    "ProfileInfo",
    "ProfileNotFound",
    "Report",
    "SourceUnavailable",
    "UnsupportedEffect",
    "__version__",
    "from_braket",
    "from_cirq_google",
    "from_ibm_csv",
    "from_qiskit_backend",
    "json_schema",
    "load",
    "profiles",
    "pull",
]
