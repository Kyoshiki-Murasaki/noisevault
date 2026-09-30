"""Exceptions and warnings raised by NoiseVault."""

from __future__ import annotations


class NoiseVaultError(Exception):
    """Base class for every expected NoiseVault failure."""


class LayoutError(NoiseVaultError, ValueError):
    """A circuit-to-device qubit mapping is incomplete, not injective, or uses a bad qubit."""


class DisabledGateError(NoiseVaultError):
    """The profile marks this gate on these qubits as disabled."""


class MissingCalibrationError(NoiseVaultError):
    """No calibration exists for an operation and the caller asked for an error."""


class UnsupportedEffect(NoiseVaultError):
    """A profile effect demands a treatment the target framework cannot provide."""


class AmbiguousRef(NoiseVaultError, LookupError):
    """A profile reference matches more than one profile."""


class ProfileNotFound(NoiseVaultError, LookupError):
    """No profile matches a reference."""


class FingerprintMismatch(NoiseVaultError, ValueError):
    """A loaded profile does not carry the fingerprint the caller expected."""


class SourceUnavailable(NoiseVaultError):
    """A calibration source cannot be reached or is not installed."""


class NoiseVaultWarning(UserWarning):
    """Base class for NoiseVault warnings."""


class NoiseApproximationWarning(NoiseVaultWarning):
    """A conversion used an approximation the caller should know about."""


class MigrationWarning(NoiseVaultWarning):
    """A file in an older format was upgraded in memory."""
