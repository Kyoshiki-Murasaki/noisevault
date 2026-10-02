"""Exceptions and warnings raised by NoiseVault."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class NoiseVaultError(Exception):
    """Base class for every expected NoiseVault failure.

    ``hint`` is the next step, if there is one. ``str()`` gives the message and then the hint,
    and ``message`` gives the message alone.
    """

    def __init__(self, *args: object, hint: str | None = None) -> None:
        super().__init__(*args)
        self.hint = hint

    @property
    def message(self) -> str:
        return super().__str__()

    def __str__(self) -> str:
        return f"{self.message}; {self.hint}" if self.hint else self.message


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


class SourceDataError(NoiseVaultError, ValueError):
    """A source importer cannot read the calibration data it was given."""


def did_you_mean(given: str, choices: Iterable[str]) -> str:
    """``did you mean 'X'? `` for the choice closest to a mistyped value; '' when none is close."""
    close = difflib.get_close_matches(given, list(choices), n=1)
    return f"did you mean '{close[0]}'? " if close else ""


class NoiseVaultWarning(UserWarning):
    """Base class for NoiseVault warnings."""


class NoiseApproximationWarning(NoiseVaultWarning):
    """A conversion used an approximation the caller should know about."""


class MigrationWarning(NoiseVaultWarning):
    """A file in an older format was upgraded in memory."""


REPOSITORY = "https://github.com/Kyoshiki-Murasaki/noisevault"


def install_hint(extra: str) -> str:
    """The pip command that adds an optional extra, e.g. ``install_hint("cirq")``."""
    return f'pip install "noisevault[{extra}] @ git+{REPOSITORY}"'
