from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any


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


class CountsError(NoiseVaultError, ValueError):
    """A counts file is malformed, or its run is one ``nv compare`` cannot score.

    The message is one line that starts with the file or the field, and ``hint`` says what to do.
    """


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except RecursionError:
        text = raw.decode(json.detect_encoding(raw), "surrogatepass")
    depth = deepest = at = 0
    for token in _STRING_OR_BRACKET.finditer(text):
        if token[0] in ("[", "{"):
            depth += 1
            if depth > deepest:
                deepest, at = depth, token.start()
        elif token[0] in ("]", "}"):
            depth -= 1
    raise json.JSONDecodeError(f"nested {deepest} levels deep", text, at)


_STRING_OR_BRACKET = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')


def unreadable(source: str, exc: Exception) -> str:
    if isinstance(exc, UnicodeDecodeError):
        line = exc.object.count(b"\n", 0, exc.start) + 1
        return f"{source} is not UTF-8 text (byte {exc.object[exc.start]:#04x} on line {line})"
    if isinstance(exc, json.JSONDecodeError):
        reason = exc.msg.removesuffix(" at")
        where = f"{reason[:1].lower()}{reason[1:]} at line {exc.lineno}, column {exc.colno}"
        return f"{source} is not JSON ({where})"
    if isinstance(exc, ValueError):
        return f"{source} is not JSON ({exc})"
    return f"{source} is a damaged gzip file ({exc})"


def did_you_mean(given: str, choices: Iterable[str]) -> str:
    close = difflib.get_close_matches(given, list(choices), n=1)
    return f"did you mean '{close[0]}'? " if close else ""


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def joined(words: Sequence[str], conjunction: str = "and") -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} {conjunction} {words[-1]}"


def qubit_loci(*loci: Sequence[int]) -> str:
    labels = ["-".join(map(str, locus)) for locus in loci]
    if len(labels) > 4:
        labels = [*labels[:3], f"{len(labels) - 3} more"]
    single = len(loci) == 1 and len(loci[0]) == 1
    return f"qubit {joined(labels)}" if single else f"qubits {joined(labels)}"


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
