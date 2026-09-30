"""Command line: ``noisevault`` (alias ``nv``)."""

from __future__ import annotations

import platform
import warnings
import zlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from . import __version__
from .catalog import bundled_profiles, vault_dir
from .errors import NoiseVaultError
from .profile import Profile, load_file

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="NoiseVault: real device noise, pinned and portable.",
)
out = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False, soft_wrap=True)

_OPTIONAL = (
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "cirq-core",
    "cirq-google",
    "pennylane",
    "stim",
    "pymatching",
)


@app.callback()
def main() -> None:
    """NoiseVault: real device noise, pinned and portable."""


@app.command()
def doctor() -> None:
    """Show versions of NoiseVault, Python and the optional frameworks, and where profiles live."""
    table = Table("component", "status")
    table.add_row("noisevault", __version__)
    table.add_row("python", platform.python_version())
    for package in _OPTIONAL:
        try:
            table.add_row(package, version(package))
        except PackageNotFoundError:
            table.add_row(package, "not installed")
    vault = vault_dir()
    count = len(list(vault.glob("*.json*"))) if vault.is_dir() else 0
    table.add_row("vault", f"{vault} ({count} files)")
    table.add_row("bundled profiles", str(len(bundled_profiles())))
    out.print(table)


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Profile: .json or .json.gz, format 1.0 or 0.1.")],
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
) -> None:
    """Check a profile file against the format rules."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            profile = load_file(file)
        except FileNotFoundError:
            _fail(f"no file {file}")
        except (OSError, EOFError, zlib.error) as exc:  # a directory, bad gzip or cut-off file
            _fail(f"cannot read {file}: {exc}")
        except ValidationError as exc:
            for line in _validation_lines(exc):
                err.print(f"error: {line}", markup=False)
            raise typer.Exit(1) from None
        except (ValueError, NoiseVaultError) as exc:
            _fail(str(exc))
    notes = [str(w.message) for w in caught] + _soft_issues(profile)
    when = profile.device.calibrated_at.isoformat() if profile.device.calibrated_at else "undated"
    out.print(
        f"ok: {profile.id} calibrated {when}, {profile.device.num_qubits} qubits,"
        f" {len(profile.gates)} gates, {len(profile.calibrations)} calibration records,"
        f" {profile.short_fingerprint}",
        markup=False,
    )
    for note in notes:
        err.print(f"warning: {note}", markup=False)
    if strict and notes:
        raise typer.Exit(1)


def _fail(message: str) -> None:
    err.print(f"error: {message}", markup=False)
    raise typer.Exit(1)


def _validation_lines(exc: ValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        for part in message.split("\n"):
            lines.append(f"{where}: {part}" if where else part)
    return lines


def _soft_issues(profile: Profile) -> list[str]:
    """Legal but noteworthy values: they change what a conversion produces."""
    notes = []
    for i in range(profile.device.num_qubits):
        q = profile.table.qubit(i)
        if q.t1_ns is not None and q.t2_ns is not None and q.t2_ns > 2 * q.t1_ns:
            notes.append(f"qubit {i}: T2 exceeds 2*T1; conversions clamp it to 2*T1")
    for record in profile.calibrations:
        if record.scope == "cycle":
            notes.append(f"{record.gate} {list(record.qubits)}: error is per cycle, not per gate")
    return notes
