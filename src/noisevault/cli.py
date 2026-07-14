from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .pilot import run_pilot
from .providers.ibm import harvest_ibm
from .providers.qiskit_fake import import_fake_backend, list_fake_backends
from .schema import validate_snapshot
from .snapshot_io import load_snapshot

app = typer.Typer(no_args_is_help=True, help="NoiseVault pilot CLI")
console = Console()


@app.command("doctor")
def doctor() -> None:
    """Check package and optional framework availability."""
    import importlib.util
    import platform

    table = Table("Component", "Status")
    table.add_row("Python", platform.python_version())
    for name, module in (
        ("Qiskit Aer", "qiskit_aer"),
        ("IBM Runtime", "qiskit_ibm_runtime"),
        ("Cirq", "cirq"),
        ("PennyLane", "pennylane"),
        ("Matplotlib", "matplotlib"),
    ):
        table.add_row(name, "available" if importlib.util.find_spec(module) else "missing")
    console.print(table)


@app.command("validate")
def validate(path: Path) -> None:
    snapshot = load_snapshot(path, require_valid=False)
    report = validate_snapshot(snapshot)
    for issue in report.issues:
        console.print(f"[{issue.severity}] {issue.path}: {issue.message}")
    raise typer.Exit(0 if report.valid else 1)


@app.command("list-fakes")
def list_fakes() -> None:
    for name in list_fake_backends():
        console.print(name)


@app.command("import-fake")
def import_fake(name: str, output_root: Path = Path("snapshots")) -> None:
    console.print(import_fake_backend(name, output_root))


@app.command("harvest-ibm")
def harvest(
    output_root: Path = Path("snapshots"),
    max_backends: int = 2,
    backend: list[str] | None = None,
) -> None:
    paths = harvest_ibm(output_root, max_backends=max_backends, backend_names=backend)
    for path in paths:
        console.print(path)


@app.command("run-pilot")
def run(
    root: Path = Path("."),
    mode: str = typer.Option("auto", help="auto, offline, or live"),
    profile: str = typer.Option("full", help="smoke or full"),
) -> None:
    result = run_pilot(root, mode=mode, profile=profile)
    console.print_json(json.dumps(result["summary"]))
    raise typer.Exit(0 if result["summary"]["status"] != "PARTIAL" else 2)


if __name__ == "__main__":
    app()
