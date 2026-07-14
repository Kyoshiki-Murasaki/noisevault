from __future__ import annotations

import inspect
from pathlib import Path

from ..snapshot_io import save_snapshot
from .ibm import build_snapshot_from_backend, snapshot_destination


def list_fake_backends() -> list[str]:
    try:
        import qiskit_ibm_runtime.fake_provider as fake_provider
    except ImportError as exc:
        raise RuntimeError("Install noisevault[pilot] to import Qiskit fake backends.") from exc
    names = []
    for name, _value in inspect.getmembers(fake_provider, inspect.isclass):
        if name.startswith("Fake") and name.endswith("V2"):
            names.append(name)
    return sorted(names)


def import_fake_backend(name: str, output_root: str | Path = "snapshots") -> Path:
    import qiskit_ibm_runtime.fake_provider as fake_provider

    if not hasattr(fake_provider, name):
        available = ", ".join(list_fake_backends()[:20])
        raise ValueError(f"Unknown fake backend {name}. Available examples: {available}")
    backend = getattr(fake_provider, name)()
    snapshot = build_snapshot_from_backend(
        backend,
        source=f"qiskit_ibm_runtime.fake_provider.{name} packaged calibration",
    )
    snapshot.provider = "qiskit_fake"
    imported_at = snapshot.captured_at
    if snapshot.provenance.source_timestamp:
        snapshot.captured_at = snapshot.provenance.source_timestamp
        snapshot.provenance.notes.append(f"Imported into NoiseVault at {imported_at}.")
    return save_snapshot(snapshot, snapshot_destination(output_root, snapshot))
