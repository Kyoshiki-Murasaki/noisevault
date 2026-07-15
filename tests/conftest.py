from __future__ import annotations

from pathlib import Path

import pytest

from noisevault.snapshot_io import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def demo_snapshot(root):
    return load_snapshot(root / "snapshots/demo/demo_linear_5_2026-07-14.json")
