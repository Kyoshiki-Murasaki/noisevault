from __future__ import annotations

from noisevault.schema import validate_snapshot
from noisevault.snapshot_io import load_snapshot, snapshot_content_hash


def test_demo_snapshot_valid(demo_snapshot):
    report = validate_snapshot(demo_snapshot)
    assert report.valid
    assert not report.errors


def test_snapshot_hash_stable(root):
    path = root / "snapshots/demo/demo_linear_5_2026-07-14.json"
    first = snapshot_content_hash(load_snapshot(path))
    second = snapshot_content_hash(load_snapshot(path))
    assert first == second
    assert first.startswith("sha256:")
