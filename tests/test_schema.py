from __future__ import annotations

import math

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


def test_all_committed_snapshots_validate(root):
    paths = sorted((root / "snapshots").rglob("*.json"))
    assert paths
    for path in paths:
        assert validate_snapshot(load_snapshot(path)).valid, path


def test_nonfinite_calibration_is_rejected(demo_snapshot):
    snapshot = demo_snapshot.model_copy(deep=True)
    snapshot.qubits[0].t1_us = math.nan
    report = validate_snapshot(snapshot)
    assert not report.valid
    assert any(issue.code == "nonfinite_value" for issue in report.errors)
