from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from noisevault.pilot import _choose_drift_pair
from noisevault.snapshot_io import load_snapshot


def test_drift_pair_rejects_duplicate_payloads(root):
    old = load_snapshot(root / "snapshots/demo/demo_linear_5_2024-01-15.json")
    new = load_snapshot(root / "snapshots/demo/demo_linear_5_2026-07-14.json")
    duplicate = deepcopy(new)
    duplicate.captured_at = "2026-07-15T09:00:00Z"

    older, newer = _choose_drift_pair(
        [
            (Path("old.json"), old),
            (Path("new.json"), new),
            (Path("duplicate.json"), duplicate),
        ]
    )

    assert older[1].provenance.raw_hash != newer[1].provenance.raw_hash
    assert older[0] == Path("old.json")
    assert newer[1].provenance.raw_hash == new.provenance.raw_hash
    assert newer[1].provenance.source_timestamp == new.provenance.source_timestamp
