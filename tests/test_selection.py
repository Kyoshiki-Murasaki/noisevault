from __future__ import annotations

from noisevault.select import best_connected_subset, connected


def test_connected_subset(demo_snapshot):
    for size in range(2, 6):
        subset = best_connected_subset(demo_snapshot, size)
        assert len(subset) == size
        assert connected(demo_snapshot, subset)
