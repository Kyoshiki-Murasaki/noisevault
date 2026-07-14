from __future__ import annotations

import numpy as np
import pytest

from noisevault.metrics import hellinger_fidelity, total_variation_distance


def test_metrics_identity_and_extremes():
    p = np.array([0.5, 0.5])
    assert total_variation_distance(p, p) == pytest.approx(0.0)
    assert hellinger_fidelity(p, p) == pytest.approx(1.0)
    assert total_variation_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert hellinger_fidelity([1, 0], [0, 1]) == pytest.approx(0.0)
