from __future__ import annotations

import numpy as np

from noisevault.readout import apply_readout_error, confusion_matrix


def test_confusion_matrix_columns_sum_to_one():
    matrix = confusion_matrix(0.2, 0.1)
    assert np.allclose(matrix.sum(axis=0), [1.0, 1.0])


def test_readout_preserves_probability_mass(demo_snapshot):
    probabilities = np.zeros(4)
    probabilities[0] = 1.0
    result = apply_readout_error(probabilities, demo_snapshot, [0, 1])
    assert np.isclose(result.sum(), 1.0)
    assert np.all(result >= 0)
