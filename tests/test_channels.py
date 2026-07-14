from __future__ import annotations

import numpy as np

from noisevault.channels import depolarizing_kraus, thermal_relaxation_kraus


def assert_trace_preserving(kraus):
    dimension = kraus[0].shape[0]
    accumulator = np.zeros((dimension, dimension), dtype=complex)
    for operator in kraus:
        accumulator += operator.conj().T @ operator
    assert np.allclose(accumulator, np.eye(dimension), atol=1e-10)


def test_thermal_channel_is_cptp():
    assert_trace_preserving(thermal_relaxation_kraus(100.0, 80.0, 500.0))


def test_depolarizing_channels_are_cptp():
    assert_trace_preserving(depolarizing_kraus(0.01, 1))
    assert_trace_preserving(depolarizing_kraus(0.03, 2))
