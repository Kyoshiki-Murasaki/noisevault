from __future__ import annotations

import numpy as np
import pytest

from noisevault.channels import (
    CalibrationUnavailableError,
    channel_sequence_for_operation,
    depolarizing_kraus,
    find_gate_calibration,
    thermal_relaxation_kraus,
)


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


def test_partial_relaxation_data_remains_physical_and_nontrivial():
    for kraus in (
        thermal_relaxation_kraus(100.0, None, 500.0),
        thermal_relaxation_kraus(None, 80.0, 500.0),
    ):
        assert len(kraus) > 1
        assert_trace_preserving(kraus)


def test_exact_alias_and_directed_calibration_are_required(demo_snapshot):
    assert find_gate_calibration(demo_snapshot, "h", (0,)).name == "h"
    directed = demo_snapshot.model_copy(deep=True)
    directed.gates = [
        gate
        for gate in directed.gates
        if len(gate.qubits) != 2 or tuple(gate.qubits) == (0, 1)
    ]
    with pytest.raises(CalibrationUnavailableError):
        channel_sequence_for_operation(directed, "cx", (0, 1), (1, 0))
