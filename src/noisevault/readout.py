from __future__ import annotations

import numpy as np

from .metrics import normalize_probabilities
from .models import DeviceNoiseSnapshot


def confusion_matrix(prob_meas0_prep1: float | None, prob_meas1_prep0: float | None) -> np.ndarray:
    p01 = float(prob_meas0_prep1 or 0.0)  # measured 0 given prepared 1
    p10 = float(prob_meas1_prep0 or 0.0)  # measured 1 given prepared 0
    return np.array([[1.0 - p10, p01], [p10, 1.0 - p01]], dtype=float)


def apply_readout_error(
    probabilities: np.ndarray | list[float],
    snapshot: DeviceNoiseSnapshot,
    physical_qubits: list[int],
) -> np.ndarray:
    matrix = np.array([[1.0]])
    by_index = {qubit.index: qubit for qubit in snapshot.qubits}
    for physical in physical_qubits:
        qubit = by_index[physical]
        matrix = np.kron(
            matrix,
            confusion_matrix(qubit.prob_meas0_prep1, qubit.prob_meas1_prep0),
        )
    result = matrix @ normalize_probabilities(probabilities)
    return normalize_probabilities(result)
