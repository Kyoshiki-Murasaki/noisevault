from __future__ import annotations

import numpy as np


def normalize_probabilities(values: np.ndarray | list[float]) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    if probs.ndim != 1:
        raise ValueError("Probability vector must be one-dimensional.")
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Probability vector has zero mass.")
    return probs / total


def total_variation_distance(p: np.ndarray | list[float], q: np.ndarray | list[float]) -> float:
    p_arr = normalize_probabilities(p)
    q_arr = normalize_probabilities(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"Shape mismatch: {p_arr.shape} vs {q_arr.shape}")
    return float(0.5 * np.abs(p_arr - q_arr).sum())


def hellinger_fidelity(p: np.ndarray | list[float], q: np.ndarray | list[float]) -> float:
    p_arr = normalize_probabilities(p)
    q_arr = normalize_probabilities(q)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"Shape mismatch: {p_arr.shape} vs {q_arr.shape}")
    coefficient = float(np.sqrt(p_arr * q_arr).sum())
    return coefficient**2


def success_probability_zero(probs: np.ndarray | list[float]) -> float:
    return float(normalize_probabilities(probs)[0])
