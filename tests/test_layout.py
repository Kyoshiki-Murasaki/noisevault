from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import toy

from noisevault.errors import LayoutError
from noisevault.layout import normalize_layout, suggest_layout
from noisevault.profile import Profile


def _line(n: int, **sections) -> Profile:
    return Profile.model_validate(
        toy(
            device={"name": "line", "technology": "superconducting", "num_qubits": n},
            connectivity={"edges": [[i, i + 1] for i in range(n - 1)]},
            **sections,
        )
    )


def test_integer_labels_default_to_identity() -> None:
    assert normalize_layout([2, 0], None, _line(3)) == {2: 2, 0: 0}
    assert normalize_layout([np.int64(1)], None, _line(3)) == {1: 1}


def test_mapping_and_sequence_layouts() -> None:
    profile = _line(5)
    assert normalize_layout(["a", "b"], {"a": 4, "b": 1, "unused": 0}, profile) == {"a": 4, "b": 1}
    assert normalize_layout([0, 1], [3, 2], profile) == {0: 3, 1: 2}


@pytest.mark.parametrize(
    ("labels", "layout", "match"),
    [
        (["a"], None, "no integer index"),
        ([True], None, "no integer index"),
        ([0, 1], {0: 1}, r"no physical qubit for \[1\]"),
        ([0, 1], {0: 2, 1: 2}, "both 0 and 1 to qubit 2"),
        ([0], {0: 9}, "qubits 0..4"),
        ([0], {0: -1}, "qubits 0..4"),
        ([0], {0: "q3"}, "not a qubit index"),
        ([0], {0: 3}, "disabled"),
    ],
)
def test_layout_errors(labels, layout, match) -> None:
    profile = _line(5, qubits=[{"index": 3, "disabled": True}])
    with pytest.raises(LayoutError, match=match):
        normalize_layout(labels, layout, profile)


def test_adapters_can_supply_their_own_integer_rule() -> None:
    class LineQubit:
        def __init__(self, x: int) -> None:
            self.x = x

    qubits = [LineQubit(1), LineQubit(4)]
    mapping = normalize_layout(qubits, None, _line(5), index_of=lambda q: q.x)
    assert list(mapping.values()) == [1, 4]


def _grid(rows: int, cols: int, seed: int = 3) -> Profile:
    rng = np.random.default_rng(seed)
    n = rows * cols
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    calibrations = [
        {"gate": "cz", "qubits": list(e), "avg_infidelity": float(rng.uniform(2e-3, 2e-2))}
        for e in edges
    ]
    calibrations += [
        {"gate": "sx", "qubits": [q], "avg_infidelity": float(rng.uniform(1e-4, 1e-3))}
        for q in range(n)
    ]
    qubits = [{"index": q, "readout": {"error": float(rng.uniform(5e-3, 5e-2))}} for q in range(n)]
    return Profile.model_validate(
        toy(
            device={"name": "grid", "technology": "superconducting", "num_qubits": n},
            connectivity={"edges": [list(e) for e in edges]},
            calibrations=calibrations,
            qubits=qubits,
        )
    )


def _is_chain(profile: Profile, layout: dict[int, int]) -> bool:
    edges = {frozenset(e) for e in profile.table.edges()}
    path = [layout[i] for i in range(len(layout))]
    return len(set(path)) == len(path) and all(
        frozenset(pair) in edges for pair in zip(path, path[1:], strict=False)
    )


def test_suggest_layout_is_a_deterministic_connected_chain_of_good_qubits() -> None:
    profile = _grid(4, 4)
    layout = suggest_layout(profile, 5)
    assert sorted(layout) == [0, 1, 2, 3, 4] and _is_chain(profile, layout)
    assert suggest_layout(_grid(4, 4), 5) == layout
    worst = max(range(16), key=lambda q: profile.table.qubit(q).readout[0])
    assert worst not in layout.values()


def test_suggest_layout_avoids_disabled_qubits_and_gates() -> None:
    profile = _line(5, qubits=[{"index": 1, "disabled": True}])
    assert set(suggest_layout(profile, 3).values()) == {2, 3, 4}
    broken = _line(5, calibrations=[{"gate": "cz", "qubits": [2, 3], "disabled": True}])
    with pytest.raises(LayoutError, match="no connected chain"):
        suggest_layout(broken, 4)
    assert set(suggest_layout(broken, 3).values()) == {0, 1, 2}


def test_suggest_layout_is_fast_at_156_qubits() -> None:
    profile = _grid(12, 13)
    profile.table.typical(1, (0,))  # table built outside the timing
    start = time.perf_counter()
    layout = suggest_layout(profile, 20)
    elapsed = time.perf_counter() - start
    assert _is_chain(profile, layout) and elapsed < 0.5


def test_suggest_layout_on_all_to_all() -> None:
    profile = Profile.uniform(
        "ions", technology="trapped_ion", num_qubits=56, one_qubit_error=3e-5, two_qubit_error=1e-3
    )
    assert len(set(suggest_layout(profile, 10).values())) == 10
    with pytest.raises(LayoutError):
        suggest_layout(profile, 57)


def test_suggest_layout_on_all_to_all_takes_the_best_qubits_fast() -> None:
    good = [5, 17, 140]
    data = Profile.uniform(
        "ions",
        technology="trapped_ion",
        num_qubits=156,
        one_qubit_error=1e-3,
        two_qubit_error=1e-2,
        readout_error=0.02,
    ).to_dict()
    data["qubits"] = [{"index": q, "readout": {"error": 0.001}} for q in good]
    data["qubits"].append({"index": 3, "disabled": True})
    profile = Profile.model_validate(data)
    profile.table.typical(1, (0,))  # table built outside the timing
    start = time.perf_counter()
    layout = suggest_layout(profile, 100)
    elapsed = time.perf_counter() - start
    assert sorted(layout) == list(range(100)) and len(set(layout.values())) == 100
    assert set(good) <= set(layout.values()) and 3 not in layout.values()
    assert elapsed < 0.05
