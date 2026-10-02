from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import noisevault as nv
from noisevault import compare as fit
from noisevault.compare import CircuitScore, Comparison, NoEstimate, compare
from noisevault.counts import MeasuredCounts, PlannedCircuit, load_counts, plan, simulate
from noisevault.errors import CountsError, NoiseVaultError
from noisevault.metrics import scale_readout
from noisevault.profile import ErrorFactor, Profile
from noisevault.reference import Op
from noisevault.reference import probabilities as reference

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "kingston-simulated.counts.json"
KINGSTON = "ibm_kingston@2026-04-15"
TRUE_GATE, TRUE_READOUT = 1.8, 1.3
EXAMPLE_SEED = 1
RUN_AT = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
LATER = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
slow = pytest.mark.skipif(
    os.environ.get("NOISEVAULT_SLOW") != "1", reason="set NOISEVAULT_SLOW=1 to run (3 minutes)"
)


def scaled(profile: Profile, gate: float | None = None, readout: float | None = None) -> Profile:
    factors = {k: {"factor": v} for k, v in (("gates", gate), ("readout", readout)) if v}
    return profile.model_copy(update={"unmodeled_error": factors})


def example_counts() -> MeasuredCounts:
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, TRUE_GATE, TRUE_READOUT)
    return simulate(truth, plan(kingston), shots=4000, seed=EXAMPLE_SEED, run_at=RUN_AT)


def written(
    profile: Profile, circuits: Sequence[tuple[str, Sequence[Op], dict[str, int]]]
) -> MeasuredCounts:
    return MeasuredCounts.model_validate(
        {
            "nv_counts": "1.0",
            "source": "simulated",
            "profile": {"id": profile.id, "fingerprint": profile.calibration_fingerprint},
            "backend": profile.device.name,
            "run_at": LATER.isoformat(),
            "bit_order": "clbit0_left",
            "execution": {
                "client": "hand-written",
                "transpiled": False,
                "gate_twirling": False,
                "measure_twirling": False,
                "dynamical_decoupling": False,
                "init_qubits": True,
            },
            "circuits": [
                {
                    "name": name,
                    "qubits": list(range(len(next(iter(counts))))),
                    "ops": [[op.name, list(op.qubits), list(op.params)] for op in ops],
                    "shots": sum(counts.values()),
                    "counts": counts,
                }
                for name, ops, counts in circuits
            ],
        }
    )


def rebound(counts: MeasuredCounts, profile: Profile) -> MeasuredCounts:
    binding = {"id": profile.id, "fingerprint": profile.calibration_fingerprint}
    return counts.model_copy(update={"profile": binding})


def moved(counts: MeasuredCounts, circuit: str, source: str, target: str, n: int) -> MeasuredCounts:
    data = counts.to_dict()
    for entry in data["circuits"]:
        if entry["name"] == circuit:
            entry["counts"][source] -= n
            entry["counts"][target] = entry["counts"].get(target, 0) + n
    return MeasuredCounts.model_validate(data)


def covers(result: ErrorFactor | NoEstimate, truth: float) -> bool:
    if not isinstance(result, ErrorFactor):
        return False
    return (result.low or 0) <= truth <= (result.high or math.inf)


def toy() -> Profile:
    return Profile.uniform(
        "toy", technology="superconducting", num_qubits=1, one_qubit_error=0.001, readout_error=0.01
    )


def one_qubit(name: str, gates: dict[str, Any], **sections: Any) -> Profile:
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {"name": name, "technology": "superconducting", "num_qubits": 1},
            "connectivity": "all_to_all",
            "gates": gates,
            **sections,
        }
    )


XX = (Op("x", (0,)), Op("x", (0,)))


def test_the_example_counts_give_back_the_factors_they_were_simulated_with() -> None:
    result = compare(nv.load(KINGSTON), load_counts(EXAMPLE))
    assert covers(result.gates, TRUE_GATE), result.gates
    assert covers(result.readout, TRUE_READOUT), result.readout
    assert result.gates.high / result.gates.low < 1.8


def test_the_example_counts_file_is_current() -> None:
    expected = example_counts()
    assert load_counts(EXAMPLE).sha256 == expected.sha256, (
        f"{EXAMPLE.relative_to(ROOT)} is stale; regenerate it with python tests/test_compare.py"
    )


@slow
def test_each_interval_covers_its_truth_on_the_plan() -> None:
    kingston = nv.load(KINGSTON)
    truth, circuits = scaled(kingston, TRUE_GATE, TRUE_READOUT), plan(kingston)
    hits = {"gate": 0, "readout": 0}
    for seed in range(1000, 1100):
        result = compare(kingston, simulate(truth, circuits, shots=4000, seed=seed, run_at=RUN_AT))
        hits["gate"] += covers(result.gates, TRUE_GATE)
        hits["readout"] += covers(result.readout, TRUE_READOUT)
    assert 88 <= hits["gate"] <= 100 and 88 <= hits["readout"] <= 100, hits


def sparse_counts(errors: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit("sparse", {"x": {"avg_infidelity": 0.0}}, readout={"error": 0.00055})
    circuits = {c.name: c.ops for c in plan(profile)}
    assert list(circuits) == ["single_qubit", "readout"]
    half = errors // 2
    return profile, written(
        profile,
        [
            ("single_qubit", circuits["single_qubit"], {"0": 2000 - half, "1": half}),
            ("readout", circuits["readout"], {"0": 2000 - errors + half, "1": errors - half}),
        ],
    )


def test_sparse_counts_with_no_error_keep_an_upper_end_the_chi_square_cutoff_misses() -> None:
    result = compare(*sparse_counts(0))
    assert isinstance(result.readout, ErrorFactor) and result.readout.bound == "lower"
    assert result.readout.high > 1.3


def test_sparse_counts_cover_the_true_factor_over_every_error_count() -> None:
    p = scale_readout((0.00055, 0.00055), 1.0)[0]
    coverage = 0.0
    for errors in range(80):
        weight = math.comb(4000, errors) * p**errors * (1 - p) ** (4000 - errors)
        coverage += weight * covers(compare(*sparse_counts(errors)).readout, 1.0)
    assert coverage >= 0.95


def dense(errors: int) -> tuple[Profile, MeasuredCounts]:
    profile = Profile.uniform(
        "dense", technology="superconducting", num_qubits=1, one_qubit_error=0.0, readout_error=0.1
    )
    shots = 1_000_000
    return profile, written(profile, [("readout", (), {"0": shots - errors, "1": errors})])


def test_dense_counts_refine_the_maximum_between_grid_points() -> None:
    result = compare(*dense(100698))
    assert isinstance(result.readout, ErrorFactor)
    assert abs(result.readout.factor - 1.007832) < 1e-4
    assert result.readout.low < result.readout.factor < result.readout.high
    assert result.readout.describe() == "x1.008 (95% interval 1.001 to 1.015)"
    assert result.dof == 0 and result.dispersion == 1


def test_dense_counts_cover_the_true_factor_without_wide_intervals() -> None:
    rng = np.random.default_rng(2024)
    hits = sum(
        covers(compare(*dense(int(rng.binomial(1_000_000, 0.1)))).readout, 1.0) for _ in range(200)
    )
    assert 180 <= hits <= 199


def test_two_runs_of_one_plan_build_the_surface_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []
    build = fit._Surface.build.__func__

    def counting(cls: type, base: Profile, circuits: Sequence[PlannedCircuit]) -> Any:
        built.append(circuits)
        return build(cls, base, circuits)

    monkeypatch.setattr(fit._Surface, "build", classmethod(counting))
    monkeypatch.setattr(fit._Surface, "_built", type(fit._Surface._built)())
    profile = toy()
    circuits = [PlannedCircuit(name="xx", qubits=(0,), ops=XX)]
    for seed in (1, 2):
        compare(profile, simulate(profile, circuits, shots=4000, seed=seed, run_at=LATER))
    assert len(built) == 1


def test_the_surface_is_the_reference_at_every_node_and_a_distribution_between() -> None:
    kingston = nv.load(KINGSTON)
    circuits = plan(kingston)
    surface = fit._Surface.cached(kingston, circuits)
    readouts = cycle((surface.readout[0], 0.0, surface.readout[-1]))
    for node, readout in zip(surface.gate_nodes, readouts, strict=False):
        truth = scaled(kingston, math.exp(node), math.exp(readout))
        expected = np.concatenate(
            [
                reference(truth, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
                for c in circuits
            ]
        )
        assert np.abs(surface.probs_at([node], [readout])[0] - expected).max() < 1e-12
    between = surface.probs_at(
        (surface.gate_nodes[:-1] + surface.gate_nodes[1:]) / 2,
        np.zeros(len(surface.gate_nodes) - 1),
    )
    assert between.min() >= 0
    for part in surface.slices:
        assert np.abs(between[:, part].sum(axis=1) - 1).max() < 1e-12


def floor_counts(zero: int, one: int) -> tuple[Profile, MeasuredCounts]:
    profile = one_qubit(
        "idle",
        {"id": {"avg_infidelity": 0.001, "duration_ns": 100}},
        idle={"t1_us": 100, "t2_us": 200},
    )
    one_id, two_ids = (Op("id", (0,)),), (Op("id", (0,)), Op("id", (0,)))
    return profile, written(
        profile,
        [("one_id", one_id, {"0": zero, "1": one}), ("two_ids", two_ids, {"0": zero + one})],
    )


def test_zero_probability_cells_give_finite_likelihoods_in_the_fit_and_every_draw() -> None:
    profile, counts = floor_counts(4000, 0)
    (floor,) = fit._floor_factors(profile, counts.circuits)
    result = compare(profile, counts)
    assert math.isfinite(result.deviance) and result.dof == 1 and result.p_value is not None
    assert "NaN" not in json.dumps(result.to_dict())
    assert result.gates.factor == pytest.approx(floor) and result.gates.bound == "lower"
    above = compare(*floor_counts(3999, 1))
    assert above.gates.factor > floor * 1.01
    assert "NaN" not in json.dumps(above.to_dict())


@pytest.mark.parametrize(
    ("ref", "dof"),
    [("ibm_kingston@2026-04-15", 38), ("ibm_fez@2025-02-26", 30), ("ibm_boston@2026-04-17", 26)],
)
def test_degrees_of_freedom_count_the_outcomes_the_profile_supports(ref: str, dof: int) -> None:
    profile = nv.load(ref)
    run_at = max(RUN_AT, profile.device.calibrated_at)
    result = compare(profile, simulate(profile, plan(profile), shots=4000, seed=3, run_at=run_at))
    assert result.dof == dof


def test_a_readout_only_file_has_no_degrees_of_freedom_and_is_not_tested() -> None:
    profile = toy()
    result = compare(profile, written(profile, [("readout", (), {"0": 3960, "1": 40})]))
    assert (result.dof, result.dispersion, result.p_value) == (0, 1, None)
    assert "fit             not testable (no degrees of freedom left after fitting)" in str(result)


def test_a_flat_gate_axis_keeps_the_readout_estimate_and_an_upper_gate_bound() -> None:
    kingston = nv.load(KINGSTON)
    truth = scaled(kingston, 0.05, 1.0)
    result = compare(kingston, simulate(truth, plan(kingston), shots=4000, seed=5, run_at=RUN_AT))
    assert covers(result.readout, 1.0), result.readout
    assert isinstance(result.gates, ErrorFactor) and result.gates.bound == "lower"
    assert result.gates.low is None and result.gates.high < 1
    lowest_floor = min(fit._floor_factors(kingston, plan(kingston)))
    assert result.gates.factor == pytest.approx(lowest_floor)


def test_a_gate_axis_flat_at_the_estimate_is_not_a_collinearity() -> None:
    gates = {"rz": {"virtual": True}, "sx": {"avg_infidelity": 1e-4, "duration_ns": 100}}
    idle = {"t1_us": 111, "t2_us": 222}
    profile = one_qubit("slow", gates, readout={"error": 0.01}, idle=idle)
    circuits = plan(profile)
    result = compare(profile, simulate(profile, circuits, shots=4000, seed=0, run_at=LATER))
    assert isinstance(result.gates, ErrorFactor) and result.gates.bound == "lower"
    assert result.gates.factor == pytest.approx(1.0)
    assert covers(result.readout, 1.0), result.readout
    (floor,) = fit._floor_factors(profile, circuits)
    assert 2 < floor < 4
    surface = fit._Surface.cached(profile, circuits)
    expected = np.concatenate(
        [
            reference(scaled(profile, floor, 1.0), c.ops, 1, layout=c.qubits, unknown_gates="error")
            for c in circuits
        ]
    )
    assert np.abs(surface.probs_at([math.log(floor)], [0.0])[0] - expected).max() < 1e-12


def test_x_then_x_counts_identify_neither_factor() -> None:
    profile = toy()
    counts = simulate(
        profile, [PlannedCircuit(name="xx", qubits=(0,), ops=XX)], shots=4000, seed=2, run_at=LATER
    )
    result = compare(profile, counts)
    same = NoEstimate("gate error and readout error move these counts the same way")
    assert (result.gates, result.readout) == (same, same)
    center = fit._exact(profile, counts.circuits, 1.0, 1.0)
    eigen = np.linalg.eigvalsh(fit._fisher(profile, counts, center, 1.0, 1.0))
    assert eigen[0] < fit.SEPARABLE * eigen[1]
    assert (result.dof, result.p_value) == (0, None)
    assert "not testable (no degrees of freedom left after fitting)" in str(result)


def test_readout_counts_with_no_observed_error_give_only_an_upper_end() -> None:
    profile = toy()
    result = compare(profile, written(profile, [("readout", (), {"0": 4000})]))
    assert isinstance(result.readout, ErrorFactor)
    assert (result.readout.bound, result.readout.low) == ("lower", None)
    assert result.readout.high is not None


def test_notes_name_what_the_factors_cannot_scale_or_charge() -> None:
    profile = one_qubit("odd", {"x": {"pauli": [0.1, 0.0, 0.1]}}, readout={"error": 0.01})
    profile = profile.model_copy(
        update={"device": {**profile.to_dict()["device"], "num_qubits": 3}}
    )
    ops = (Op("x", (0,)), Op("x", (1,)), Op("x", (2,)), Op("delay", (0,), (100.0,)))
    counts = {"111": 3400, "011": 200, "101": 200, "110": 200}
    result = compare(profile, written(profile, [("three_x", ops, counts)]))
    assert result.notes == (
        "x on qubits 0, 1 and 2 is not scaled (it has a negative Pauli-Lindblad rate)",
        "delays on qubit 0 add no idle error (no T1 or T2 stated)",
    )
    assert result.gates == NoEstimate("no circuit's outcomes move with gate error")
    lines = str(result).split("\n")
    note = lines.index("note            x on qubits 0, 1 and 2 is not scaled")
    assert lines[note + 1 : note + 3] == [
        "                (it has a negative Pauli-Lindblad rate)",
        "                delays on qubit 0 add no idle error (no T1 or T2 stated)",
    ]


def excess_readout(seed: int) -> MeasuredCounts:
    kingston = nv.load(KINGSTON)
    data = kingston.to_dict()
    record = next(q for q in data["qubits"] if q["index"] == 150)
    a, b = kingston.table.qubit(150).readout
    record["readout"] = {"p1_given_0": 3 * a, "p0_given_1": 3 * b}
    truth = Profile.from_dict(data)
    return rebound(simulate(truth, plan(kingston), shots=4000, seed=seed, run_at=RUN_AT), kingston)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_excess_readout_error_on_one_qubit_is_a_poor_fit(seed: int) -> None:
    result = compare(nv.load(KINGSTON), excess_readout(seed))
    assert result.p_value < 0.01
    assert "no one pair of factors fits every circuit" in str(result)


def test_counts_simulated_from_the_factors_fit_within_shot_noise() -> None:
    result = compare(nv.load(KINGSTON), load_counts(EXAMPLE))
    assert result.p_value >= 0.01


def fez_with_impossible_shots(n: int) -> tuple[Profile, MeasuredCounts]:
    fez = nv.load("ibm_fez@2025-02-26")
    run_at = datetime(2025, 2, 27, 11, 42, tzinfo=UTC)
    counts = simulate(fez, plan(fez), shots=4000, seed=7, run_at=run_at)
    return fez, moved(counts, "readout", "0000", "1000", n)


def test_shots_the_profile_rules_out_set_p_to_zero_and_count_in_the_tvd() -> None:
    fez, counts = fez_with_impossible_shots(9)
    result = compare(fez, counts)
    assert result.p_value == 0
    assert {s.name: s.impossible for s in result.circuits}["readout"] == 9
    stated = np.concatenate(
        [
            reference(fez, c.ops, len(c.qubits), layout=c.qubits, unknown_gates="error")
            for c in counts.circuits
        ]
    )
    for c, score, start in zip(
        counts.circuits, result.circuits, np.cumsum([0, 16, 8, 4]), strict=True
    ):
        part = stated[start : start + 2 ** len(c.qubits)]
        assert score.tvd_profile == pytest.approx(0.5 * np.abs(c.vector() / c.shots - part).sum())
    assert "qubit 136 has P(1|0) = 0, and 9 readout shots read it as 1" in str(result)
    assert result.fitted_profile().unmodeled_error.fit.impossible_shots == 9


def test_counts_the_profile_rules_out_entirely_identify_neither_factor() -> None:
    fez, counts = fez_with_impossible_shots(0)
    data = counts.to_dict()
    data["circuits"] = [
        {**c, "counts": {"1000": 4000}} for c in data["circuits"] if c["name"] == "readout"
    ]
    result = compare(fez, MeasuredCounts.model_validate(data))
    assert result.gates == result.readout == NoEstimate("the profile rules out every shot")
    assert result.p_value == 0 and result.impossible_shots == 4000


def refused(profile: Profile, counts: MeasuredCounts) -> str:
    with pytest.raises(CountsError) as caught:
        compare(profile, counts)
    message = str(caught.value)
    assert "\n" not in message
    return message


def test_counts_planned_from_another_calibration_are_refused() -> None:
    kingston = nv.load(KINGSTON)
    data = kingston.to_dict()
    data["qubits"][0]["t1_us"] = data["qubits"][0].get("t1_us", 100) + 1
    edited = Profile.from_dict(data)
    message = refused(edited, load_counts(EXAMPLE))
    assert f"planned from nv:{kingston.fingerprint[:12]}" in message
    assert f"nv list` to find nv:{kingston.fingerprint[:12]}" in message


def test_counts_from_another_backend_are_refused() -> None:
    counts = load_counts(EXAMPLE).model_copy(update={"backend": "ibm_fez"})
    assert "ran on ibm_fez, but the profile describes ibm_kingston" in refused(
        nv.load(KINGSTON), counts
    )


def test_counts_that_ran_before_their_calibration_are_refused() -> None:
    counts = load_counts(EXAMPLE).model_copy(update={"run_at": "2026-04-01T00:00:00Z"})
    assert "ran at 2026-04-01 00:00Z, before the calibration" in refused(nv.load(KINGSTON), counts)


def test_an_op_the_profile_does_not_calibrate_on_its_qubits_is_refused() -> None:
    kingston = nv.load(KINGSTON)
    data = load_counts(EXAMPLE).to_dict()
    data["circuits"][0]["ops"].append(["cz", [0, 2], []])
    message = refused(kingston, MeasuredCounts.model_validate(data))
    assert message.startswith("circuit ghz_chain: cz on qubits 148-150: ")


def test_profile_compare_is_compare_and_import_noisevault_leaves_the_fit_unloaded() -> None:
    profile = scaled(toy(), 1.5, 1.2)
    counts = written(toy(), [("readout", (), {"0": 3960, "1": 40})])
    assert profile.compare(counts).to_dict() == compare(profile, counts).to_dict()
    probe = "import sys, noisevault; print('noisevault.compare' in sys.modules)"
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "False"


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path) -> None:
    kingston, counts = nv.load(KINGSTON), load_counts(EXAMPLE)
    first, second = compare(kingston, counts), compare(kingston, counts)
    assert first.to_dict() == second.to_dict()
    a = first.fitted_profile().save(tmp_path / "a.json").read_bytes()
    b = second.fitted_profile().save(tmp_path / "b.json").read_bytes()
    assert a == b
    fitted = Profile.load(tmp_path / "a.json")
    again = compare(fitted, counts)
    assert (again.gates, again.readout) == (first.gates, first.readout)
    assert fitted.uncorrected().fingerprint == counts.profile.fingerprint


def test_a_gate_factor_that_is_not_identified_has_nothing_to_save() -> None:
    profile = toy()
    xx = PlannedCircuit(name="xx", qubits=(0,), ops=XX)
    neither = compare(profile, simulate(profile, [xx], shots=4000, seed=2, run_at=LATER))
    with pytest.raises(
        NoiseVaultError,
        match="^gate and readout factors are not identified, so there is nothing to save$",
    ):
        neither.fitted_profile()
    readout_only = compare(profile, written(profile, [("readout", (), {"0": 3960, "1": 40})]))
    with pytest.raises(
        NoiseVaultError, match="^the gate factor is not identified, so there is nothing to save$"
    ):
        readout_only.fitted_profile()


def test_a_readout_factor_that_is_not_identified_is_left_out_of_the_fitted_profile() -> None:
    profile = Profile.uniform(
        "quiet", technology="superconducting", num_qubits=1, one_qubit_error=0.002
    )
    circuits = [PlannedCircuit(name="mirror", qubits=(0,), ops=(Op("sx", (0,)),) * 4)]
    result = compare(profile, simulate(profile, circuits, shots=4000, seed=4, run_at=LATER))
    assert result.readout == NoEstimate("the measured qubits state no readout error")
    unmodeled = result.fitted_profile().unmodeled_error
    assert unmodeled.readout is None and unmodeled.gates.factor == pytest.approx(
        result.gates.factor, rel=1e-5
    )


@pytest.mark.parametrize(
    ("estimate", "words"),
    [
        (ErrorFactor(factor=1.84, low=1.54, high=2.12), "x1.84 (95% interval 1.54 to 2.12)"),
        (
            ErrorFactor(factor=0.096, high=0.311, bound="lower"),
            "x0.096 (95% interval, at most 0.311)",
        ),
        (ErrorFactor(factor=20.0, low=11.2, bound="upper"), "x20 (95% interval, at least 11.2)"),
        (
            ErrorFactor(factor=1.00783, low=1.00109, high=1.0146),
            "x1.008 (95% interval 1.001 to 1.015)",
        ),
        (ErrorFactor(factor=1.0, low=1.0, high=1.0004), "x1 (95% interval 1 to 1.0004)"),
        (ErrorFactor(factor=1.0004), "x1"),
    ],
)
def test_an_estimate_reads_with_the_fewest_digits_that_tell_its_values_apart(
    estimate: ErrorFactor, words: str
) -> None:
    assert estimate.describe() == words


def layout(result: Comparison) -> list[str]:
    lines = result.summary(counts_file="run.counts.json").split("\n")
    assert all(len(line) <= 80 for line in lines), max(lines, key=len)
    assert lines[3] == "" and lines[4].startswith("circuit ")
    table_end = lines.index("", 4)
    rows = lines[5:table_end]
    assert len(rows) == len(result.circuits)
    impossible = result.impossible_shots > 0
    assert lines[4].endswith("  impossible shots") == impossible
    for row in rows:
        fitted, noise = (float(v) for v in row.split()[3:5])
        assert row.endswith("  beyond noise") == (not impossible and fitted > noise), row
    block = lines[table_end + 1 :]
    block = block[: block.index("")] if "" in block else block
    labels = {"gate errors", "readout errors", "fit", "next", "note", ""}
    assert all(line[:16].rstrip() in labels and line[16] != " " for line in block), block
    fitted = isinstance(result.gates, ErrorFactor) or isinstance(result.readout, ErrorFactor)
    assert (lines[-3:] == list(fit.NOTE)) == fitted
    return lines


def test_a_good_fit_reads_as_designed() -> None:
    lines = layout(compare(nv.load(KINGSTON), load_counts(EXAMPLE)))
    short = nv.load(KINGSTON).short_fingerprint
    assert lines[0] == f"ibm_kingston@2026-04-15 {short} on qubits 148-149-150-151"
    assert lines[1].startswith("counts run.counts.json, simulated, sha256:")
    assert lines[2] == "run 2026-04-16 09:30Z, 26 h after calibration"
    assert lines[4] == "circuit       shots  profile TVD  fitted TVD  noise TVD 95%"
    assert any(line.startswith("gate errors     x") for line in lines)
    assert any(line.startswith("readout errors  x") for line in lines)
    assert any(line.startswith("fit             within shot noise") for line in lines)


def test_a_poor_fit_puts_the_verdict_first_and_names_the_circuits_below() -> None:
    result = compare(nv.load(KINGSTON), excess_readout(0))
    lines = layout(result)
    flagged = [row.split()[0] for row in lines[5:9] if row.endswith("  beyond noise")]
    verdict = lines.index(f"fit             beyond shot noise (p = {result.p_value:.2g})")
    assert flagged and lines[verdict + 1 : verdict + 3] == [
        f"                on {', '.join(flagged[:-1])} and {flagged[-1]}"
        if len(flagged) > 1
        else f"                on {flagged[0]}",
        "                no one pair of factors fits every circuit",
    ]


def poor(result: Comparison, scores: Sequence[tuple[str, float, float]]) -> Comparison:
    rows = tuple(
        CircuitScore(name, (0, 1), 4000, 0.05, fitted, noise, 0) for name, fitted, noise in scores
    )
    return replace(result, circuits=rows, p_value=0.0025)


def test_the_flag_follows_the_printed_values() -> None:
    result = poor(
        compare(nv.load(KINGSTON), load_counts(EXAMPLE)),
        [("tied", 0.00412, 0.00408), ("over", 0.0050, 0.0041), ("under", 0.0031, 0.0040)],
    )
    lines = layout(result)
    assert lines[5] == "tied      4000       0.0500      0.0041         0.0041"
    assert lines[6].endswith("0.0050         0.0041  beyond noise")
    assert "                on over" in lines


def test_a_node_profile_built_without_revalidation_equals_the_validated_one() -> None:
    kingston = nv.load(KINGSTON)
    built, validated = (
        fit._scaled_without_revalidation(kingston, 1.37, 0.8),
        scaled(kingston, 1.37, 0.8),
    )
    assert built.fingerprint == validated.fingerprint
    assert built.to_dict() == validated.to_dict()
    for c in plan(kingston):
        for q in c.qubits:
            assert built.table.qubit(q) == validated.table.qubit(q)
        for op in c.ops:
            if op.name != "delay":
                qubits = [c.qubits[i] for i in op.qubits]
                assert built.table.gate(op.name, qubits) == validated.table.gate(op.name, qubits)


def test_a_ruled_out_fit_adds_the_impossible_column_and_names_the_qubit() -> None:
    lines = layout(compare(*fez_with_impossible_shots(9)))
    assert (
        lines[4] == "circuit       shots  profile TVD  fitted TVD  noise TVD 95%  impossible shots"
    )
    assert "fit             ruled out (p = 0)" in lines
    assert "                the fit uses the other 15991 shots" in lines


def test_factors_the_counts_do_not_identify_print_no_note() -> None:
    profile = toy()
    counts = simulate(
        profile, [PlannedCircuit(name="xx", qubits=(0,), ops=XX)], shots=4000, seed=2, run_at=LATER
    )
    lines = layout(compare(profile, counts))
    assert lines[0] == f"toy nv:{profile.fingerprint[:12]} on qubit 0"
    assert lines[2] == "run 2026-10-01 12:00Z, calibration age unknown (the profile has no date)"
    assert lines[-6:] == [
        "gate errors     not identified",
        "readout errors  not identified",
        "                gate error and readout error move these counts the same way",
        "fit             not testable (no degrees of freedom left after fitting)",
        "next            add a circuit with no gates, which only readout error moves.",
        "                The circuits from noisevault.counts.plan() include one.",
    ]


if __name__ == "__main__":
    example_counts().save(EXAMPLE)
    print(f"wrote {EXAMPLE.relative_to(ROOT)}")
