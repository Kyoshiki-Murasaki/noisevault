"""Google profiles from the calibrations cirq_google ships."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault import gates, metrics
from noisevault.reference import Op, probabilities

cirq_google = require("cirq_google")
from noisevault.sources import google  # noqa: E402

_PROFILES: dict[str, nv.Profile] = {}


def profile(name: str) -> nv.Profile:
    if name not in _PROFILES:
        _PROFILES[name] = google.from_cirq_google(name)
    return _PROFILES[name]


def _avg(record: nv.profile.CalibrationRecord, arity: int) -> float:
    return metrics.to_avg_infidelity(*record.metric, arity)


def test_willow_pink_reproduces_the_published_spec_sheet() -> None:
    # Willow spec sheet (2024-12-09), QEC chip, "average error": 1Q 0.035% +- 0.029%,
    # CZ 0.33% +- 0.18%, measurement 0.77%, T1 68 +- 13 us (mean +- sd over the chip).
    willow = profile("willow_pink")
    one = np.array([_avg(r, 1) for r in willow.calibrations if r.gate == "r"])
    cz = np.array([_avg(r, 2) for r in willow.calibrations if r.gate == "cz"])
    t1 = np.array([q.t1_us for q in willow.qubits])
    readout = np.array([sum(q.readout.pair) / 2 for q in willow.qubits])
    assert one.mean() == pytest.approx(0.035e-2, abs=0.0005e-2)
    assert cz.mean() == pytest.approx(0.33e-2, abs=0.005e-2)
    assert readout.mean() == pytest.approx(0.77e-2, abs=0.005e-2)
    assert t1.mean() == pytest.approx(68, abs=0.5)
    # the spreads agree to within 5% (the sheet may use a slightly different calibration)
    spreads = (one.std(ddof=1), cz.std(ddof=1), t1.std(ddof=1))
    assert spreads == pytest.approx((0.029e-2, 0.18e-2, 13), rel=0.05)


def test_per_cycle_errors_become_per_gate_by_removing_both_single_qubit_errors() -> None:
    rainbow = profile("rainbow")
    calibration = cirq_google.engine.load_median_device_calibration("rainbow")
    label_of = {q.index: q.label for q in rainbow.qubits}
    one = {r.qubits[0]: r.process_infidelity for r in rainbow.calibrations if r.gate == "r"}
    cycle = {
        tuple(sorted(f"q({q.row},{q.col})" for q in key)): value[0]
        for key, value in calibration[
            "two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle"
        ].items()
    }
    records = [r for r in rainbow.calibrations if r.gate == "sycamore"]
    assert len(records) == len(cycle) == 32
    clamped = 0
    for record in records:
        a, b = record.qubits
        raw = cycle[tuple(sorted((label_of[a], label_of[b])))]
        inferred = raw - one[a] - one[b]
        if inferred < 0:
            clamped += 1
            assert record.process_infidelity == 0
        else:
            assert record.process_infidelity == pytest.approx(inferred, rel=1e-12)
    assert clamped == 2
    assert any(
        "below zero" in note and "q(4,1)-q(4,2)" in note for note in rainbow.provenance.notes
    )


def test_willow_pink_keeps_its_two_qubit_value_and_says_why() -> None:
    willow = profile("willow_pink")
    calibration = cirq_google.engine.load_median_device_calibration("willow_pink")
    raw = sorted(
        v[0] for v in calibration["two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"].values()
    )
    stored = sorted(r.process_infidelity for r in willow.calibrations if r.gate == "cz")
    assert stored == pytest.approx(raw, rel=1e-12)
    assert "not confirmed" in willow.gates["cz"].assumption
    assert "willow_pink" in google.NOT_BUNDLED


def test_bundle_holds_only_resolved_processors() -> None:
    assert [p.id for p in google.bundled_profiles()] == ["google_rainbow", "google_weber"]


def test_qubits_are_grid_qubits_in_row_column_order() -> None:
    rainbow = profile("rainbow")
    coords = [tuple(q.coords) for q in rainbow.qubits]
    assert coords == sorted(coords)
    assert [q.index for q in rainbow.qubits] == list(range(23))
    assert all(
        q.label == f"q({int(r)},{int(c)})" for q, (r, c) in zip(rainbow.qubits, coords, strict=True)
    )
    assert len(rainbow.connectivity.edges) == 32


def test_coherence_and_readout_follow_cirq_google_conventions() -> None:
    willow = profile("willow_pink")
    props = cirq_google.engine.load_device_noise_properties("willow_pink")
    calibration = cirq_google.engine.load_median_device_calibration("willow_pink")
    record = willow.qubits[0]
    grid = require("cirq").GridQubit(*map(int, record.coords))
    t1, tphi = props.t1_ns[grid], props.tphi_ns[grid]
    assert record.t1_us == pytest.approx(t1 / 1000)
    assert record.t2_us == pytest.approx(1 / (1 / (2 * t1) + 1 / tphi) / 1000)
    p00 = calibration["single_qubit_p00_error"][(grid,)][0]
    p11 = calibration["single_qubit_p11_error"][(grid,)][0]
    assert record.readout.pair == pytest.approx((p00, p11))
    assert willow.gates["rz"].virtual
    assert willow.gates["cz"].duration_ns == 42


def test_coherent_fsim_errors_are_carried_and_reported_omitted() -> None:
    willow = profile("willow_pink")
    effects = [e for e in willow.effects if e.type == "coherent_overrotation"]
    assert len(effects) == 182
    assert all(e.allow == "omit" and e.gate == "cz" and 0 <= e.prob < 0.01 for e in effects)
    assert len(willow.extensions["cirq_google"]["fsim_errors"]["cz"]) == 182
    model = willow.to_cirq(unknown_gates="error")
    assert "effect coherent_overrotation on cz" in model.report.omitted


def test_provenance() -> None:
    prov = profile("weber").provenance
    assert (prov.data_kind, prov.source_kind) == ("measured", "package_snapshot")
    assert (prov.license, prov.redistributable) == ("Apache-2.0", "yes")
    assert prov.attribution == "Google Quantum AI (via cirq-google)"
    assert prov.source_hash.startswith("sha256:") and "weber_2021_11_03" in prov.source_url


@pytest.mark.parametrize("name", list(google.PROCESSORS))
def test_every_processor_converts_on_its_natives(name: str) -> None:
    prof = profile(name)
    a, b = prof.connectivity.edges[0]
    for gate in (g for g in prof.gates if prof.table.arity(g) == 2 and g in gates.GATES):
        ops = [Op("r", (0,), (np.pi / 2, 0.3)), Op("rz", (1,), (0.2,)), Op(gate, (0, 1))]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            probs = probabilities(prof, ops, 2, layout=[a, b])
        assert probs.sum() == pytest.approx(1)


def test_sycamore_gets_its_own_calibration_in_cirq() -> None:
    cirq = require("cirq")
    rainbow = profile("rainbow")
    a, b = (
        cirq.GridQubit(*map(int, rainbow.qubits[i].coords)) for i in rainbow.connectivity.edges[0]
    )
    model = rainbow.to_cirq(unknown_gates="error")  # an uncalibrated gate would raise
    circuit = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5)(a), cirq_google.SYC(a, b)
    )
    rho = cirq.DensityMatrixSimulator(noise=model).simulate(circuit).final_density_matrix
    assert np.trace(rho @ rho).real < 0.999  # the gate's calibrated noise was applied


def test_public_api_and_errors() -> None:
    assert nv.from_cirq_google("google_rainbow").fingerprint == profile("rainbow").fingerprint
    with pytest.raises(ValueError, match="rainbow, weber, willow_pink"):
        nv.from_cirq_google("sycamore")


def test_cirq_google_before_1_6_says_to_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    # cirq-google 1.5 has no load_device_noise_properties and no willow_pink calibration
    from cirq_google.engine import virtual_engine_factory as factory

    monkeypatch.delattr(factory, "load_device_noise_properties")
    monkeypatch.setattr(factory, "MEDIAN_CALIBRATIONS", {"rainbow": "x", "weber": "y"})
    for name in ("rainbow", "willow_pink"):
        with pytest.raises(nv.errors.SourceUnavailable, match="cirq-google>=1.6"):
            google.from_cirq_google(name)
