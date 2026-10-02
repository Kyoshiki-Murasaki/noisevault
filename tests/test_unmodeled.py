from __future__ import annotations

import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault import metrics
from noisevault.catalog import ProfileInfo, bundled_profiles
from noisevault.channels import pauli_twirl
from noisevault.check import FRAMEWORKS, check, plan_circuits
from noisevault.conversion import resolve_op
from noisevault.reference import probabilities
from noisevault.report import Report
from noisevault.table import GateNoise

KINGSTON = "ibm_kingston@2026-04-15"
FACTORS = (0, 0.5, 20, 1e3)


@pytest.mark.slow
@pytest.mark.parametrize("info", bundled_profiles(), ids=lambda info: info.id)
def test_every_bundled_profile_stays_physical_at_any_factor(info: ProfileInfo) -> None:
    base = info.load()
    chain, circuits = plan_circuits(base, None, purpose="check")
    loci = {(r.gate, r.qubits) for r in base.calibrations}
    loci |= {(op.name, tuple(chain[q] for q in op.qubits)) for c in circuits for op in c.ops}
    for axis in ("gates", "readout"):
        for factor in FACTORS:
            where = f"{axis} x{factor}"
            scaled = base.model_copy(update={"unmodeled_error": {axis: {"factor": factor}}})
            table = scaled.table
            for name, qubits in loci:
                found = table.gate(name, qubits)
                if isinstance(found, GateNoise) and found.avg_infidelity is not None:
                    limit = metrics.max_avg_infidelity(len(qubits))
                    assert 0 <= found.avg_infidelity <= limit, (where, name, qubits)
            for q in range(scaled.device.num_qubits):
                pair = table.qubit(q).readout
                assert pair is None or 0 <= min(pair) <= max(pair) <= 1, (where, q)
            for circuit in circuits:
                layout = chain[: circuit.num_qubits]
                probs = probabilities(scaled, circuit.ops, circuit.num_qubits, layout=layout)
                assert np.isfinite(probs).all(), (where, circuit.name)
                assert probs.sum() == pytest.approx(1), (where, circuit.name)


def test_every_gate_on_a_bundled_check_chain_twirls_to_a_channel_that_scales() -> None:
    twirls = {}
    for info in bundled_profiles():
        profile = info.load()
        chain, circuits = plan_circuits(profile, None, purpose="check")
        report = Report.start(profile, "reference", None)
        for op in {op for circuit in circuits for op in circuit.ops}:
            wires = tuple(chain[q] for q in op.qubits)
            built = resolve_op(profile.table, op.name, wires, unknown_gates="error", report=report)
            if built.channels:
                twirls[info.id, op.name, wires] = pauli_twirl(built.channels, wires)
    assert len(twirls) == 283
    assert [
        locus
        for locus, pauli in twirls.items()
        if metrics.unscalable("pauli", pauli, len(locus[2]))
    ] == []


def test_every_enabled_kingston_cz_twirls_to_a_channel_that_scales() -> None:
    kingston = nv.load(KINGSTON)
    table = kingston.table
    report = Report.start(kingston, "reference", None)
    twirls = {
        pair: pauli_twirl(
            resolve_op(table, "cz", pair, unknown_gates="error", report=report).channels, pair
        )
        for pair in table.listed_pairs()
        if table.allowed("cz", pair)
    }
    assert len(twirls) == 169
    assert [pair for pair, pauli in twirls.items() if metrics.unscalable("pauli", pauli, 2)] == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_fitted_kingston_passes_check_and_every_report_states_the_factors(
    framework: str,
) -> None:
    require("qiskit_aer" if framework == "qiskit" else framework)
    fitted = nv.load(KINGSTON).model_copy(
        update={"unmodeled_error": {"gates": {"factor": 1.8}, "readout": {"factor": 1.3}}}
    )
    result = check(fitted, frameworks=[framework])
    assert result.passed, result.summary()
    (ran,) = result.frameworks
    assert ran.report.splitlines()[1] == (
        "unmodeled error: gate errors x1.8; readout errors x1.3; T1, T2 and preparation error"
        " are not scaled; readout of qubit 146 is not scaled (no better than chance)"
    )
