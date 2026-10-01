"""The Qiskit export reproduces Aer's NoiseModel.from_backend on real IBM calibrations.

Readout is off on both sides (from_backend symmetrizes it; see test_qiskit.py). Circuits are
transpiled for the fake backend and run as density matrices, so the comparison is exact.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from conftest import FAKES_ADDED_IN, MANILA_V01, migrated, needs_runtime, require

from noisevault.profile import Profile

require("qiskit_aer")
require("qiskit_ibm_runtime")

import qiskit_ibm_runtime.fake_provider as fake_provider  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit import Gate  # noqa: E402
from qiskit.quantum_info import Kraus  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, kraus_error  # noqa: E402

import noisevault as nv  # noqa: E402
from noisevault.frameworks.qiskit import to_qiskit  # noqa: E402

MODERN = ["FakeFez", "FakeTorino", "FakeSherbrooke", "FakeKingston", "FakeManilaV2"]


def profile_from_target(backend: Any) -> Profile:
    """A profile holding exactly what from_backend reads from a backend's Target."""
    target = backend.target
    gates: dict[str, dict] = {}
    records, edges = [], set()
    for name in target.operation_names:
        if not isinstance(target.operation_from_name(name), Gate):
            continue
        if name == "rz":
            gates[name] = {"virtual": True}
            continue
        gates[name] = {}
        for qargs, props in target[name].items():
            if len(qargs) == 2:
                edges.add(qargs)
            record: dict[str, Any] = {"gate": name, "qubits": list(qargs)}
            if props.error is not None and props.error >= 1 - 1 / (2 ** len(qargs) + 1):
                record["disabled"] = True  # IBM marks broken gates with error 1
            elif props.error is not None:
                record["avg_infidelity"] = props.error
                record["duration_ns"] = props.duration * 1e9
            records.append(record)
    qubits = []
    for q, qp in enumerate(target.qubit_properties):
        measure = target["measure"][(q,)]
        qubits.append(
            {
                "index": q,
                "t1_us": None if qp.t1 is None else qp.t1 * 1e6,
                "t2_us": None if qp.t2 is None else qp.t2 * 1e6,
                "readout": {"error": measure.error},
            }
        )
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {
                "name": backend.name,
                "vendor": "ibm",
                "technology": "superconducting",
                "num_qubits": target.num_qubits,
            },
            "connectivity": {"edges": sorted(edges), "directed": True},
            "gates": gates,
            "qubits": qubits,
            "calibrations": records,
        }
    )


def from_noisevault_source(backend: Any) -> Profile:
    try:
        return nv.from_qiskit_backend(backend)
    except NotImplementedError as exc:
        pytest.skip(f"noisevault's IBM source is not available yet ({exc})")


SOURCES = {"target": profile_from_target, "noisevault": from_noisevault_source}


def as_kraus(model: NoiseModel, circuits: list[QuantumCircuit]) -> NoiseModel:
    """The channels ``circuits`` use, written as Kraus maps.

    Aer simulates from_backend's reset-form relaxation errors up to ~3e-9 off the exact
    channel (the Kraus form and noisevault's reference agree to 1e-14), so this is the
    like-for-like comparison. Aer has no public accessor for a model's local errors.
    """
    used = {
        (i.operation.name, tuple(c.find_bit(q).index for q in i.qubits))
        for c in circuits
        for i in c.data
    }
    out = NoiseModel(basis_gates=model.basis_gates)
    for name, by_qubits in model._local_quantum_errors.items():
        for qubits, error in by_qubits.items():
            if (name, tuple(qubits)) in used:
                out.add_quantum_error(kraus_error(Kraus(error).data), name, qubits)
    return out


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def mirror(n: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    half = QuantumCircuit(n)
    for layer in range(2):
        for q in range(n):
            half.rx(rng.uniform(0, 2 * np.pi), q)
            half.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(layer % 2, n - 1, 2):
            half.cx(q, q + 1)
    qc = half.copy()
    qc.barrier()
    return qc.compose(half.inverse())


def probabilities(
    sim: AerSimulator, compiled: list[tuple[QuantumCircuit, list[int]]]
) -> list[np.ndarray]:
    """Exact outcome probabilities on each circuit's layout qubits, in one Aer job."""
    batch = []
    for circuit, layout in compiled:
        circuit = circuit.copy()
        circuit.save_probabilities(layout)
        batch.append(circuit)
    result = sim.run(batch).result()
    return [np.asarray(result.data(i)["probabilities"]) for i in range(len(batch))]


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


@pytest.fixture(scope="module", params=MODERN)
def aer(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Circuits compiled for one fake backend, and from_backend's simulators for them."""
    if request.param in FAKES_ADDED_IN:
        needs_runtime(FAKES_ADDED_IN[request.param])
    backend = getattr(fake_provider, request.param)()
    layouts = profile_from_target(backend).suggest_layout(5)
    compiled = []
    for circuit in (ghz(3), ghz(5), mirror(4, seed=1)):
        layout = [layouts[i] for i in range(circuit.num_qubits)]
        compiled.append(
            (transpile(circuit, backend, initial_layout=layout, seed_transpiler=11), layout)
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NoiseModel.from_backend(backend, readout_error=False)
    kraus = as_kraus(model, [c for c, _ in compiled])
    return {
        "compiled": compiled,
        "raw": AerSimulator(method="density_matrix", noise_model=model),
        "kraus": AerSimulator(method="density_matrix", noise_model=kraus),
        "backend": backend,
    }


@pytest.mark.parametrize("source", SOURCES)
def test_density_matrix_parity_with_from_backend(aer: dict[str, Any], source: str) -> None:
    profile = SOURCES[source](aer["backend"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = to_qiskit(profile, readout=False)
    sim.set_options(method="density_matrix")
    ours = probabilities(sim, aer["compiled"])
    worst = {
        kind: max(map(tvd, ours, probabilities(aer[kind], aer["compiled"])))
        for kind in ("raw", "kraus")
    }
    assert worst["kraus"] <= 1e-9
    assert worst["raw"] <= 1e-8


def test_initial_layout_gets_the_noise_of_those_physical_qubits() -> None:
    # 0.1 keyed noise on circuit indices, so a circuit placed on qubits 3 and 4 ran noiseless.
    backend = fake_provider.FakeManilaV2()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = to_qiskit(migrated(MANILA_V01), readout=False)
        theirs = AerSimulator(
            method="density_matrix",
            noise_model=NoiseModel.from_backend(backend, readout_error=False),
        )
    sim.set_options(method="density_matrix")
    compiled = transpile(ghz(2), backend, initial_layout=[3, 4], seed_transpiler=3)
    [ours] = probabilities(sim, [(compiled, [3, 4])])
    [aers] = probabilities(theirs, [(compiled, [3, 4])])
    ideal = np.array([0.5, 0, 0, 0.5])
    assert tvd(ours, aers) <= 1e-9
    assert tvd(ours, ideal) > 1e-3
