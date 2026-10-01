import numpy as np
import pytest
from conftest import require

import noisevault as nv
from noisevault.reference import Op, _apply, probabilities


def _dense_apply(rho, kraus, wires, n):
    """Embed each Kraus operator into the full space with explicit permutations."""
    dim = 2**n
    rest = [q for q in range(n) if q not in wires]
    order = list(wires) + rest
    perm = np.zeros((dim, dim))
    for index in range(dim):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
        permuted = [bits[q] for q in order]
        perm[int("".join(map(str, permuted)), 2), index] = 1
    full = [perm.T @ np.kron(k, np.eye(2 ** len(rest))) @ perm for k in kraus]
    flat = rho.reshape(dim, dim)
    return sum(k @ flat @ k.conj().T for k in full)


@pytest.mark.parametrize("wires", [(0,), (2,), (2, 0), (1, 2), (0, 1)])
def test_apply_matches_dense_embedding(wires):
    rng = np.random.default_rng(len(wires) * 10 + wires[0])
    n = 3
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    d = 2 ** len(wires)
    kraus = [rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for _ in range(3)]
    got = _apply(rho.reshape((2,) * 6), kraus, wires, n).reshape(8, 8)
    assert np.allclose(got, _dense_apply(rho, kraus, list(wires), n))


def test_noiseless_ghz_and_readout_confusion():
    ideal = nv.Profile.uniform(
        "ideal", technology="trapped_ion", num_qubits=3, one_qubit_error=0.0, two_qubit_error=0.0
    )
    ghz = [Op("h", (0,)), Op("cx", (0, 1)), Op("cx", (1, 2))]
    probs = probabilities(ideal, ghz, 3)
    assert probs[0] == pytest.approx(0.5) and probs[7] == pytest.approx(0.5)

    noisy_readout = nv.Profile.uniform(
        "readout",
        technology="trapped_ion",
        num_qubits=1,
        one_qubit_error=0.0,
        two_qubit_error=0.0,
        readout_error=0.03,
    )
    probs = probabilities(noisy_readout, [Op("x", (0,))], 1)
    assert probs == pytest.approx([0.03, 0.97])


def test_matches_aer_from_backend_on_native_circuit():
    require("qiskit_aer")
    require("qiskit_ibm_runtime")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    profile = nv.load("ibm_manila")
    ops = [Op("sx", (0,)), Op("rz", (0,), (0.3,)), Op("x", (1,)), Op("sx", (1,))]
    ops += [Op("cx", (1, 0)), Op("sx", (2,)), Op("cx", (1, 2))]
    ours = probabilities(profile, ops, 3, unknown_gates="error", readout=False)

    qc = QuantumCircuit(3)
    for op in ops:
        getattr(qc, op.name)(*op.params, *op.qubits)
    qc.save_density_matrix()
    noise = NoiseModel.from_backend(FakeManilaV2(), readout_error=False)
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    rho = sim.run(qc).result().data()["density_matrix"]
    aer = DensityMatrix(rho).probabilities()  # little-endian: qubit 0 is the least significant
    aer = aer.reshape((2,) * 3).transpose(2, 1, 0).reshape(8)
    assert 0.5 * np.abs(ours - aer).sum() < 1e-9
