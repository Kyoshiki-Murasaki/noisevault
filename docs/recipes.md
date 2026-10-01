# Recipes

Short, task-focused guides. The test suite runs every Python block on this page and checks the
output it shows. The drift recipe needs the network, and the Mitiq recipe needs Mitiq. The
[examples](../examples) folder has longer versions.

## Pin and cite a calibration for a paper

A result simulated under device noise is reproducible only if readers can get the same noise.
Pin the calibration by its fingerprint and publish both.

1. Pull the calibration you want, or pick a bundled one with `nv list`:

   ```bash
   nv pull ibm_fez
   ```

2. Print the citation. It names the source, the calibration time and the full fingerprint:

   ```bash
   nv cite ibm_fez@2025-02-26
   nv cite ibm_fez@2025-02-26 --bibtex
   ```

3. Load the profile by ref and fingerprint in your code, and save the conversion report next to
   your results:

```python
import json

import noisevault as nv

fez = nv.load("ibm_fez@2025-02-26", expect="nv:06404cefa54f")
sim = fez.to_qiskit()
# ... run your circuits ...
with open("noise_manifest.json", "w") as f:
    json.dump(sim.report.to_dict(), f, indent=2)
print(fez.citation())
```

If the profile ever changes, `expect=` raises `FingerprintMismatch` instead of running with
different noise. The manifest records the profile id, fingerprint, framework versions, options
and everything the export approximated.

Readers of a pulled (not bundled) calibration need the profile file itself. Check its terms
with `nv show REF` before you publish it; the fingerprint alone is always shareable.

## Compare drift between two dates

IBM's public endpoint keeps calibration history. Pull two dates and diff them:

```bash
nv pull ibm_fez --at 2026-07-03
nv pull ibm_fez --at 2026-10-01
nv diff ibm_fez@2026-07-02 ibm_fez@2026-09-30
```

`nv pull --at` returns the newest calibration before that time. Here the pulls print
`ibm_fez@2026-07-02T23:05:55Z` and `ibm_fez@2026-09-30T23:14:35Z`, so the diff uses those dates.
For other dates, use the dates your pulls print. The diff lists device medians, the qubits and
pairs that changed most, and gates that were disabled or re-enabled. The start of the output:

```
ibm_fez@2026-07-02 -> ibm_fez@2026-09-30  (90 days later)
device median    before     after  change
T1 (us)           119.6       118   -1.3%
T2 (us)           89.44     92.16   +3.0%
1q error       3.05e-04  3.16e-04   +3.6%
2q error       2.72e-03  2.61e-03   -3.9%
readout error  8.18e-03  1.05e-02  +28.4%

largest changes by qubit
qubit  metric           before     after    change
61     T2 (us)            5.25     98.94  +1784.8%
123    readout error  6.35e-03  1.06e-01  +1565.4%
149    1q error       1.40e-03  1.71e-02  +1121.7%
113    T1 (us)           18.95     155.4   +720.0%
30     T1 (us)           17.59     114.1   +548.6%
...
```

The gate-error and coherence medians moved by 4% or less, and the readout median rose 28%.
Single qubits changed by much more. On qubit 61, T2 went from 5.25 us to 98.94 us. A simulation
pinned to one date can differ from the same device three months later. `nv diff --json` gives
the same data for scripts, and `profile.diff(other)` returns it in Python.
[examples/drift.py](../examples/drift.py) diffs today's calibration against the one from 90
days ago.

## Mitigate errors with Mitiq

A NoiseVault simulator is a noisy backend for Mitiq. The executor transpiles each folded circuit
at `optimization_level=0`, so the transpiler translates the inverse gates that folding adds
without cancelling them.

Mitiq is optional and supports Python up to 3.12. Its Qiskit conversion also needs `ply`:

```bash
pip install "noisevault[qiskit] @ git+https://github.com/Kyoshiki-Murasaki/noisevault" mitiq ply
```

```python
from mitiq import zne
from qiskit import QuantumCircuit, transpile

import noisevault as nv

fez = nv.load("ibm_fez")
sim = fez.to_qiskit()
layout = list(fez.suggest_layout(3).values())


def p000(circuit: QuantumCircuit) -> float:
    measured = circuit.copy()
    measured.measure_all()
    native = transpile(measured, sim, initial_layout=layout, optimization_level=0)
    counts = sim.run(native, shots=20_000, seed_simulator=7).result().get_counts()
    return counts.get("000", 0) / 20_000


half = QuantumCircuit(3)
for _ in range(6):
    for q in range(3):
        half.rz(0.7 * (q + 1), q)
        half.sx(q)
    half.cz(0, 1)
    half.cz(1, 2)
mirror = half.compose(half.inverse())  # ideally returns |000> with probability 1

factory = zne.inference.RichardsonFactory(scale_factors=[1, 2, 3])
mitigated = zne.execute_with_zne(mirror, p000, factory=factory, scale_noise=zne.scaling.fold_global)
print(f"noisy {p000(mirror):.3f}, mitigated {mitigated:.3f}")
# noisy 0.887, mitigated 0.996
```

Richardson extrapolation from scale factors 1, 2 and 3 removes most of the error here.
[examples/mitigation_zne.py](../examples/mitigation_zne.py) runs the same recipe with random
angles.

## Run a QEC memory experiment with Stim and PyMatching

Stim builds the code circuit, NoiseVault adds the device noise, and PyMatching decodes. A
hypothetical square-grid device with `coords` on each qubit lets `layout_from_coords` place the
code by its `QUBIT_COORDS`:

```python
import numpy as np
import pymatching
import stim

import noisevault as nv

side = 7
edges = [(r * side + c, r * side + c + 1) for r in range(side) for c in range(side - 1)]
edges += [(r * side + c, (r + 1) * side + c) for r in range(side - 1) for c in range(side)]
grid = nv.Profile.uniform(
    "grid_49",
    technology="superconducting",
    num_qubits=side * side,
    one_qubit_error=2e-4,
    two_qubit_error=1.5e-3,
    readout_error=5e-3,
    t1_us=100,
    t2_us=80,
    one_qubit_ns=25,
    two_qubit_ns=40,
    connectivity=edges,
)
data = grid.to_dict()
data["qubits"] = [{"index": i, "coords": divmod(i, side)} for i in range(side * side)]
data["prep"] = {"error": 1e-3}
grid = nv.Profile.from_dict(data)

code = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
noisy = grid.to_stim(code, layout=nv.stim.layout_from_coords(code, grid), tick_ns=50)
matching = pymatching.Matching.from_stim_circuit(noisy)
detectors, observables = noisy.compile_detector_sampler(seed=1).sample(
    20_000, separate_observables=True
)
failures = np.sum(matching.decode_batch(detectors)[:, 0] != observables[:, 0])
print(f"logical error rate per 3 rounds: {failures / 20_000:.4f}")
```

The CNOTs in Stim's circuit get the grid's `cx` calibration, and `tick_ns=50` adds relaxation on
idle qubits at each `TICK`. To run on a real chip's topology, use a profile whose qubits record
`coords`, such as `google_weber`, and check that `layout_from_coords` finds a placement.
[examples/qec_surface_code.py](../examples/qec_surface_code.py) compares distances 3 and 5.

## Describe a hypothetical device

`Profile.uniform` gives every 1- and 2-qubit registry gate one error per arity. Z-family gates
such as `s` and `t` are free through a virtual `rz`. The exceptions are `swap`, `cxswap`,
`swapcx` and `czswap`, which each take more than one entangling gate, so decompose them first.
In every other circuit of 1- and 2-qubit gates, each gate gets its own calibrated noise and none
falls back to typical noise:

```python
import noisevault as nv

ions = nv.Profile.uniform(
    "toy-ions",
    technology="trapped_ion",
    num_qubits=12,
    one_qubit_error=3e-5,
    two_qubit_error=1e-3,
    readout_error=2e-3,
    t2_us=1e6,
    one_qubit_ns=10_000,
    two_qubit_ns=200_000,
)
ions.save("toy-ions.json")
print(nv.load("toy-ions.json").short_fingerprint == ions.short_fingerprint)
```

A neutral-atom device in gate mode loses atoms during imaging. Record that as an `atom_loss`
effect. No export models effects yet, so the report lists it as omitted rather than dropping it
silently:

```python
import stim

import noisevault as nv

atoms = nv.Profile.uniform(
    "toy-atoms",
    technology="neutral_atom",
    num_qubits=12,
    one_qubit_error=5e-4,
    two_qubit_error=5e-3,
    readout_error=1e-2,
    t1_us=4e6,
    t2_us=1.5e6,
    one_qubit_ns=500,
    two_qubit_ns=250,
)
atoms = atoms.model_copy(update={"effects": [{"type": "atom_loss", "on": "readout", "prob": 5e-3}]})
noisy = atoms.to_stim(stim.Circuit("H 0\nCX 0 1\nM 0 1"))
print([line for line in noisy.report.summary().splitlines() if line.startswith("omitted")])
# ['omitted: effect atom_loss on readout, initial state preparation error ...']
```

To require that an effect be modeled, set `"allow": "approximate"` or `"exact"` on it. Exports
then raise `UnsupportedEffect` instead of omitting it.

Edit the saved JSON by hand to give qubits or pairs their own values (see
[Profile format](profile-format.md)), then run `nv validate toy-ions.json`.

## Train a PennyLane circuit under device noise

`qml.add_noise` applies the model to a QNode on `default.mixed`. Write the circuit in the
profile's native gates (for IBM: `RZ`, `SX`, `CZ`) so every gate gets its own calibration:

```python
import pennylane as qml
from pennylane import numpy as pnp

import noisevault as nv

fez = nv.load("ibm_fez")
model = fez.to_pennylane(layout=fez.suggest_layout(2))


@qml.qnode(qml.device("default.mixed", wires=2), diff_method="backprop")
def cost(params):
    for wire in (0, 1):
        qml.SX(wire)
        qml.RZ(params[wire], wire)
        qml.SX(wire)
    qml.CZ([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


noisy = qml.add_noise(cost, model)
params = pnp.array([0.4, -0.2], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.3)
for _ in range(20):
    params = opt.step(noisy, params)
print(f"noisy {noisy(params):.4f}, ideal at the same angles {cost(params):.4f}")
```

The noisy minimum stays above -1 because gate and readout errors shrink the expectation value.
[examples/pennylane_gradient.py](../examples/pennylane_gradient.py) trains a 3-qubit version.
