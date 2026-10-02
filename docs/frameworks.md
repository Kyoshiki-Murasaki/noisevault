# Frameworks

Each export turns a profile into an object of the framework's own type, so it plugs into code
you already have. Each one carries a `.report` that lists what the simulator reproduces exactly,
what it approximates, what it leaves out, and what the profile does not know. Print it with
`.report.summary()`, or save `.report.to_dict()` next to your results.

| Framework | Call | Returns | Simulate with |
| --- | --- | --- | --- |
| Qiskit | `profile.to_qiskit()` | `NoiseVaultSimulator`, an `AerSimulator` | `sim.run(transpile(circuit, sim))` |
| Cirq | `profile.to_cirq()` | `NoiseVaultNoiseModel`, a `cirq.NoiseModel` | `cirq.DensityMatrixSimulator(noise=model)` |
| PennyLane | `profile.to_pennylane()` | `NoiseVaultPennyLaneModel`, a `qml.NoiseModel` | `qml.add_noise(qnode, model)` on `default.mixed` |
| Stim | `profile.to_stim(circuit)` | `NoiseVaultStimCircuit`, a `stim.Circuit` | its own samplers, or `detector_error_model()` |

Each framework installs with the extra of the same name: `qiskit`, `cirq`, `pennylane` or
`stim`. The `all` extra installs every one. Importing `noisevault` imports no framework.

```bash
pip install "noisevault[qiskit] @ git+https://github.com/Kyoshiki-Murasaki/noisevault"
```

Every export takes `unknown_gates`. With `"typical"` (the default), a gate the profile does not
calibrate gets the noise of the typical native gate of its arity, with a warning. With
`"error"`, it raises `MissingCalibrationError`. [Conventions](conventions.md) defines that rule,
the channels and the readout matrix.

A gate that the profile does not define takes the calibration of a defined gate that it equals.
`sx` and `x` take `rx`, and `zz` takes `rzz`. The fixed phase gates `z`, `s`, `t` and their
inverses take `p`, which they equal exactly, and else `rz`, which they equal up to a global
phase. `u1` follows the same order, and `p` itself falls back to `rz`. Cirq, PennyLane, Stim and the `nv check` reference share
this rule. Qiskit circuits are compiled to the profile's natives first, so there the Qiskit
transpiler picks the gate.

## Scale the errors a calibration leaves out

A profile can set `unmodeled_error`, which holds factors on its gate and readout error rates.
[Profile format](profile-format.md#unmodeled-error) describes the field. Every export applies
the factors, and so does the reference that `nv check` compares with. To see what twice the
gate error does, copy a profile with a factor of 2:

```python
import stim

import noisevault as nv

fez = nv.load("ibm_fez")
doubled = fez.model_copy(update={"unmodeled_error": {"gates": {"factor": 2.0}}})
code = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)
noisy = doubled.to_stim(code, layout=doubled.suggest_layout(code.num_qubits))
print(noisy.report.summary().splitlines()[1])
# unmodeled error: gate errors x2; T1, T2 and preparation error are not scaled
```

The report states the factors on its second line, and `report.to_dict()` holds the same text
under `unmodeled_error`. The line also names each error that the factors leave as stated. T1,
T2 and preparation error never scale. Neither does an error with no valid power, such as a
readout pair that is no better than chance. The report of a profile without `unmodeled_error`
has no such line.

## Choose qubits

Profiles number physical qubits from 0. Integer circuit qubits map to the same physical qubit
unless you pass a `layout`. To run a small circuit on good qubits, ask the profile for a chain:

```python
import noisevault as nv

fez = nv.load("ibm_fez")
print(fez.suggest_layout(4))
# {0: 136, 1: 143, 2: 142, 3: 141}
```

`suggest_layout(n)` returns a connected chain of n enabled qubits with low summed gate and
readout error. It prefers complete qubits. A complete qubit has every single-qubit native,
apart from the identity, that is usable on at least one qubit of the device. The records that
enable or disable a gate decide where it is usable.
It is a starting point, not a placer. A layout onto a disabled or missing qubit raises
`LayoutError` with the fix.

## Qiskit

`to_qiskit()` builds an Aer simulator with a Qiskit `Target`: every native gate on every
allowed locus, with the error and duration the simulator applies. `transpile(circuit, sim)`
therefore compiles to the device's natives, routes around disabled gates and places circuits by
noise. The noise model is keyed on physical qubits.

```python
from qiskit import QuantumCircuit, transpile

import noisevault as nv

sim = nv.load("ibm_fez").to_qiskit()
ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure_all()
counts = sim.run(transpile(ghz, sim, seed_transpiler=1), shots=2000).result().get_counts()
print(counts)
print(sim.report.summary())
```

`sim.run` accepts only circuits already in native gates on allowed loci. Anything else raises
`CircuitNotNativeError` with the `transpile` call that fixes it. `sim.target`, `sim.noise_model`
and `sim.profile` are available for inspection.

What the report can list:

- **Exact.** Gate noise as an Aer `QuantumError` per native and physical locus. Readout as
  P(1|0) and P(0|1) per qubit with Aer's `ReadoutError`. Thermal relaxation during `delay`.
  A bit flip with the preparation error after each `reset`, when the profile has one.
- **Approximated.** T2 values above 2 T1 are clamped. Natives with no Qiskit instruction of
  their own are exported under the gate that contains them: `zz` as `rzz` and `ms` as `rxx`,
  and the alias gets the native's noise at any angle. Google's `sqrt_iswap` is an instruction of
  its own, but Qiskit's transpiler reaches it only through `cx` (two `sqrt_iswap` each), so a
  general two-qubit block costs six where three would do; the report says so. The initial state
  is ideal, which the report notes when the profile has a preparation error.
- **Omitted.** Idle time outside explicit delays. Transpile with `scheduling_method="alap"` to
  insert delays on idle qubits. Effects. Natives Qiskit cannot target, such as Google's
  `sycamore`, are left out of the simulator and named in the report.
- **Unknown.** Values the profile lacks. The bundled IBM snapshots have no preparation error, so
  resets add none. A `delay` on a qubit with no T1, T2 or dephasing rate adds no noise, and the
  report names the qubit.

`readout=False` leaves measurements noiseless.

On a profile with disabled qubits or gates, transpile with
`initial_layout=list(profile.suggest_layout(n).values())`. Qiskit's `optimization_level=0`
places circuit qubit i on physical qubit i, and levels 1 to 3 do not check that a qubit has the
single-qubit gates a circuit needs. `suggest_layout` does check. It skips a qubit that lacks a
single-qubit native other qubits have, so the chain has the same basis as the rest of the
device. This holds even when the qubit's other gates could make the missing one: an IBM qubit
without `x` is skipped although `rz` and `sx` can make `x`. When no chain of n complete qubits
exists, `suggest_layout` uses as few incomplete ones as it can and warns with the qubits and
their missing gates. The transpiler can fail on those qubits.

## Cirq

`to_cirq()` returns a noise model that follows every gate with its channels as
`cirq.KrausChannel` operations on the same qubits.

```python
import cirq

import noisevault as nv

model = nv.load("quantinuum_h1-1").to_cirq()
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(a),  # the native r gate
    (cirq.ZZ**0.5).on(a, b),  # the native zz gate
    cirq.measure(a, b, key="m"),
)
result = cirq.DensityMatrixSimulator(noise=model, seed=1).run(circuit, repetitions=2000)
print(result.histogram(key="m"))
print(model.report.summary())
```

Qubits map this way: `LineQubit(i)` is device qubit i, and `GridQubit(r, c)` is the qubit whose
`coords` are `[r, c]` (Google profiles record them). Other qubit types need
`layout={qubit: index, ...}`. Gates match by Cirq class and exponent. `cirq.X**0.5` is `sx`,
`cirq.ZZ**0.5` is `zz`, and `cirq.Z**t` is the phase gate `p`, or `s`, `t` or their inverses at
those exponents. `cirq.PhasedXZGate` is split into the `r` gate and `cirq.Z**z`, and each part
gets the noise it would get on its own.

What the report can list:

- **Exact.** Gate noise as Kraus channels after each gate. Readout error, as a `confusion_map`
  on mid-circuit measurements and the equivalent channel before terminal ones. The preparation
  error after each reset. Thermal relaxation during `cirq.WaitGate`.
- **Approximated.** The state after a terminal measurement includes the readout flips. Sampled
  results are exact; to inspect states, build the model with `readout=False`.
- **Omitted.** The preparation error of the initial state, idle time outside `WaitGate`, and the
  profile's leakage effects.

Classically controlled operations raise an error that says how to restructure the circuit.

## PennyLane

`to_pennylane()` returns a `qml.NoiseModel`. Apply it to a QNode on `default.mixed` with
`qml.add_noise`. Gradients flow through the noise channels.

```python
import pennylane as qml

import noisevault as nv

fez = nv.load("ibm_fez")
model = fez.to_pennylane(layout=fez.suggest_layout(2))


@qml.qnode(qml.device("default.mixed", wires=2))
def circuit(theta):
    qml.SX(0)
    qml.RZ(theta, 0)
    qml.SX(0)
    qml.CZ([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


noisy = qml.add_noise(circuit, model)
print(noisy(0.3), qml.grad(noisy)(qml.numpy.array(0.3, requires_grad=True)))
print(model.report.summary())
```

Integer wire i maps to device qubit i. Other wire labels need `layout`. A measurement without
wires, such as `qml.probs()`, gets readout error only on the wires that the circuit's operations
touch. To give every wire readout error, pass `wires=` to the measurement.

What the report can list:

- **Exact.** Gate errors. Readout errors, applied before each measurement in its measured
  basis.
- **Approximated.** The initial state is ideal. State preparation with `BasisState`,
  `StatePrep`, `QubitDensityMatrix`, `AmplitudeEmbedding` or `BasisEmbedding` is noiseless.
  Templates that prepare a state with gates, such as `MottonenStatePreparation`, get noise like
  other templates. `qml.add_noise` at its default `level="user"` noises `qml.adjoint` gates and
  templates through their decomposition; pass `level="top"` to noise `Adjoint(SX)`, `Adjoint(S)`
  and `Adjoint(T)` as the profile's `sxdg`, `sdg` and `tdg`. Operator arithmetic, such as
  `qml.prod`, `@`, `qml.pow`, `qml.exp` or `qml.ctrl`, gets the noise of the gates it
  decomposes into, after the whole operator. An operator with its own gate name, such as
  `qml.CNOT` or `qml.CRX`, is noised as one gate. The basis rotation
  before a Pauli measurement is ideal. Gates conditioned on mid-circuit measurements get their
  noise whether or not the condition holds. A measurement without wires gets readout error on
  the wires that the circuit's operations and measurements use.
- **Omitted.** Idle time, because PennyLane circuits have no timing. Readout on mid-circuit
  measurements. Readout on observables not measured in one product basis. Readout on
  `qml.classical_shadow` and `qml.shadow_expval`, which pick a random measurement basis for each
  shot after the noise model acts. Effects.

Operator arithmetic with no decomposition into gates, such as `qml.sum`, raises an error. It
has no gate noise, and `default.mixed` cannot run it. Before PennyLane 0.45, a QNode leaves
`qml.sum`, `qml.Hamiltonian` and `qml.s_prod` off its tape, so the circuit runs without them
and the model never sees them.

`qml.add_noise` keeps only part of a shot vector's results (`shots=[100, 200]`) when readout
noise is on, so the model raises an error for shot vectors instead of returning wrong numbers.
Run each shot count separately, or pass `readout=False`.

PennyLane has no operation named after the `r`, `zz` and `ms` natives of trapped-ion profiles.
A `qml.Rot(a, theta, -a)` gets the profile's `r` noise and `qml.IsingZZ(pi/2)` its `zz` noise.
On a profile with an `ms` native, `qml.IsingXX(±pi/2)` and `qml.IsingYY(±pi/2)` get its `ms`
noise. Other angles, and gates the profile has no native for, get typical noise with a warning,
and the report counts each use. A broadcast operation gets one noise channel for all its
elements, so the model raises an error when its angles call for different gates' noise, such as
`qml.IsingXX` over `[pi/2, 0.4]`. Expand the broadcast before adding noise:
`qml.add_noise(qml.transforms.broadcast_expand(qnode), model)`.

Every wire a circuit uses, including wires it only measures, is checked against the layout and
the profile, also with `readout=False`. A model built by adding or subtracting noise models
checks only the wires its operations or readout reach.

## Stim

`to_stim(circuit)` returns a copy of a Stim circuit with each gate followed by the Pauli twirl
of its channel, as `PAULI_CHANNEL_1` or `PAULI_CHANNEL_2`, or on three or more qubits as a chain
of `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`. Annotations, `REPEAT` blocks,
detectors and observables pass through, so decoders such as PyMatching work on the result.

```python
import stim

import noisevault as nv

fez = nv.load("ibm_fez")
code = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5)
noisy = fez.to_stim(code, layout=fez.suggest_layout(code.num_qubits))
dem = noisy.detector_error_model()
shots = noisy.compile_detector_sampler(seed=1).sample(10_000)
print(dem.num_errors, shots.mean())
print(noisy.report.summary())
```

`CX` is not a Fez native, so here it gets the noise of `cz` on the same pair, with a warning,
and the report says so. For a grid device, `noisevault.stim.layout_from_coords(circuit, profile)`
places a circuit by matching its `QUBIT_COORDS` to the profile's qubit coords.

In `MPP` and `SPP`, a Pauli product is reduced first. Factors on one qubit multiply, a qubit
whose factors cancel is neither read out nor busy, and a product that reduces to the identity
gets no readout flip.

Options:

- `readout="symmetrize"` (default) flips each result with the mean of P(1|0) and P(0|1).
  `readout="exact"` keeps measurements perfect inside the circuit and stores the asymmetric
  error for `noisevault.stim.sample_with_readout(noisy, shots)`.
  `readout="none"` adds no readout error.
- `tick_ns=` is the duration of one `TICK` layer. Qubits idle in a layer get twirled relaxation
  for that time.
- `existing_noise` decides what happens to noise already in the circuit: `"error"` (default)
  raises, `"keep"` keeps it and `"strip"` removes it.

What the report can list:

- **Exact.** Readout error, only with `readout="exact"` and `sample_with_readout`.
- **Approximated.** Gate noise is the Pauli twirl of each gate's channel, which keeps each
  gate's average fidelity and drops relaxation's bias toward |0>. `detector_error_model()`
  treats the Pauli channel components as independent (Stim's `approximate_disjoint_errors`, on
  by default here). Symmetric readout. Idle noise per `TICK` when `tick_ns` is set.
- **Omitted.** The initial preparation error, idle noise without `tick_ns`, and effects.

Stim simulates Clifford circuits only. Write non-Clifford circuits for one of the other three
frameworks.

## Check a conversion

`nv check REF` runs small circuits through each installed export and compares them with
NoiseVault's own density-matrix reference simulator. Use it after changing a profile by hand.
Stim is sampled with exact readout and compared with the reference after each gate's Pauli
twirl. Its widest circuit and a circuit that only measures are also sampled with the default
symmetrized readout and compared with a reference that uses each qubit's mean readout error.
The measurement-only circuit is needed because readout error leaves a uniform distribution
unchanged. When the profile calibrates `p`, the check also runs a fixed phase gate that the
profile does not define, such as `s`, which every export must charge as `p`.

A framework that cannot express one of a circuit's gates runs the circuit without it, or skips
the circuit when nothing useful is left. The `circuits` column counts only circuits that ran
whole, for example `3 of 4, 1 reduced`. A line under the table names each gate left out and
why, such as `stim: two_qubit_natives ran without rxx, ryy, rzz`. A pass covers only the gates
that ran. `--json` lists the same under each framework's `not_run`, with `ran_without` naming
the gates a reduced circuit left out.
