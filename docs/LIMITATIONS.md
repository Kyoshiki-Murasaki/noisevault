# Limitations

A NoiseVault model is built from calibration numbers. It approximates the hardware. It is not a
digital twin, and results from it are predictions of a model, not of the device.

## What the model is

Every model is Markovian and local. Each gate is followed by its own channel on its own qubits,
each measurement gets its own readout error, and nothing depends on what happened earlier in the
circuit. The channel for a gate is built from a few numbers: one error metric, a duration, T1
and T2. Two gates with the same numbers get the same channel, whatever the device does
physically.

The numbers come from benchmarks, mostly randomized benchmarking. A benchmark average describes
typical circuits, not any specific one. Errors that depend on the circuit (the gate's angle,
neighboring operations, the time since the last calibration) are not in the numbers.

## What every export leaves out

- **Idle noise outside stated durations.** Qubits relax only during Qiskit `delay`, Cirq
  `WaitGate` and Stim `TICK` layers with `tick_ns`. Time a qubit spends waiting while others
  run gates adds no noise unless you schedule the circuit. PennyLane circuits never get idle
  noise.
- **Crosstalk.** A gate on one pair does not disturb its neighbors. Always-on ZZ coupling,
  spectator errors and measurement-induced disturbance of other qubits are not modeled.
- **Leakage.** Population never leaves the qubit subspace. Where a vendor's error includes
  leakage (Quantinuum), the leakage counts toward the gate's error but acts as ordinary
  depolarizing noise.
- **Coherent errors.** Every gate error becomes stochastic noise. Systematic over-rotations
  add up coherently on hardware and would not in the model. Google's fSim coherent errors are
  recorded but not applied.
- **Effects.** Profiles can record `leakage`, `atom_loss`, `erasure`, crosstalk and coherent
  over-rotation. No export models them in this release. They are listed as omitted in every
  report.
- **Initial-state preparation error.** Circuits start in an ideal |0...0>. Preparation error
  applies only after explicit resets, and only when the profile has it. The bundled IBM
  snapshots have none.
- **Correlated readout.** Each qubit's readout error is independent of the others.

## Approximations inside the exports

- **Relaxation floor.** When relaxation alone exceeds a gate's stated error, the gate keeps the
  relaxation and is noisier than stated. On the bundled `ibm_fez` profile this happens on 82
  loci: `id`, `sx` and `x` on the same 26 qubits, and `cz` on 2 pairs in both directions. The
  largest change is `cz[91, 98]`, from 0.0031 to 0.0039. The report lists every case.
- **T2 above 2 T1** is clamped to 2 T1.
- **Gates the profile does not calibrate** get the noise of the typical native gate by default.
  An un-transpiled circuit then has the wrong gate count. Transpile to native gates for
  realistic results.
- **Double counting of single-qubit noise.** IBM and Quantinuum two-qubit errors come from
  benchmarks that include single-qubit gates. When a circuit's single-qubit gates also get their
  own noise, part of that error counts twice. The extra error is roughly that of the
  single-qubit gates inside the benchmark's two-qubit layer.
- **Stim** applies the Pauli twirl of each channel. Twirling keeps the average fidelity but drops
  relaxation's bias toward |0>. `detector_error_model()` treats Pauli channel components as
  independent, an O(p^2) change. Readout is symmetrized by default.
- **Qiskit** exports `zz` and `ms` natives as `rzz` and `rxx`, which then get the native's noise
  at any angle.

## Limits of the data

- **Device medians.** IonQ publishes only device-wide medians, and Quantinuum's published
  benchmarks pool all gate zones. Every qubit and pair of those profiles gets the same values,
  so per-qubit variation is missing.
- **Unstated metrics.** IonQ does not say which fidelity it reports. NoiseVault reads it as
  average gate fidelity and records that assumption in the profile. If the number is a process
  fidelity, the model's gate errors are too high by a factor (d + 1) / d: 1.5 for one qubit and
  1.25 for two.
- **Old bundled snapshots.** The bundled profiles are fixed past calibrations: Google's from
  November 2021, IBM's from May 2024 to April 2026, Quantinuum's from August 2023 to August
  2025. The devices have been recalibrated since, and some are retired. Pull a current
  calibration when you need one.
- **Google willow_pink** is importable but not bundled, because the scope of its two-qubit error
  values (per cycle or per gate) is unresolved.
- **IBM's public endpoint is undocumented.** IBM can change or remove it without notice. The
  account source (`--source ibm-account`) uses the documented `qiskit-ibm-runtime` API.

## What has been checked, and what has not

The test suite checks that circuits simulated with the Qiskit, Cirq and PennyLane exports match
NoiseVault's reference simulator to a total variation distance of 1e-9, that Stim agrees with
the twirled reference within sampling error, and that the Qiskit export matches Aer's
`NoiseModel.from_backend` on IBM fake backends. `nv check` runs a similar comparison for any
profile.

These checks show the exports implement the model consistently. None of them compares the model
with outcomes measured on hardware. NoiseVault has not yet been validated against hardware
runs, so treat its predictions as estimates whose error against the device is unknown.

## Out of scope

Analog neutral-atom programs (time-dependent Rydberg Hamiltonians) and continuous-variable
photonics do not fit a gate-on-qubit model. Profiles can carry their parameters in
`extensions`, but no export uses them. See
[How technologies map](profile-format.md#how-technologies-map).
