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
  largest change is `cz[91, 98]`, from 0.0031 to 0.0039. The report lists every case. A stated
  error too large for any channel to reach (far above real calibrations) is recorded the same
  way, with the achieved error below the stated one.
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
with outcomes measured on hardware. `nv compare` makes that comparison for counts you measure.
It scores a profile on them and fits how far its gate and readout error rates must scale to
match the device. [Measure a profile against hardware](recipes.md#measure-a-profile-against-hardware)
shows how. [How nv compare fits the factors](#how-nv-compare-fits-the-factors) explains the fit and
how its intervals were tested.

NoiseVault has not yet been validated against hardware runs. No counts from a real device have
been compared, so treat its predictions as estimates whose error against the device is unknown.

## How nv compare fits the factors

`nv compare` reports a gate factor and a readout factor, each with a 95% interval, and a p-value
for the fit. How far to trust those numbers depends on how the fit computes them and on what the
tests have shown.

### Which counts it accepts

`nv compare` refuses counts that name another device or another calibration fingerprint, that
ran before the calibration, or that use an op the profile does not calibrate on the measured
qubits. The fit starts from the calibration without `unmodeled_error`. The factors it reports
therefore multiply the stated error rates, even when the profile already carries factors.

### The likelihood

The fit is a multinomial maximum likelihood over all circuits at once, with one factor on every
gate error rate and one on every readout error rate. It searches each factor from 0.05 to 20.
NoiseVault's reference simulator computes the outcome probabilities on the profile with the
candidate factors set in `unmodeled_error`. That is the model every export applies, and
`nv check` tests each export against the same reference. A factor therefore means the same thing
in the fit as in a saved profile.

The reference runs exactly at 25 gate factors, evenly spaced in log factor, and at each factor
where a measured gate's scaled error equals its relaxation floor. Between those runs, the
probabilities are linear in log factor, so they stay between 0 and 1. The fit applies readout
error exactly at every point. The deviance, the fitted TVDs and the counts drawn for the p-value all
come from an exact run at the fitted factors.

### Which factors the counts identify

Gate error and readout error can move counts the same way. The `readout` circuit from `plan()` has
no gates, so only readout error moves it, and that circuit separates the two factors.

`nv compare` reports a factor as not identified when no circuit moves with it, when fewer than
100 shots fall in circuits that move with it, or when its interval reaches both ends of the
search range. It reports both factors as not identified when they move the counts in the same
direction, that is, when the smallest eigenvalue of the Fisher information is below 1e-6 of the
largest. A factor whose interval reaches one end of the range keeps a one-sided interval, printed
with "at most" or "at least".

### The goodness-of-fit test

The deviance is twice the gap between the log-likelihood of the counts' own frequencies and
that of the fitted model. The test compares the deviance with its degrees of freedom. Those are
the outcomes the profile can produce at some factor in the range, minus one for each circuit,
minus the rank of the Fisher information. That rank is 2 when the counts determine both factors,
and less when they do not.

The p-value ranks the deviance among the deviances of 400 sets of counts drawn from the fitted
model, so the smallest p it can report is 1/401, about 0.0025. Below 0.01, `nv compare` reports
the fit as beyond shot noise, which means that no one pair of factors fits every circuit. With no
degrees of freedom left, it reports the fit as not testable and gives no p-value.

The `noise TVD 95%` column comes from the same draws. It is the 95th percentile of the TVD
between each drawn set and the model refitted to it, and `nv compare` flags each circuit whose
fitted TVD exceeds it. The draws are seeded from the SHA-256 of the counts, so the same profile
and counts always give the same output.

### The intervals

Each 95% interval holds the factor values that a likelihood-ratio test does not reject at the 5%
level, with the other factor refitted at each value. The starting cutoff is the chi-square value
3.84. When the deviance exceeds its degrees of freedom, the cutoff is multiplied by their ratio,
the dispersion, so a fit that misses by more than shot noise gets wider intervals.

The chi-square cutoff is a large-sample approximation, and it covers too little when a factor
rests on a few error shots. On one qubit with a readout error of 0.00055 and 4000 shots, about
two shots read wrong, and intervals from the chi-square cutoff alone contain the true factor with
probability 0.86. NoiseVault therefore checks each end of an interval again by simulation. It
draws 400 sets of counts from the model at that end, with the same shots per circuit, and
computes the likelihood-ratio statistic on each. The end passes when at least 5% of the drawn
statistics are as large as the observed one. The end moves outward while it passes, and
NoiseVault locates the outermost passing end to within 2% of the chi-square half-width. A value
inside the chi-square cutoff always passes, so this step can only widen an interval. In the
one-qubit case, the probability rises to 0.975.

### Outcomes the profile rules out

An outcome can have probability 0 at every factor, as when the profile states a readout error of
exactly 0 for a measured qubit. No pair of factors explains a shot on such an outcome, so those
shots leave the likelihood, and the fit uses the other shots. The shots still count in the TVDs,
and they set p to 0, so the fit reads `ruled out`. When the profile rules out every shot, neither
factor is identified.

### What the tests show, and what they do not

A CI job simulates 100 runs of the ibm_kingston plan at 4000 shots per circuit, with gate errors
x1.8 and readout errors x1.3. It requires each interval to contain its true factor in 88 to 100
of the runs. The gate interval contains the true gate factor in 93 runs, and the readout
interval contains the true readout factor in 95. To run the job locally, set
`NOISEVAULT_SLOW=1` and run `pytest tests/test_compare.py -k each_interval_covers`.

Two faster tests run with the rest of the suite. In the one-qubit case above, the probability
must be at least 0.95. On one qubit with a readout error of 0.1 and a million shots, the interval
must contain the true factor in 180 to 199 of 200 runs.

Each of these tests draws its counts from the model it fits, and the two-factor test covers one
device at one pair of true factors. The tests show that the intervals cover factors the model can
express. They do not show how the factors behave when the device differs from the model in a way
that no pair of factors captures. On such counts, the goodness-of-fit test is the check, and an
interval describes the best pair of factors, not the device. No counts from a real device have
been compared yet.

## What the unmodeled-error factors absorb

`nv compare` fits one factor on every gate error rate and one on every readout error rate. The
gate factor absorbs any error beyond the calibration that acts like more gate error: crosstalk,
leakage, coherent error, idle error beyond T1 and T2, and drift since the calibration. Profiles
that state no durations or no T1 and T2, such as the bundled Quantinuum ones, also put their
idle and transport error into the gate factor.

- **One chain represents the device.** `nv compare` fits the factors on the three or four qubits
  that `nv check` picks, a well-calibrated chain. A saved profile applies them to every qubit, and
  other qubits can have more or less excess error. `fit.qubits` in the profile names the
  measured qubits.
- **One gate factor covers every gate.** The check circuits cannot separate excess error on
  one-qubit gates from excess error on two-qubit gates, so one factor scales both. The fitted
  value averages the two kinds by the error each contributes to the check circuits. Two-qubit
  gates carry 63% of the stated gate error in those circuits on the bundled ibm_kingston, 66% on
  ibm_fez and 94% on quantinuum_h2-1. If the excess sits mostly in one kind, a circuit with a
  different mix gets too much or too little error.
- **Short circuits, stochastic error.** On IBM Heron devices each check circuit runs for less
  than 1 µs before measurement, with no spectator qubits, no parallel layers and no mid-circuit
  measurement. The factors scale stochastic error, so coherent error that builds up over many
  gates on the device stays stochastic in the model. Whether a factor fitted on these circuits
  predicts the logical error rate of a QEC circuit is untested.
- **Some values never scale.** T1, T2, dephasing, preparation error, gate durations and effects
  keep their stated values. An error with no valid power, such as a readout pair no better than
  chance or a Pauli channel with a negative Pauli-Lindblad rate, also stays as stated, and
  `nv compare` and every report name it.

## Out of scope

Analog neutral-atom programs (time-dependent Rydberg Hamiltonians) and continuous-variable
photonics do not fit a gate-on-qubit model. Profiles can carry their parameters in
`extensions`, but no export uses them. See
[How technologies map](profile-format.md#how-technologies-map).
