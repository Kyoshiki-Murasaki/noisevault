# Pilot methodology

## Canonical channel convention

For every ideal operation, NoiseVault inserts:

1. An `n`-qubit residual depolarizing channel whose strength is solved after accounting for relaxation, so the composed channel targets the archived average gate infidelity.
2. Independent zero-temperature amplitude relaxation and pure dephasing on each participating qubit for the archived gate duration.
3. Asymmetric classical readout confusion after the complete circuit.

For a reported total average infidelity `e`, Hilbert-space dimension `d`, and relaxation fidelity `F_relax`, the residual depolarizing strength is `lambda = d*(e - (1-F_relax))/(d*F_relax - 1)`, clipped to the completely-positive bound. This follows Qiskit Aer's backend-noise construction. If relaxation alone already exceeds the reported error, no depolarization is added and the relaxation floor is retained and reported. An earlier version of this pilot incorrectly assigned the full `e` to depolarization before adding relaxation, which double-counted the source error. That has since been corrected.

For relaxation, amplitude damping supplies the `exp(-t/(2*T1))` coherence component. The residual pure-dephasing rate is `max(0, 1/T2 - 1/(2*T1))`. A phase-flip Kraus channel is selected so its off-diagonal attenuation equals the residual coherence factor.

Partial relaxation data remain usable: missing `T1` means no amplitude damping while a present `T2` still supplies pure dephasing; missing `T2` with present `T1` uses the amplitude-damping coherence limit `T2=2*T1`. Missing gate duration uses the documented 60 ns one-qubit pilot default and is labeled `pilot_default_missing_duration` in channel metadata. A missing directed two-qubit calibration is an error, not a zero-error or relaxation-only gate.

Exact operation names take precedence over documented portable fallbacks (`H/RX → SX/X`, `CX → ECR/CZ`). Ordered two-qubit qargs are never reversed implicitly. Pilot subsets are ordered directed paths with a calibration on every consecutive entangling edge; qubit-count comparisons use prefixes of one nested path.

## Why explicit Kraus channels are used in Experiment B

Framework-native noise attachment can introduce unrelated differences: transpiler basis changes, operation scheduling, readout support, or different meanings for a high-level error parameter. Experiment B therefore constructs the public Qiskit Aer, Cirq, and PennyLane converter objects and asks each framework's independent density-matrix engine to simulate the resulting canonical channel sequence exactly. The independent reference runner expands the mathematical channels directly and is not used by the three framework wrappers.

Experiment A separately asks whether the calibrated schema reconstructs Qiskit Aer's native backend-derived approximation. It uses prefix qargs, exact directed basis calibrations, disabled readout on both sides, state-sensitive basis circuits, and per-gate average-infidelity checks in addition to output-distribution TVD.

## Endianness and readout

NoiseVault's canonical probability order is big-endian logical-wire order: wire `0` is the most significant bit in the displayed bit string. The Qiskit adapter reverses local subsystem axes for multi-qubit Kraus matrices at the Qiskit API boundary, and the runner converts Qiskit's little-endian state indexing back into canonical order. Cirq receives an explicit `qubit_order`, and PennyLane receives an explicit wire order plus order-sensitive predicates for directed operations.

The canonical asymmetric readout matrix is column-stochastic, `M[measured, prepared]`, and is applied as `M @ p`. Qiskit Aer's `ReadoutError` API expects prepared states by row, so the public Aer adapter transposes this matrix only at that boundary. Experiment B and D apply the common exact classical post-processor once after density-matrix simulation.

## Statistical policy

Core comparisons use exact density matrices, so thresholds are not contaminated by shot noise. Any optional hardware or sampled comparison must state shots and an estimated sampling floor separately.

## Success thresholds

- Experiment A: maximum TVD strictly below `0.01` and reconstructed per-gate model targets within `1e-10` absolute error.
- Experiment B: maximum pairwise TVD at or below `0.02` for density-matrix runs.
- Sampled fallback, only when exact simulation is unavailable: `0.05`, with shot count disclosed.

Threshold misses must be reported, not tuned away or hidden.
