# Pilot methodology

## Canonical channel convention

For every ideal operation, NoiseVault inserts:

1. An `n`-qubit depolarizing channel whose average gate infidelity matches the archived gate error.
2. Independent zero-temperature amplitude relaxation and pure dephasing on each participating qubit for the archived gate duration.
3. Asymmetric classical readout confusion after the complete circuit.

The conversion from average infidelity `e` to depolarizing strength `lambda` uses `lambda = e*d/(d-1)` for Hilbert-space dimension `d`. The channel is represented by a Pauli Kraus decomposition.

For relaxation, amplitude damping supplies the `exp(-t/(2*T1))` coherence component. The residual pure-dephasing rate is `max(0, 1/T2 - 1/(2*T1))`. A phase-flip Kraus channel is selected so its off-diagonal attenuation equals the residual coherence factor.

## Why explicit Kraus channels are used in Experiment B

Framework-native noise attachment can introduce unrelated differences: transpiler basis changes, operation scheduling, readout support, or different meanings for a high-level error parameter. Experiment B therefore materializes one canonical channel sequence and asks each framework to simulate that sequence exactly. This is the correct test of converter agreement.

Experiment A separately asks whether the calibrated schema reconstructs Qiskit Aer's native backend-derived approximation.

## Endianness

NoiseVault's canonical probability order is big-endian logical-wire order: wire `0` is the most significant bit in the displayed bit string. The Qiskit runner explicitly converts Qiskit's little-endian state indexing into this order. Cirq receives an explicit `qubit_order`, and PennyLane receives an explicit wire order.

## Statistical policy

Core comparisons use exact density matrices, so thresholds are not contaminated by shot noise. Any optional hardware or sampled comparison must state shots and an estimated sampling floor separately.

## Success thresholds

- Experiment A: maximum TVD below `0.01`.
- Experiment B: maximum pairwise TVD at or below `0.02` for density-matrix runs.
- Sampled fallback, only when exact simulation is unavailable: `0.05`, with shot count disclosed.

Threshold misses must be reported, not tuned away or hidden.
