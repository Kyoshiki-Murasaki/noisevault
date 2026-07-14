# Limitations and non-claims

NoiseVault's pilot validates data portability and converter consistency. It does not establish a complete physical model of a quantum processor.

Known omissions include coherent over-rotations, crosstalk, leakage, spectator effects, pulse distortions, non-Markovian dynamics, drift between calibration and workload execution, correlated readout, calibration uncertainty, and workload-dependent compiler behavior.

The IBM importer is defensive against API evolution but cannot guarantee that every provider field survives undocumented upstream changes. Autonomous execution should update public API calls when required, preserve backwards-compatible schema semantics, and record every material change in `reports/AUTONOMOUS_RUN_LOG.md`.

The CSV importer is a fallback because dashboard column names have changed over time. It rejects an unrecognizable file instead of guessing.

A fake-provider snapshot is a real packaged calibration artifact but is not a contemporaneous live-device measurement. A deterministic demo fixture is neither. Reports must retain those distinctions.
