# NoiseVault Pilot

NoiseVault is a pilot implementation of an open, versioned archive of calibrated quantum-device noise models. A dated JSON snapshot can be validated, subsetted, converted into Qiskit Aer, Cirq, and PennyLane noise representations, and exercised against one common benchmark suite.

This repository is deliberately structured for autonomous execution by Claude Cowork or another coding agent. Begin with [`START_HERE.md`](START_HERE.md). The agent should begin with [`COWORK_TASK.md`](COWORK_TASK.md), run the bootstrap script, repair any dependency/API drift it encounters, execute the complete pilot, and leave a clean final report.

## What is already implemented

- Schema `0.1` with timestamps, calibration provenance, raw-payload hashes, qubit properties, gate properties, coupling maps, and reserved extensions.
- Physicality validation, including probability bounds, operational flags, coupling consistency, and non-fatal `T2 > 2*T1` detection.
- IBM Quantum Platform harvester, Qiskit fake-provider importer, and a defensive calibration-CSV importer.
- Deterministic connected-subset selection for 2–5 qubit experiments.
- A framework-neutral benchmark IR with GHZ, invertible mirror, and RB-style return circuits.
- Canonical noise construction using depolarization, zero-temperature amplitude relaxation, pure dephasing, and asymmetric readout confusion.
- Qiskit Aer, Cirq, and PennyLane converters and exact density-matrix runners.
- Four pilot experiments, figures, threshold checks, machine-readable results, and a Markdown report.
- Synthetic dated fixtures that exercise the entire pipeline when no credentials or network access exist. These fixtures are explicitly labeled and never presented as real device evidence.

## One-command execution

```bash
unzip noisevault-pilot.zip
cd noisevault-pilot
bash scripts/bootstrap.sh
```

The script creates `.venv`, installs the pilot dependencies, runs the tests and linter, attempts live IBM calibration harvesting using existing credentials, imports a packaged fake backend, executes the full experiment suite, and writes:

```text
experiments/PILOT_RESULTS.md
experiments/results/pilot_results.json
experiments/figures/cross_framework_tvd.png
experiments/figures/ghz_fidelity.png
experiments/figures/drift_comparison.png
reports/RESULTS_FOR_INTERPRETATION.md   # created by Cowork after the final run
```

## IBM authentication

The core pilot does not require credentials. For live calibration snapshots, either save an IBM Quantum Platform account using the current `qiskit-ibm-runtime` authentication flow or export:

```bash
export IBM_QUANTUM_TOKEN="..."
export IBM_QUANTUM_INSTANCE="..."   # optional
```

No token is committed. The harvester reads calibration metadata only and does not submit QPU jobs. Hardware execution is outside the core pilot and must remain opt-in.

## Manual commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pilot]'

noisevault doctor
noisevault validate snapshots/demo/demo_linear_5_2026-07-14.json
noisevault list-fakes
noisevault import-fake FakeManilaV2
noisevault harvest-ibm --max-backends 2
noisevault run-pilot --mode auto --profile full
```

For a faster API and pipeline check:

```bash
python scripts/run_pilot.py --mode offline --profile smoke
```

## Python quickstart

```python
from noisevault import load_snapshot
from noisevault.adapters import to_qiskit_aer, to_cirq, to_pennylane
from noisevault.select import best_connected_subset

snapshot = load_snapshot("snapshots/demo/demo_linear_5_2026-07-14.json")
physical_qubits = best_connected_subset(snapshot, 3)

qiskit_conversion = to_qiskit_aer(snapshot, physical_qubits)
cirq_conversion = to_cirq(snapshot, physical_qubits)
pennylane_conversion = to_pennylane(snapshot, physical_qubits)
```

## Experiment definitions

### A. Native Aer reconstruction

A packaged Qiskit fake backend is imported into the NoiseVault schema. Qiskit Aer's native `NoiseModel.from_backend()` is compared with the snapshot-reconstructed model on circuits already expressed in the backend basis. Readout is disabled for this density-matrix comparison. The target maximum TVD is `0.01`.

This experiment intentionally uses prefix qubits because Aer noise models attach errors to absolute qargs. The code does not make undocumented private-object mutations to remap arbitrary physical qubits.

### B. Cross-framework agreement

The same abstract circuits are converted through the exported Qiskit Aer, Cirq, and PennyLane adapters and run in each framework's independent density-matrix engine. Asymmetric readout confusion is applied once through common classical post-processing. The density-matrix target is maximum pairwise TVD `<= 0.02`.

This validates converter consistency. It does not prove that calibration-derived Markovian noise fully predicts hardware.

### C. GHZ degradation

Classical fidelity to the ideal GHZ output distribution is plotted against qubit count for all frameworks. A physically plausible noise model should degrade with circuit size while remaining consistent across converters.

### D. Versioned drift

Two dated snapshots of one backend are compared at the parameter and simulated-distribution levels. The committed pair is synthetic and exists only for deterministic CI. When live or packaged data are available, the runner preferentially selects those pairs and states their provenance in the report.

## Repository layout

```text
src/noisevault/              package code
  adapters/                  framework converters
  providers/                 IBM, fake-provider, and CSV ingestion
  runners/                   exact framework simulations
snapshots/                   the versioned archive
experiments/                 generated report, results, and figures
reports/                     report templates and final handoff
scripts/                     autonomous bootstrap and pilot entry points
tests/                       schema, channel, metric, selection, and runner tests
docs/                        schema, methodology, limitations, and source plan
.github/workflows/           CI and scheduled harvesting templates
.claude/CLAUDE.md            persistent instructions for Claude Cowork
COWORK_TASK.md                autonomous acceptance criteria
```

## Scientific limitations

The schema stores calibration-derived parameters. The pilot's canonical model is Markovian and local: depolarization, `T1/T2` relaxation, and readout confusion. It does not claim to capture coherent errors, crosstalk, leakage, non-Markovian effects, context dependence, pulse-level behavior, or calibration uncertainty. Optional extension fields exist so richer characterizations can be added without silently changing schema semantics.

## License

Apache-2.0.
