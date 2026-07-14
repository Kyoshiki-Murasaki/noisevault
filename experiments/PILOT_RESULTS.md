# NoiseVault Pilot Results

**Overall status:** `OFFLINE_PILOT_VERIFIED_WITH_CALIBRATION_ARTIFACTS`

Run timestamp: `2026-07-14T08:35:46.232203Z`  
Execution mode: `auto` / `full`  
Primary snapshot: `snapshots/qiskit_fake/fake_manila/2024-05-27T15-27-23-03-00.json`  
Data classification: **Packaged Qiskit fake-backend calibrations plus deterministic fixtures; no live hardware claim.**

## Scope and integrity statement

No result is labeled as live hardware evidence unless its snapshot provenance identifies the IBM Quantum Platform. Committed demo snapshots are synthetic fixtures and are retained only so CI and autonomous agents can verify the entire pipeline without credentials.

## Headline numbers

- Snapshots discovered: **3** across **2** backend names.
- Frameworks completed: **qiskit, cirq, pennylane**.
- Benchmark circuits with all requested framework results: **36**.
- Maximum pairwise cross-framework TVD: **0.00000** (threshold 0.02000).
- Native Aer reconstruction maximum TVD: **0.00132** (target 0.01000).
- Drift demonstration simulation TVD: **0.01415**.

## Experiment A — Qiskit native-model reconstruction

Status: `complete`. Backend: `fake_manila`. Maximum TVD: **0.00132**.

- Comparison uses prefix qubits so native and reconstructed qargs align without private NoiseModel remapping.
- Readout error is disabled in both models; Experiment B/D apply readout through the shared exact post-processor.

## Experiment B — Cross-framework agreement

Status: `complete`. Pairwise TVD maximum: **0.00000**; mean: **0.00000**.

The comparison uses the same abstract circuit, the same calibrated parameters, the same canonical Kraus channels, exact density-matrix simulation, and one common asymmetric readout-confusion post-processor. This isolates framework conversion rather than transpiler policy.

![Cross-framework agreement](figures/cross_framework_tvd.png)

## Experiment C — Noise realism sanity check

Status: `complete`. The plotted quantity is classical fidelity to the ideal GHZ distribution as qubit count increases.

![GHZ fidelity](figures/ghz_fidelity.png)

## Experiment D — Versioned drift demonstration

Status: `complete`. Older snapshot: `snapshots/demo/demo_linear_5_2024-01-15.json`. Newer snapshot: `snapshots/demo/demo_linear_5_2026-07-14.json`.

Pair classification: **deterministic archive-semantics demonstration**. Providers: `demo, demo`. Source timestamps: `2024-01-15T09:00:00Z` → `2026-07-14T09:00:00Z`.

The representative circuit distribution changed by TVD **0.01415** between the two distinct snapshot payloads. Only a pair containing live IBM provenance is treated as empirical current-device evidence.

![Calibration drift](figures/drift_comparison.png)

## Validation and failures

- `live_harvest`: IBM Quantum authentication is unavailable. Set IBM_QUANTUM_TOKEN or save an IBM Quantum Platform account.

## Interpretation constraints

Calibration-derived Markovian channels do not capture crosstalk, leakage, non-Markovian effects, coherent calibration errors, or workload-dependent drift. Cross-framework agreement proves converter consistency, not hardware accuracy. A live hardware point is intentionally treated as optional and must never be fabricated when credentials, queue access, or free allocation are unavailable.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pilot]'
python scripts/run_pilot.py --mode auto --profile full
```

Machine-readable results, including per-framework probability vectors, are in `results/pilot_results.json`.
