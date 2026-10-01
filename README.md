# NoiseVault

*Real device noise, pinned and portable.*

NoiseVault pins the calibrated noise of a real quantum device, on a date, as one small file.
It turns that file into a working noise model for Qiskit, Cirq, PennyLane or Stim, with a report
of what each simulator reproduced exactly and what it approximated.

Calibration-derived models approximate the hardware. They are not a digital twin.

## Status

Version 0.2 is in development. The profile format 1.0, the offline catalog, profile validation
and hypothetical-device profiles work today. The Qiskit, Cirq, PennyLane and Stim exports are
being built. Until they land, `to_qiskit()`, `to_cirq()`, `to_pennylane()` and `to_stim()` raise
`NotImplementedError`.

## Install

From a checkout, with every optional framework and the test tools:

```bash
pip install -e '.[dev]'
```

The core package needs only numpy, pydantic, typer and rich. Each framework is an extra:
`qiskit`, `ibm`, `cirq`, `pennylane`, `stim`, `google`, or `all`.

## Use

```python
import noisevault as nv

manila = nv.load("ibm_manila")  # bundled, works offline
print(manila.summary())
print(manila.citation())
print(manila.suggest_layout(3))

ions = nv.Profile.uniform(
    "toy-ions", technology="trapped_ion", num_qubits=20,
    one_qubit_error=3e-5, two_qubit_error=1e-3, readout_error=2e-3,
)
ions.save("toy-ions.json")
print(ions.id, ions.short_fingerprint)
```

A profile's fingerprint depends only on its physics. Pin it when you load a profile, and a later
change to that profile fails loudly: `nv.load("ibm_manila", expect="nv:fc7183858c84")`.

From the command line (`nv` is a short alias of `noisevault`):

```bash
noisevault validate toy-ions.json
noisevault doctor
```

## License

Apache-2.0. Bundled calibration data keeps the license and attribution of its source; see
[NOTICE](NOTICE).
