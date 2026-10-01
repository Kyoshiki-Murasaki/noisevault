# Braket fixture

Synthetic device properties with made-up numbers, shaped like documents saved with
`AwsDevice(arn).properties.json()` and validated against the `amazon-braket-schemas` models:

- `iqm_capabilities_v1.json`: IQM-style capabilities with standardized v1, qubits numbered from 1
- `rigetti_standardized_v1.json`: only the standardized v1 part, with directed CNOT records, a
  gap in the qubit numbers and a fidelity type NoiseVault does not know
- `ionq_capabilities_v3.json`: IonQ-style capabilities with device-level standardized v3

They describe no real device.
