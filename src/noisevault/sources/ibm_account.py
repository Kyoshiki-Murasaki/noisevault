"""IBM Quantum calibrations through the user's own account (qiskit-ibm-runtime).

The account is the one saved with ``QiskitRuntimeService.save_account``, or the API key in
``IBM_QUANTUM_TOKEN`` (with ``IBM_QUANTUM_INSTANCE`` when the key sees several instances).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date, datetime
from typing import Any

from ..errors import SourceUnavailable, install_hint
from ..profile import Profile
from .qiskit_backend import (
    as_utc,
    calibration_from_properties,
    now_utc,
    processor_name,
    sha256_bytes,
    to_profile,
)

CHANNEL = "ibm_quantum_platform"
SETUP = (
    "set IBM_QUANTUM_TOKEN to your IBM Quantum API key, or pull without an account with"
    " source='ibm'"
)


def pull(device: str, *, at: str | date | datetime | None = None) -> Profile:
    """The calibration of ``device`` now, or the newest one older than ``at``, via the account.

    Everything comes from ``backend.properties(datetime=at)``, so gates, qubits and readout are
    one snapshot even for a past ``at`` (the backend's Target always describes the present).
    """
    service = _service()
    try:
        backend = service.backend(device)
    except Exception as exc:  # the runtime raises several types for "not visible to you"
        raise SourceUnavailable(
            f"your IBM account cannot open {device} ({exc})",
            hint="list the devices your account can see with QiskitRuntimeService().backends()",
        ) from None
    when = None if at is None else as_utc(at)
    try:
        props = backend.properties(datetime=when)
    except Exception as exc:  # API, protocol and network errors all surface here
        raise SourceUnavailable(
            f"IBM did not return the calibration of {device} ({exc})",
            hint="try again later, or pull without an account with source='ibm'",
        ) from None
    if props is None:
        if at is None:
            raise SourceUnavailable(
                f"IBM returned no calibration for {device}; retired devices have none",
                hint="run nv list to see every profile you can load offline",
            )
        raise SourceUnavailable(
            f"IBM returned no calibration for {device} before {at}", hint="pick a later date"
        )
    data = props.to_dict()
    cal = replace(
        calibration_from_properties(data),
        name=backend.name,
        processor=processor_name(getattr(backend, "processor_type", None)),
    )
    return to_profile(
        cal,
        {
            "data_kind": "measured",
            "source_kind": "account_api",
            "source": f"IBM Quantum account, qiskit-ibm-runtime {_runtime_version()}",
            "attribution": "IBM Quantum",
            "redistributable": "unknown",
            "retrieved_at": now_utc(),
            "source_hash": sha256_bytes(
                json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            ),
        },
    )


def _service() -> Any:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise SourceUnavailable(
            "source='ibm-account' needs qiskit-ibm-runtime", hint=install_hint("ibm")
        ) from None
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    options: dict[str, Any] = {"channel": CHANNEL}
    if token:
        options |= {"token": token, "instance": os.environ.get("IBM_QUANTUM_INSTANCE")}
    try:
        return QiskitRuntimeService(**options)
    except Exception as exc:  # AccountNotFoundError, invalid key, network: all mean "set up"
        raise SourceUnavailable(
            f"could not open your IBM Quantum account ({exc})", hint=SETUP
        ) from None


def _runtime_version() -> str:
    import qiskit_ibm_runtime

    return qiskit_ibm_runtime.__version__


def bundled_profiles() -> list[Profile]:
    """Account data is never bundled: it is yours, under IBM's terms."""
    return []
