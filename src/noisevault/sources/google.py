"""Google processors from the calibrations cirq_google ships, read the way cirq_google reads them.

cirq_google bundles one representative calibration per virtual processor and converts it to
noise with ``noise_properties_from_calibration``. This module takes T1, Tphi, readout errors,
gate times and fSim coherent errors from that conversion, and the per-gate errors from the
calibration itself, since their scope needs care:

- 1Q ``single_qubit_rb_pauli_error_per_gate``: randomized benchmarking Pauli error per
  PhasedXZ gate, stored as process infidelity (Pauli error = process infidelity).
- 2Q ``..._xeb_pauli_error_per_cycle``: Cirq's calibration docs define an XEB cycle as a random
  single-qubit gate on each qubit followed by the entangler, and say the value "is the error rate
  per cycle (both the 1 qubit gates as well as the 2 qubit gate)". Google's inferred per-gate
  error subtracts the single-qubit RB contribution. rainbow and weber are converted that way.
  willow_pink carries the same key, but its values reproduce Google's published per-gate CZ
  error for that chip (below), so its scope is contradictory: it is read as per gate, stated in
  the gate's ``assumption``, and not bundled.

cirq_google's own noise model applies the per-cycle value to the entangler as is; its legacy
rainbow and weber model also charges Z gates 25 ns and the 1Q error, while the device spec gives
Z gates 0 ns (virtual), which is what the profile uses.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from .. import __version__
from ..errors import SourceUnavailable, install_hint
from ..profile import Profile

if TYPE_CHECKING:
    import cirq  # noqa: F401


@dataclass(frozen=True)
class GoogleProcessor:
    chip: str
    two_qubit_scope: Literal["cycle", "unresolved"]
    one_qubit_measured: Literal["isolated"] | None  # the legacy docs say isolated RB


PROCESSORS: Mapping[str, GoogleProcessor] = {
    "rainbow": GoogleProcessor("Sycamore", "cycle", "isolated"),
    "weber": GoogleProcessor("Sycamore", "cycle", "isolated"),
    "willow_pink": GoogleProcessor("Willow", "unresolved", None),
}
NOT_BUNDLED = {
    name: "its 2Q XEB values are labelled per cycle but match Google's published per-gate CZ"
    " error, so their scope is unresolved"
    for name, spec in PROCESSORS.items()
    if spec.two_qubit_scope == "unresolved"
}
# cirq gate class -> canonical name, for the entanglers cirq_google has calibration metrics for
_TWO_QUBIT = {"SycamoreGate": "sycamore", "ISwapPowGate": "sqrt_iswap", "CZPowGate": "cz"}
_ONE_QUBIT_TIME_GATE = "PhasedXZGate"
_TWO_QUBIT_ASSUMPTION = {
    "cycle": "Google XEB Pauli error per cycle (a random single-qubit gate on each qubit, then"
    " the entangler), minus both qubits' 1Q RB Pauli errors as Google infers its per-gate"
    " error; stored as the total process infidelity, including the coherent fSim part",
    "unresolved": "Google labels this XEB Pauli error per cycle, but for willow_pink the values"
    " reproduce Google's published per-gate CZ error (Willow spec sheet, QEC chip, 0.33% +-"
    " 0.18% average error = 4/5 of the file's mean and spread), so it is read unchanged as the"
    " per-gate process infidelity, as cirq_google's own noise model does; the scope is not"
    " confirmed",
}


def bundled_profiles() -> list[Profile]:
    """rainbow and weber; willow_pink is left out while its 2Q error scope is unresolved."""
    return [from_cirq_google(name) for name in PROCESSORS if name not in NOT_BUNDLED]


def from_cirq_google(processor_id: str) -> Profile:
    """The profile of a cirq_google virtual processor (``rainbow``, ``weber``, ``willow_pink``)."""
    name = processor_id.strip().lower().removeprefix("google_")
    if name not in PROCESSORS:
        raise ValueError(
            f"unknown Google processor {processor_id!r}; choose one of {', '.join(PROCESSORS)}"
        )
    try:
        import cirq_google
        from cirq_google.engine import virtual_engine_factory as factory
    except ImportError:
        raise SourceUnavailable(
            "cirq_google is not installed", hint=install_hint("google")
        ) from None
    # both arrived in cirq-google 1.6; 1.5 has neither, and no willow_pink calibration
    if not hasattr(factory, "load_device_noise_properties") or name not in getattr(
        factory, "MEDIAN_CALIBRATIONS", {}
    ):
        raise SourceUnavailable(
            f"cirq-google {cirq_google.__version__} has no {name} calibration with a noise"
            " conversion",
            hint="pip install -U 'cirq-google>=1.6'",
        )
    calibration = factory.load_median_device_calibration(name)
    properties = factory.load_device_noise_properties(name)
    device = factory.create_device_from_processor_id(name)
    file_name = factory.MEDIAN_CALIBRATIONS[name]
    raw = (resources.files("cirq_google.devices.calibrations") / file_name).read_bytes()
    return _profile(
        name,
        PROCESSORS[name],
        calibration,
        properties,
        device,
        source=f"cirq-google {cirq_google.__version__} {file_name}",
        source_url="https://github.com/quantumlib/Cirq/blob/"
        f"v{cirq_google.__version__}/cirq-google/cirq_google/devices/calibrations/{file_name}",
        source_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _profile(
    name: str,
    spec: GoogleProcessor,
    calibration: Any,
    properties: Any,
    device: Any,
    *,
    source: str,
    source_url: str,
    source_hash: str,
) -> Profile:
    qubits = sorted(device.metadata.qubit_set)
    index = {q: i for i, q in enumerate(qubits)}
    times = {gate.__name__: float(ns) for gate, ns in properties.gate_times_ns.items()}
    one_qubit = _per_qubit(calibration, "single_qubit_rb_pauli_error_per_gate")
    notes: list[str] = []

    gates: dict[str, dict[str, Any]] = {
        "rz": {"virtual": True},
        "r": {
            "duration_ns": times[_ONE_QUBIT_TIME_GATE],
            "method": "rb",
            "measured": spec.one_qubit_measured,
            "statistic": "individual",
            "assumption": "Google 1Q RB Pauli error per PhasedXZ gate, stored as process"
            " infidelity; it is the gate's total error",
        },
    }
    records = [
        {"gate": "r", "qubits": [index[q]], "process_infidelity": value}
        for q, value in one_qubit.items()
    ]
    entanglers, pair_records = _entanglers(calibration, spec, index, one_qubit, times, notes)
    gates |= entanglers
    records += pair_records
    effects, fsim = _coherent_effects(properties.fsim_errors, index)
    qubit_records = [_qubit(q, index[q], properties) for q in qubits]
    edges = sorted(sorted((index[a], index[b])) for a, b in device.metadata.qubit_pairs)
    if "sycamore" in gates:
        notes.append(
            "sycamore is Google's Sycamore gate, FSim(pi/2, pi/6); the gate registry does not list"
            " it, so the Cirq export matches it by class and other exports report it omitted"
        )
    if "sqrt_iswap" in gates:
        notes.append(
            "sqrt_iswap holds cirq_google's sqrt_iswap metrics, which cirq_google applies to"
            " every ISwapPowGate, including ISWAP**-0.5"
        )
    return Profile.model_validate(
        {
            "noisevault": "1.0",
            "device": {
                "vendor": "google",
                "name": name,
                "technology": "superconducting",
                "num_qubits": len(qubits),
                "processor": spec.chip,
                "calibrated_at": datetime.fromtimestamp(calibration.timestamp / 1000, UTC),
            },
            "connectivity": {"edges": edges, "directed": False},
            "gates": gates,
            "qubits": qubit_records,
            "calibrations": records,
            "effects": effects,
            "provenance": {
                "data_kind": "measured",
                "source_kind": "package_snapshot",
                "source": source,
                "source_url": source_url,
                "license": "Apache-2.0",
                "attribution": "Google Quantum AI (via cirq-google)",
                "redistributable": "yes",
                "retrieved_at": datetime.now(UTC),
                "source_hash": source_hash,
                "tool": f"noisevault {__version__}",
                "notes": [
                    "cirq_google describes its bundled calibrations as roughly representative"
                    " of the chip's median performance",
                    "T1 is single_qubit_idle_t1_micros; T2 is 1/(1/(2 T1) + 1/Tphi) with the"
                    " Tphi cirq_google infers from the RB incoherent error (Tphi = 1e10 ns"
                    " where it finds no dephasing)",
                    "readout p1_given_0 and p0_given_1 are single_qubit_p00_error and"
                    " single_qubit_p11_error",
                    "Z rotations are virtual: the device spec gives ZPowGate 0 ns",
                    "coherent_overrotation effects: prob is 1 - F_e of cirq_google's fSim error"
                    " unitary for that pair (the part its model applies coherently), to 12"
                    " significant digits; the gate error already includes it; the angles are in"
                    " extensions.cirq_google.fsim_errors",
                    *notes,
                ],
            },
            "extensions": {"cirq_google": {"fsim_errors": fsim}},
        }
    )


def _entanglers(
    calibration: Any,
    spec: GoogleProcessor,
    index: Mapping[Any, int],
    one_qubit: Mapping[Any, float],
    times: Mapping[str, float],
    notes: list[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Definitions and per-pair records of every entangler the calibration has XEB data for."""
    from cirq_google.engine.calibration_to_noise_properties import GATE_PREFIX_PAIRS

    defs: dict[str, dict[str, Any]] = {}
    records = []
    for gate_type, prefix in GATE_PREFIX_PAIRS.items():
        cycle = _per_pair(calibration, f"{prefix}_xeb_pauli_error_per_cycle")
        if not cycle:
            continue
        canonical = _TWO_QUBIT[gate_type.__name__]
        defs[canonical] = {
            "qubits": 2,
            "symmetric": True,
            "duration_ns": times[gate_type.__name__],
            "method": "xeb",
            "measured": "simultaneous",
            "statistic": "individual",
            "scope": "gate",
            "assumption": _TWO_QUBIT_ASSUMPTION[spec.two_qubit_scope],
        }
        clamped = []
        for (a, b), error in cycle.items():
            if spec.two_qubit_scope == "cycle":
                error -= one_qubit[a] + one_qubit[b]
                if error < 0:
                    clamped.append(f"{_label(a)}-{_label(b)} ({error:.2g})")
                    error = 0.0
            records.append(
                {"gate": canonical, "qubits": [index[a], index[b]], "process_infidelity": error}
            )
        if clamped:
            notes.append(
                f"{canonical}: the inferred per-gate error came out below zero on"
                f" {', '.join(clamped)} (1Q RB is isolated, XEB parallel); set to 0"
            )
    return defs, records


def _qubit(qubit: cirq.GridQubit, index: int, properties: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": index,
        "label": _label(qubit),
        "coords": [qubit.row, qubit.col],
    }
    t1 = properties.t1_ns.get(qubit)
    tphi = properties.tphi_ns.get(qubit)
    if t1 is not None:
        record["t1_us"] = t1 / 1000
        if tphi is not None:
            record["t2_us"] = 1 / (1 / (2 * t1) + 1 / tphi) / 1000
    readout = properties.readout_errors.get(qubit)
    if readout is not None:
        record["readout"] = {"p1_given_0": float(readout[0]), "p0_given_1": float(readout[1])}
    return record


def _coherent_effects(
    fsim_errors: Mapping[Any, Any], index: Mapping[Any, int]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    """One omitted coherent_overrotation effect per gate and pair, and the fitted angles."""
    effects: list[dict[str, Any]] = []
    angles: dict[str, dict[str, dict[str, float]]] = {}
    seen: set[tuple[str, tuple[int, int]]] = set()
    for op_id, error_gate in fsim_errors.items():
        canonical = _TWO_QUBIT[op_id.gate_type.__name__]
        a, b = sorted(op_id.qubits, key=lambda q: index[q])
        pair = (index[a], index[b])
        if (canonical, pair) in seen:
            continue
        seen.add((canonical, pair))
        infidelity = float(f"{_fsim_infidelity(error_gate):.12g}")
        effects.append(
            {"type": "coherent_overrotation", "gate": canonical, "qubits": list(pair),
             "prob": infidelity}
        )  # fmt: skip
        angles.setdefault(canonical, {})[f"{_label(a)}-{_label(b)}"] = {
            key: float(getattr(error_gate, key)) for key in ("theta", "zeta", "chi", "gamma", "phi")
        }
    return effects, angles


def _fsim_infidelity(gate: cirq.PhasedFSimGate) -> float:
    """1 - |tr U|^2/16 of a PhasedFSimGate, whose diagonal is 1, cos(theta) e^{-i(gamma +- zeta)}
    and e^{-i(2 gamma + phi)}.

    Decimal arithmetic gives the same digits on every platform. libm's sin, cos and exp can
    differ in the last ulp, and the cancellation in 1 - |tr U|^2/16 lifts that into the digits a
    profile stores; at 40 digits over 25 survive it.
    """
    with localcontext(prec=40):
        theta, zeta, gamma, phi = (
            Decimal(x) for x in (gate.theta, gate.zeta, gate.gamma, gate.phi)
        )
        middle = 2 * _cos(theta) * _cos(zeta)
        real = 1 + middle * _cos(gamma) + _cos(2 * gamma + phi)
        imag = middle * _sin(gamma) + _sin(2 * gamma + phi)
        return float(1 - (real * real + imag * imag) / 16)


def _sin(x: Decimal) -> Decimal:
    return _taylor(x, x, 1)


def _cos(x: Decimal) -> Decimal:
    return _taylor(x, Decimal(1), 0)


def _taylor(x: Decimal, term: Decimal, n: int) -> Decimal:
    """The sine (term x, n 1) or cosine (term 1, n 0) series, summed until it stops changing."""
    total, last = term, None
    while total != last:
        last = total
        term *= -x * x / ((n + 1) * (n + 2))
        n += 2
        total += term
    return total


def _per_qubit(calibration: Any, metric: str) -> dict[Any, float]:
    if metric not in calibration:
        return {}
    return {
        calibration.key_to_qubit(key): calibration.value_to_float(value)
        for key, value in calibration[metric].items()
    }


def _per_pair(calibration: Any, metric: str) -> dict[tuple[Any, Any], float]:
    """Per unordered pair (cirq_google lists each pair once, in either order)."""
    if metric not in calibration:
        return {}
    out: dict[tuple[Any, Any], float] = {}
    for key, value in calibration[metric].items():
        a, b = calibration.key_to_qubits(key)
        pair = (a, b) if a < b else (b, a)
        if pair in out:
            raise ValueError(f"{metric} lists the pair {_label(a)}-{_label(b)} twice")
        out[pair] = calibration.value_to_float(value)
    return out


def _label(qubit: Any) -> str:
    return f"q({qubit.row},{qubit.col})"
