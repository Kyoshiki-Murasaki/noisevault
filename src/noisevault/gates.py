"""The gate registry: one row per canonical gate name.

Unitaries are big-endian: ``qubits[0]`` is the most significant tensor factor. Framework
columns are plain identifiers so this module imports no framework; adapters match on them.
Profiles may define gates missing from the registry (their arity then comes from the
profile).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

Unitary = Callable[..., np.ndarray]


@dataclass(frozen=True)
class GateInfo:
    name: str
    arity: int
    params: tuple[str, ...] = ()
    symmetric: bool = False
    family: Literal["z"] | None = None  # free when the profile's rz is virtual
    multi_entangler: bool = False  # needs several native entanglers: never gets typical noise
    unitary: Unitary | None = None  # None for non-unitary operations
    qiskit: str | None = None
    qiskit_class: str | None = None
    cirq: str | None = None  # Cirq gate class name; adapters also match exponents
    pennylane: str | None = None
    stim: tuple[str, ...] = ()


_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_P0 = np.diag([1, 0]).astype(complex)
_P1 = np.diag([0, 1]).astype(complex)
_SQ2 = np.sqrt(0.5)


def _const(matrix: np.ndarray) -> Unitary:
    return lambda: matrix.copy()


def _phase(angle: float) -> np.ndarray:
    return np.diag([1, np.exp(1j * angle)]).astype(complex)


def _controlled(u: np.ndarray) -> np.ndarray:
    return np.kron(_P0, np.eye(len(u))) + np.kron(_P1, u)


def _exp_pauli(pauli: np.ndarray, theta: float) -> np.ndarray:
    """exp(-i theta/2 P) for a Pauli string P (P squared is the identity)."""
    return np.cos(theta / 2) * np.eye(len(pauli)) - 1j * np.sin(theta / 2) * pauli


def _u(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]]
    )


def _r(theta: float, phi: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]])


def _ms(phi0: float = 0.0, phi1: float = 0.0) -> np.ndarray:
    """IonQ Molmer-Sorensen gate with phases in radians; MS(0, 0) = RXX(pi/2)."""
    a, b = np.exp(-1j * (phi0 + phi1)), np.exp(-1j * (phi0 - phi1))
    m = np.array([[1, 0, 0, -1j * a], [0, 1, -1j * b, 0], [0, -1j / b, 1, 0], [-1j / a, 0, 0, 1]])
    return _SQ2 * m


_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
_SQRT_ISWAP = np.array(
    [[1, 0, 0, 0], [0, _SQ2, 1j * _SQ2, 0], [0, 1j * _SQ2, _SQ2, 0], [0, 0, 0, 1]]
)
_ECR = _SQ2 * (np.kron(_X, _I) - np.kron(_Y, _X))
_ZZ, _XX, _YY = np.kron(_Z, _Z), np.kron(_X, _X), np.kron(_Y, _Y)

# fmt: off
_ROWS = (
    GateInfo("id", 1, unitary=_const(_I), qiskit="id", qiskit_class="IGate",
             cirq="IdentityGate", pennylane="Identity", stim=("I",)),
    GateInfo("x", 1, unitary=_const(_X), qiskit="x", qiskit_class="XGate", cirq="XPowGate",
             pennylane="PauliX", stim=("X",)),
    GateInfo("y", 1, unitary=_const(_Y), qiskit="y", qiskit_class="YGate", cirq="YPowGate",
             pennylane="PauliY", stim=("Y",)),
    GateInfo("z", 1, family="z", unitary=_const(_Z), qiskit="z", qiskit_class="ZGate",
             cirq="ZPowGate", pennylane="PauliZ", stim=("Z",)),
    GateInfo("h", 1, unitary=_const(_SQ2 * np.array([[1, 1], [1, -1]], dtype=complex)),
             qiskit="h", qiskit_class="HGate", cirq="HPowGate", pennylane="Hadamard",
             stim=("H",)),
    GateInfo("s", 1, family="z", unitary=_const(_phase(np.pi / 2)), qiskit="s",
             qiskit_class="SGate", cirq="ZPowGate", pennylane="S", stim=("S", "SQRT_Z")),
    GateInfo("sdg", 1, family="z", unitary=_const(_phase(-np.pi / 2)), qiskit="sdg",
             qiskit_class="SdgGate", cirq="ZPowGate", pennylane="Adjoint(S)",
             stim=("S_DAG", "SQRT_Z_DAG")),
    GateInfo("t", 1, family="z", unitary=_const(_phase(np.pi / 4)), qiskit="t",
             qiskit_class="TGate", cirq="ZPowGate", pennylane="T"),
    GateInfo("tdg", 1, family="z", unitary=_const(_phase(-np.pi / 4)), qiskit="tdg",
             qiskit_class="TdgGate", cirq="ZPowGate", pennylane="Adjoint(T)"),
    GateInfo("sx", 1, unitary=_const(_SX), qiskit="sx", qiskit_class="SXGate",
             cirq="XPowGate", pennylane="SX", stim=("SQRT_X",)),
    GateInfo("sxdg", 1, unitary=_const(_SX.conj().T), qiskit="sxdg", qiskit_class="SXdgGate",
             cirq="XPowGate", pennylane="Adjoint(SX)", stim=("SQRT_X_DAG",)),
    GateInfo("rx", 1, ("theta",), unitary=lambda theta: _exp_pauli(_X, theta), qiskit="rx",
             qiskit_class="RXGate", cirq="Rx", pennylane="RX"),
    GateInfo("ry", 1, ("theta",), unitary=lambda theta: _exp_pauli(_Y, theta), qiskit="ry",
             qiskit_class="RYGate", cirq="Ry", pennylane="RY"),
    GateInfo("rz", 1, ("theta",), family="z", unitary=lambda theta: _exp_pauli(_Z, theta),
             qiskit="rz", qiskit_class="RZGate", cirq="Rz", pennylane="RZ"),
    GateInfo("p", 1, ("lam",), family="z", unitary=_phase, qiskit="p",
             qiskit_class="PhaseGate", cirq="ZPowGate", pennylane="PhaseShift"),
    GateInfo("u1", 1, ("lam",), family="z", unitary=_phase, qiskit="u1",
             qiskit_class="U1Gate", pennylane="U1"),
    GateInfo("u", 1, ("theta", "phi", "lam"), unitary=_u, qiskit="u", qiskit_class="UGate",
             pennylane="U3"),
    GateInfo("r", 1, ("theta", "phi"), unitary=_r, qiskit="r", qiskit_class="RGate",
             cirq="PhasedXPowGate"),
    GateInfo("cx", 2, unitary=_const(_controlled(_X)), qiskit="cx", qiskit_class="CXGate",
             cirq="CXPowGate", pennylane="CNOT", stim=("CX", "CNOT", "ZCX")),
    GateInfo("cy", 2, unitary=_const(_controlled(_Y)), qiskit="cy", qiskit_class="CYGate",
             pennylane="CY", stim=("CY", "ZCY")),
    GateInfo("cz", 2, symmetric=True, unitary=_const(_controlled(_Z)), qiskit="cz",
             qiskit_class="CZGate", cirq="CZPowGate", pennylane="CZ", stim=("CZ", "ZCZ")),
    GateInfo("ecr", 2, unitary=_const(_ECR), qiskit="ecr", qiskit_class="ECRGate",
             pennylane="ECR"),
    GateInfo("swap", 2, symmetric=True, multi_entangler=True, unitary=_const(_SWAP),
             qiskit="swap", qiskit_class="SwapGate", cirq="SwapPowGate", pennylane="SWAP",
             stim=("SWAP",)),
    GateInfo("cxswap", 2, multi_entangler=True, unitary=_const(_SWAP @ _controlled(_X)),
             stim=("CXSWAP",)),
    GateInfo("swapcx", 2, multi_entangler=True, unitary=_const(_controlled(_X) @ _SWAP),
             stim=("SWAPCX",)),
    GateInfo("czswap", 2, symmetric=True, multi_entangler=True,
             unitary=_const(_SWAP @ _controlled(_Z)), stim=("CZSWAP", "SWAPCZ")),
    GateInfo("iswap", 2, symmetric=True, unitary=_const(_ISWAP), qiskit="iswap",
             qiskit_class="iSwapGate", cirq="ISwapPowGate", pennylane="ISWAP",
             stim=("ISWAP",)),
    GateInfo("sqrt_iswap", 2, symmetric=True, unitary=_const(_SQRT_ISWAP),
             cirq="ISwapPowGate", pennylane="SISWAP"),
    GateInfo("rzz", 2, ("theta",), symmetric=True, unitary=lambda theta: _exp_pauli(_ZZ, theta),
             qiskit="rzz", qiskit_class="RZZGate", cirq="ZZPowGate", pennylane="IsingZZ"),
    GateInfo("rxx", 2, ("theta",), symmetric=True, unitary=lambda theta: _exp_pauli(_XX, theta),
             qiskit="rxx", qiskit_class="RXXGate", cirq="XXPowGate", pennylane="IsingXX"),
    GateInfo("ryy", 2, ("theta",), symmetric=True, unitary=lambda theta: _exp_pauli(_YY, theta),
             qiskit="ryy", qiskit_class="RYYGate", cirq="YYPowGate", pennylane="IsingYY"),
    GateInfo("zz", 2, symmetric=True, unitary=_const(_exp_pauli(_ZZ, np.pi / 2)),
             cirq="ZZPowGate", stim=("SQRT_ZZ",)),
    GateInfo("ms", 2, ("phi0", "phi1"), symmetric=True, unitary=_ms, cirq="MSGate"),
    GateInfo("cswap", 3, multi_entangler=True, unitary=_const(_controlled(_SWAP)),
             qiskit="cswap", qiskit_class="CSwapGate", cirq="CSwapGate", pennylane="CSWAP"),
    GateInfo("ccx", 3, multi_entangler=True, unitary=_const(_controlled(_controlled(_X))),
             qiskit="ccx", qiskit_class="CCXGate", cirq="CCXPowGate", pennylane="Toffoli"),
    GateInfo("measure", 1, qiskit="measure", qiskit_class="Measure", cirq="MeasurementGate",
             stim=("M", "MZ")),
    GateInfo("reset", 1, qiskit="reset", qiskit_class="Reset", cirq="ResetChannel",
             stim=("R", "RZ")),
    GateInfo("delay", 1, ("duration",), qiskit="delay", qiskit_class="Delay", cirq="WaitGate"),
)
# fmt: on

GATES: Mapping[str, GateInfo] = MappingProxyType({row.name: row for row in _ROWS})


def lookup(name: str) -> GateInfo | None:
    return GATES.get(name)


def unitary(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    info = GATES[name]
    if info.unitary is None:
        raise ValueError(f"{name!r} has no unitary")
    return info.unitary(*params)


def is_symmetric(name: str) -> bool:
    """Registry default for whether a gate's calibration is shared by both operand orders."""
    info = GATES.get(name)
    return info.symmetric if info else False
