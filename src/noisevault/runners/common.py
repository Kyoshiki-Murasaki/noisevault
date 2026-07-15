from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    framework: str
    circuit_name: str
    probabilities: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "circuit_name": self.circuit_name,
            "probabilities": [float(x) for x in self.probabilities],
            "metadata": self.metadata,
        }
