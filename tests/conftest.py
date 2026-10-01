"""Shared test helpers.

Optional frameworks are imported through ``require``: a missing one skips the test, unless
NOISEVAULT_REQUIRE_ALL=1, which turns the skip into a failure so a full environment cannot
silently skip coverage.
"""

from __future__ import annotations

import copy
import importlib
import os
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from noisevault.errors import MigrationWarning
from noisevault.profile import Profile, load_file

V01 = Path(__file__).parent / "fixtures" / "v01"
MANILA_V01 = V01 / "fake_manila_2024-05-27T15-27-23-03-00.json"

_TOY: dict[str, Any] = {
    "noisevault": "1.0",
    "device": {"name": "toy", "vendor": "test", "technology": "superconducting", "num_qubits": 3},
    "connectivity": {"edges": [[0, 1], [1, 2]]},
    "gates": {
        "rz": {"virtual": True},
        "sx": {"avg_infidelity": 1e-3, "duration_ns": 35},
        "cz": {"avg_infidelity": 1e-2, "duration_ns": 70},
    },
}


def require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if os.environ.get("NOISEVAULT_REQUIRE_ALL") == "1":
            pytest.fail(f"{module} must be installed when NOISEVAULT_REQUIRE_ALL=1 ({exc})")
        pytest.skip(f"{module} is not installed", allow_module_level=True)


def toy(**sections: Any) -> dict[str, Any]:
    """A small valid profile dict; keyword arguments replace top-level sections."""
    data = copy.deepcopy(_TOY)
    data.update(copy.deepcopy(sections))
    return data


def migrated(path: Path) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MigrationWarning)
        return load_file(path)


@pytest.fixture(autouse=True)
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets an empty private vault, never the user's ~/.noisevault."""
    home = tmp_path / "nv_home"
    monkeypatch.setenv("NOISEVAULT_HOME", str(home))
    return home / "profiles"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: a timing or scale test that still runs by default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked ``network`` reach live endpoints, so they run only with NOISEVAULT_NETWORK=1.

    Timing budgets (``slow``) are only meaningful in a serial run; parallel workers share the CPU.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        serial_only = pytest.mark.skip(reason="timing budget is checked in a serial run")
        for item in items:
            if "slow" in item.keywords and "seconds" in item.name:
                item.add_marker(serial_only)
    if os.environ.get("NOISEVAULT_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="set NOISEVAULT_NETWORK=1 to run network tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
