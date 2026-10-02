"""Calibration sources, each imported only when used."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..errors import SourceDataError, SourceUnavailable, parse_json
from ..profile import Profile

OFFLINE_HINT = (
    "check the network connection, or run nv list to see every profile you can load offline"
)
OLDER_HINT = "pass an earlier at= to use an older calibration"


@dataclass(frozen=True)
class Origin:
    """How the errors about a source's data name the source, and the next step they give."""

    name: str
    hint: str | None = None

    def refuse(self, problem: str) -> SourceDataError:
        return SourceDataError(f"{self.name}: {problem}", hint=self.hint)

    def profile(self, data: Mapping[str, Any]) -> Profile:
        """The profile of ``data``; a value the format refuses is a SourceDataError naming it."""
        try:
            return Profile.model_validate(data)
        except ValidationError as exc:
            raise self.refuse(_first_problem(exc, data)) from None


def _first_problem(exc: ValidationError, data: Mapping[str, Any]) -> str:
    """The first field's problem, from its deepest error, which for a union is the member the
    value was meant for.
    """
    errors = exc.errors()
    field = errors[0]["loc"][:1]
    error = max((e for e in errors if e["loc"][:1] == field), key=lambda e: len(e["loc"]))
    message = error["msg"].removeprefix("Value error, ").split("\n")[0]
    if error["type"] != "value_error" and isinstance(error["input"], int | float | str):
        message += f", got {error['input']!r}"
    where = _where(error["loc"], data)
    return f"{where}: {message}" if where else message


_RECORD_NAMES: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "qubits": lambda record: f"qubit {record['index']}",
    "calibrations": lambda record: f"{record['gate']} on qubits {list(record['qubits'])}",
}


def _where(loc: Sequence[str | int], data: Mapping[str, Any]) -> str:
    """Where ``loc`` points in ``data``, such as ``readout.error of qubit 3``. A key ``data``
    lacks is pydantic's name for a union member and is left out.
    """
    keys: list[str | int] = []
    node: Any = data
    for key in loc:
        if (isinstance(node, Mapping) and key in node) or (
            isinstance(node, list | tuple) and isinstance(key, int)
        ):
            keys.append(key)
            node = node[key]
    if len(keys) > 1 and keys[0] in _RECORD_NAMES:
        record = _RECORD_NAMES[keys[0]](data[keys[0]][keys[1]])
        rest = json_path(keys[2:])
        return f"{rest} of {record}" if rest else record
    return json_path(keys)


def read_reply(raw: bytes, url: str, shape: TypeAdapter[Any], *, sender: str, hint: str) -> Any:
    """``raw`` parsed as JSON of ``shape``; SourceUnavailable naming ``url`` when it is not."""
    try:
        data = parse_json(raw)
    except ValueError:
        raise SourceUnavailable(
            f"{sender} answered {url} with something other than JSON", hint=hint
        ) from None
    try:
        shape.validate_python(data, strict=True)
    except ValidationError as exc:
        loc = exc.errors()[0]["loc"]
        where = f" at {json_path(loc)}" if loc else ""
        raise SourceUnavailable(
            f"{sender} answered {url} with JSON of the wrong shape{where}", hint=hint
        ) from None
    return data


def json_path(keys: Sequence[str | int]) -> str:
    """``standardized.oneQubitProperties['1'].oneQubitFidelity[0]``."""
    return "".join(
        f".{key}" if isinstance(key, str) and key.isidentifier() else f"[{key!r}]" for key in keys
    ).removeprefix(".")
