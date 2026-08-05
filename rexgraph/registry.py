"""
rexgraph.registry: one extension-point primitive.

The tree grew five near-registries with five different surfaces. io had
register/unregister/available, compute had register plus available, rcdb and
_serialization had register alone, connectors was a bare dict, and the temporal
rerank policies were a hardcoded tuple. Same pattern, five shapes, so extending any
one of them meant first working out which one you were in, and three of them could
take a registration but never give it back.

    FORMATS = Registry("format")
    FORMATS.register("rex", handler, extensions=[".rex"])
    FORMATS.require("rex")          # raises naming what IS registered
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class Registry:
    """A named extension point: register / unregister / available / require.

    `what` is the singular noun for the thing being registered. It only appears in
    error messages, which is precisely where it earns its place: a bare KeyError
    sends the caller to read the source to find out what was expected.
    """

    __slots__ = ("what", "_values", "_meta")

    def __init__(self, what: str):
        self.what = what
        self._values: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def register(self, name: str, value: Any, **meta: Any) -> Any:
        """Register `value` under `name`, replacing any previous entry.

        `meta` is whatever the call site needs to keep alongside it: a format's
        extensions, a backend's kind. Keeping it here rather than in a parallel dict
        is what stops the two drifting apart.
        """
        self._values[name] = value
        self._meta[name] = dict(meta)
        return value

    def unregister(self, name: str) -> Any:
        """Remove `name`. Returns what was registered, or None if it was not."""
        self._meta.pop(name, None)
        return self._values.pop(name, None)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def require(self, name: str) -> Any:
        """Like `get`, but a miss names what would have worked."""
        if name not in self._values:
            raise KeyError(
                f"unknown {self.what} {name!r}. Registered: "
                f"{', '.join(self.available()) or '(none)'}")
        return self._values[name]

    def meta(self, name: str) -> dict[str, Any]:
        return dict(self._meta.get(name, {}))

    def available(self) -> list[str]:
        return sorted(self._values)

    def items(self) -> Iterator[tuple[str, Any]]:
        for name in self.available():
            yield name, self._values[name]

    def clear(self) -> None:
        self._values.clear()
        self._meta.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __repr__(self) -> str:
        return f"<Registry {self.what}: {', '.join(self.available()) or 'empty'}>"
