"""Live values visible to the System observatory."""
from __future__ import annotations

from threading import RLock
from typing import Any


class SourceStore:
    """Named live values queried by System through RCQL."""

    def __init__(self):
        self._lock = RLock()
        self._sources: dict[str, Any] = {}

    def register(self, name: str, value: Any, policy=None) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("source name cannot be empty")
        if policy is not None:
            from rcql.capabilities import BoundSource
            value = BoundSource(value, policy)
        with self._lock:
            self._sources[name] = value

    def remove(self, name: str) -> None:
        with self._lock:
            self._sources.pop(str(name), None)

    def get(self, name: str) -> Any:
        with self._lock:
            try:
                return self._sources[str(name)]
            except KeyError as exc:
                raise KeyError(f"unknown System source {name!r}") from exc

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._sources)


sources = SourceStore()
