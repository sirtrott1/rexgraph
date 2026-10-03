"""One registry for trusted live sources; serialized names never load providers."""
from __future__ import annotations

from dataclasses import dataclass
from inspect import getattr_static
from threading import RLock

from rexgraph.registry import Registry
from .types import ValueKind


@dataclass(frozen=True)
class SourceSignature:
    name: str
    kind: ValueKind
    predicate: object
    priority: int = 0

    def __post_init__(self):
        if type(self.name) is not str or not self.name or not isinstance(self.kind, ValueKind):
            raise TypeError("source signatures require a name and ValueKind")
        if not callable(self.predicate) or type(self.priority) is not int:
            raise TypeError("source signatures require a predicate and integer priority")


class SourceRegistry(Registry):
    def __init__(self):
        super().__init__("RCQL source signature")
        self._lock = RLock()

    def register(self, signature, *, replace=False):
        if not isinstance(signature, SourceSignature) or type(replace) is not bool:
            raise TypeError("register a SourceSignature with explicit boolean replacement")
        with self._lock:
            if signature.name in self and not replace:
                raise ValueError(f"source signature {signature.name!r} is already registered")
            return super().register(signature.name, signature)

    def unregister(self, name):
        with self._lock:
            return super().unregister(name)

    def clear(self):
        with self._lock:
            super().clear()

    def classify(self, value):
        with self._lock:
            signatures = tuple(self._values.values())
        matches = []
        for signature in signatures:
            match = signature.predicate(value)
            if type(match) is not bool:
                raise TypeError(f"source signature {signature.name!r} must return a boolean")
            if match:
                matches.append(signature)
        if not matches:
            return ValueKind.UNKNOWN
        priority = max(s.priority for s in matches)
        winners = [s for s in matches if s.priority == priority]
        if len(winners) != 1:
            raise ValueError("ambiguous RCQL source signatures: " + ", ".join(sorted(s.name for s in winners)))
        return winners[0].kind


def _surface(value, *names):
    return all(getattr_static(value, name, None) is not None for name in names)


def _nominal(value, module, name):
    return any((cls.__module__, cls.__name__) == (module, name) for cls in type(value).__mro__)


SOURCES = SourceRegistry()
for _signature in (
    SourceSignature("turn-field", ValueKind.TURN_FIELD, lambda v: _nominal(v, "rexgraph.flow.turn_field", "TurnField"), 50),
    SourceSignature("rcdb", ValueKind.RCDB_STORE, lambda v: _surface(v, "get", "history"), 40),
    SourceSignature("catalog", ValueKind.CATALOG_ENTRY_SET, lambda v: _surface(v, "hash_all", "list"), 30),
    SourceSignature("temporal", ValueKind.TEMPORAL_REX, lambda v: _surface(v, "reconstruct_at", "T"), 20),
    SourceSignature("rex", ValueKind.REX, lambda v: _surface(v, "betti", "nV"), 10),
    SourceSignature("dataset", ValueKind.DATASET, lambda v: _nominal(v, "rcql.dataset_source", "DatasetSource"), 60),
):
    SOURCES.register(_signature)
del _signature

__all__ = ["SourceSignature", "SourceRegistry", "SOURCES"]
