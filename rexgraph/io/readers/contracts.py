"""Closed option declarations and bounded observations for trusted readers."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

from rexgraph.value import Absent
from rexgraph.value_codec import pack_value, unpack_value

_RESERVED = {"schema", "reader", "registry", "media_type", "prefix", "batch_size", "max_record_bytes"}


@dataclass(frozen=True)
class ReaderParameter:
    name: str
    kind: str = "value"
    default: object = Absent
    nullable: bool = False
    minimum: int | None = None
    maximum: int | None = None
    _default_bytes: bytes | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        if type(self.name) is not str or not self.name or self.name in _RESERVED:
            raise ValueError("reader parameter requires a nonreserved name")
        if self.kind not in {"value", "integer", "boolean", "text", "character", "text_sequence"}:
            raise ValueError("unknown reader parameter kind")
        if type(self.nullable) is not bool:
            raise TypeError("parameter nullability must be boolean")
        for limit in (self.minimum, self.maximum):
            if limit is not None and (type(limit) is not int or self.kind != "integer"):
                raise TypeError("integer parameters require integer bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter bounds are reversed")
        if self.default is not Absent:
            object.__setattr__(self, "default", self.check(self.default))
            object.__setattr__(self, "_default_bytes", pack_value(self.default))

    def default_value(self):
        return Absent if self._default_bytes is None else unpack_value(self._default_bytes)

    def check(self, value):
        if value is None:
            if self.nullable:
                return None
            raise TypeError(f"reader parameter {self.name!r} does not allow null")
        if self.kind == "integer":
            if type(value) is not int:
                raise TypeError(f"reader parameter {self.name!r} requires an integer")
            if self.minimum is not None and value < self.minimum or self.maximum is not None and value > self.maximum:
                raise ValueError(f"reader parameter {self.name!r} exceeds its bounds")
        elif self.kind == "boolean" and type(value) is not bool:
            raise TypeError(f"reader parameter {self.name!r} requires a boolean")
        elif self.kind in {"text", "character"}:
            if type(value) is not str or self.kind == "character" and len(value) != 1:
                raise TypeError(f"reader parameter {self.name!r} requires {self.kind}")
        elif self.kind == "text_sequence":
            if not isinstance(value, (tuple, list)) or any(type(v) is not str for v in value):
                raise TypeError(f"reader parameter {self.name!r} requires a text sequence")
            value = tuple(value)
        return unpack_value(pack_value(value))


@dataclass(frozen=True)
class ParameterSchema:
    fields: tuple[ReaderParameter, ...] = ()

    def __post_init__(self):
        fields = tuple(self.fields)
        if any(not isinstance(f, ReaderParameter) for f in fields) or len({f.name for f in fields}) != len(fields):
            raise ValueError("reader parameters require distinct declared fields")
        object.__setattr__(self, "fields", fields)

    def bind(self, options):
        if not isinstance(options, Mapping) or any(type(k) is not str for k in options):
            raise TypeError("reader options require named values")
        unknown = set(options) - {f.name for f in self.fields}
        if unknown:
            raise ValueError("undeclared reader options: " + ", ".join(sorted(unknown)))
        values = {}
        for f in self.fields:
            value = options[f.name] if f.name in options else f.default_value()
            if value is Absent:
                raise ValueError(f"required reader parameter {f.name!r} is missing")
            values[f.name] = f.check(value)
        return values


@dataclass(frozen=True)
class ReaderProfile:
    """Observed field names from an explicit, bounded profile operation.

    A profile does not choose field types, number rules or relation construction.
    The caller declares those after inspecting the observations.
    """
    reader: str
    fields: tuple[str, ...]
    sampled_records: int
    record_limit: int
    options: object = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from types import MappingProxyType
        fields = tuple(self.fields)
        if type(self.reader) is not str or not self.reader or any(type(n) is not str or not n for n in fields):
            raise ValueError("reader profile requires named fields")
        if len(set(fields)) != len(fields):
            raise ValueError("reader profile fields must be distinct")
        if (type(self.sampled_records) is not int or type(self.record_limit) is not int
                or not 0 <= self.sampled_records <= self.record_limit <= 1_000_000 or not self.record_limit):
            raise ValueError("reader profile counts exceed their bounds")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "options", MappingProxyType(unpack_value(pack_value(dict(self.options)))))


__all__ = ["ReaderParameter", "ParameterSchema", "ReaderProfile"]
