"""Typed, present source records. Readers terminate here, before construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

import numpy as np

from rexgraph import Absent, Approx, ExactArray, NumberRule, ValueRules
from rexgraph.value import convert_number
from rexgraph.value_codec import pack_value, unpack_value

_KINDS = {"number", "string", "bool", "bytes", "value", "sequence"}


def _native_source_value(value, rules):
    if isinstance(value, Decimal):
        return convert_number(value, rules.number)
    if isinstance(value, dict):
        if rules.number == NumberRule.JSON_RATIONAL and set(value) == {"numerator", "denominator"}:
            return convert_number(value, rules.number)
        return {k: _native_source_value(v, rules) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        values = [_native_source_value(v, rules) for v in value]
        return tuple(values) if isinstance(value, tuple) else values
    return value


@dataclass(frozen=True)
class SourcePointer:
    uri: str
    digest: bytes | None = None
    record: int | str | None = None
    field: str | None = None
    byte_range: tuple[int, int] | None = None

    def __post_init__(self):
        if type(self.uri) is not str or not self.uri:
            raise ValueError("source URI must be declared")
        if self.digest is not None and (type(self.digest) is not bytes or len(self.digest) != 32):
            raise ValueError("source digest requires SHA256 bytes")
        if self.record is not None and (type(self.record) not in (str, int) or type(self.record) is int and self.record < 0):
            raise ValueError("source record must be a nonnegative index or declared string")
        if self.field is not None and type(self.field) is not str:
            raise ValueError("source field must be text")
        if self.byte_range is not None:
            start, end = self.byte_range
            if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                raise ValueError("invalid source byte range")

    def as_record(self):
        return {"uri": self.uri, "digest": self.digest, "record": self.record,
                "field": self.field, "byte_range": self.byte_range}


@dataclass(frozen=True)
class RecordField:
    name: str
    kind: str = "string"
    rules: ValueRules = field(default_factory=ValueRules)
    nullable: bool = True

    def __post_init__(self):
        if type(self.name) is not str or not self.name or self.kind not in _KINDS:
            raise ValueError("record fields require a name and a supported kind")
        if not isinstance(self.rules, ValueRules) or type(self.nullable) is not bool:
            raise TypeError("record field rules and nullability must be declared")

    def convert(self, value, *, context):
        if self.kind in {"sequence", "value"}:
            absent = value is None or value is Absent or isinstance(value, str) and value in self.rules.absent_tokens
            if absent:
                value = Absent
            elif self.kind == "sequence" and not isinstance(value, (list, tuple)):
                raise TypeError(f"{context} requires a declared participant sequence")
            else:
                value = unpack_value(pack_value(_native_source_value(value, self.rules)))
        else:
            value = self.rules.convert(value, kind=self.kind, context=context)
        if value is Absent and not self.nullable:
            raise ValueError(f"{context} is required")
        return value


@dataclass(frozen=True)
class RecordSchema:
    fields: tuple[RecordField, ...]
    unknown_fields: str = "reject"

    def __post_init__(self):
        fields = tuple(self.fields)
        if any(not isinstance(f, RecordField) for f in fields) or len({f.name for f in fields}) != len(fields):
            raise ValueError("record schema requires distinct declared fields")
        if self.unknown_fields not in {"reject", "ignore"}:
            raise ValueError("unknown source fields require an explicit policy")
        object.__setattr__(self, "fields", fields)

    @property
    def names(self):
        return tuple(f.name for f in self.fields)

    def convert(self, record, pointer):
        if not isinstance(record, dict) or any(type(k) is not str for k in record):
            raise TypeError("source record must have named fields")
        if self.unknown_fields == "reject" and set(record) - set(self.names):
            raise ValueError(f"undeclared source fields at {pointer.uri}:{pointer.record}")
        values = {f.name: f.convert(record.get(f.name, Absent), context=f"{pointer.uri}:{pointer.record}:{f.name}")
                  for f in self.fields}
        return TypedRecord(values, pointer)


@dataclass(frozen=True)
class ValueArray:
    values: tuple
    kind: str
    presence: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        values = tuple(self.values)
        if self.kind not in _KINDS:
            raise ValueError("unknown value-array kind")
        object.__setattr__(self, "values", values)
        mask = np.frombuffer(bytes(v is not Absent for v in values), dtype=bool)
        object.__setattr__(self, "presence", mask)

    def exact(self):
        if self.kind != "number" or any(isinstance(v, Approx) for v in self.values):
            raise TypeError("this column is not declared exact numeric data")
        return ExactArray.from_values(self.values)

    def __len__(self):
        return len(self.values)


@dataclass(frozen=True)
class TypedRecord:
    fields: object
    pointer: SourcePointer

    def __post_init__(self):
        if not isinstance(self.pointer, SourcePointer):
            raise TypeError("typed record requires a source pointer")
        fields = dict(self.fields)
        if any(type(k) is not str or not k for k in fields):
            raise ValueError("typed records require named fields")
        object.__setattr__(self, "fields", MappingProxyType(fields))


@dataclass(frozen=True)
class RecordBatch:
    schema: RecordSchema
    columns: object
    pointers: tuple[SourcePointer, ...]

    def __post_init__(self):
        pointers = tuple(self.pointers)
        columns = dict(self.columns)
        if not isinstance(self.schema, RecordSchema) or set(columns) != set(self.schema.names):
            raise ValueError("record batch columns must match its schema")
        if any(not isinstance(p, SourcePointer) for p in pointers):
            raise TypeError("record batch requires source pointers")
        for f in self.schema.fields:
            column = columns[f.name]
            if not isinstance(column, ValueArray) or column.kind != f.kind or len(column) != len(pointers):
                raise ValueError("record batch columns must be aligned typed value arrays")
        object.__setattr__(self, "pointers", pointers)
        object.__setattr__(self, "columns", MappingProxyType(columns))

    @classmethod
    def from_records(cls, schema, records):
        records = tuple(records)
        return cls(schema, {f.name: ValueArray(tuple(r.fields[f.name] for r in records), f.kind) for f in schema.fields},
                   tuple(r.pointer for r in records))

    def records(self):
        for i, pointer in enumerate(self.pointers):
            yield TypedRecord({name: column.values[i] for name, column in self.columns.items()}, pointer)

    def __len__(self):
        return len(self.pointers)
