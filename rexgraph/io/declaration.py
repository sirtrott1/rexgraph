"""Portable, non executable dataset declarations over readers and construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
from types import MappingProxyType

from rexgraph.relations import RelationSpec, VertexTable
from rexgraph.value import ValueRules
from rexgraph.value_codec import pack_value, unpack_value
from .records import RecordField, RecordSchema


@dataclass(frozen=True)
class DatasetDeclaration:
    reader: str
    schema: RecordSchema
    relations: RelationSpec
    options: object = field(default_factory=dict)
    vertices: VertexTable | None = None
    _seal: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.reader) is not str or not self.reader or not isinstance(self.schema, RecordSchema) or not isinstance(self.relations, RelationSpec):
            raise TypeError("dataset requires a reader, record schema and relation declaration")
        if self.vertices is not None and not isinstance(self.vertices, VertexTable):
            raise TypeError("dataset vertex domain requires a VertexTable")
        options = unpack_value(pack_value(dict(self.options)))
        if any(type(k) is not str for k in options) or set(options) & {"schema", "reader", "registry"}:
            raise ValueError("dataset options cannot replace its declared contracts")
        object.__setattr__(self, "options", MappingProxyType(options))
        if self.vertices is not None:
            object.__setattr__(self, "vertices", VertexTable(**dict(self.vertices.as_record(), attributes=self.vertices.attributes)))
        object.__setattr__(self, "_seal", hashlib.sha256(pack_value(self.as_record())).digest())

    def read(self, source, *, registry=None):
        from .readers import read_batches
        self.check_state()
        return read_batches(source, schema=self.schema, reader=self.reader, registry=registry,
                            **unpack_value(pack_value(dict(self.options))))

    def construct(self, source, *, registry=None):
        from rexgraph.construction import construct
        return construct(self.read(source, registry=registry), self.relations, vertices=self.vertices)

    def to_rex(self, source, *, registry=None, **graph_options):
        from rexgraph.graph import RexGraph
        return RexGraph.from_relations(self.construct(source, registry=registry), **graph_options)

    def as_record(self):
        fields = [{"name": f.name, "kind": f.kind, "nullable": f.nullable,
                   "rules": {"number": f.rules.number.value, "absent_tokens": f.rules.absent_tokens}}
                  for f in self.schema.fields]
        vertices = None if self.vertices is None else dict(self.vertices.as_record(), attributes=self.vertices.attributes)
        return {"version": 1, "reader": self.reader,
                "schema": {"fields": fields, "unknown_fields": self.schema.unknown_fields},
                "relations": asdict(self.relations), "options": dict(self.options), "vertices": vertices}

    def to_bytes(self):
        data = pack_value(self.as_record())
        if not hmac.compare_digest(hashlib.sha256(data).digest(), self._seal):
            raise ValueError("dataset declaration changed; declare the new source mapping explicitly")
        return data

    def check_state(self):
        self.to_bytes()

    @property
    def digest(self):
        self.check_state()
        return self._seal.hex()

    @classmethod
    def from_bytes(cls, data):
        record = unpack_value(data)
        if (not isinstance(record, dict) or set(record) != {"version", "reader", "schema", "relations", "options", "vertices"}
                or type(record["version"]) is not int or record["version"] != 1):
            raise ValueError("unknown dataset declaration")
        schema = record["schema"]
        if set(schema) != {"fields", "unknown_fields"}:
            raise ValueError("unknown dataset record schema")
        if any(not isinstance(f, dict) or set(f) != {"name", "kind", "nullable", "rules"} for f in schema["fields"]):
            raise ValueError("unknown dataset field declarations")
        fields = tuple(RecordField(f["name"], f["kind"], ValueRules(**f["rules"]), f["nullable"]) for f in schema["fields"])
        return cls(record["reader"], RecordSchema(fields, schema["unknown_fields"]),
                   RelationSpec(**record["relations"]), record["options"],
                   None if record["vertices"] is None else VertexTable(**record["vertices"]))


__all__ = ["DatasetDeclaration"]
