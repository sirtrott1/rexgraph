"""Declared record to relation construction, shared by all source readers."""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral

from .relations import Relations, RelationSpec, VertexTable
from .value import Absent


def _records(records):
    from .io.records import RecordBatch, TypedRecord
    if isinstance(records, RecordBatch):
        yield from records.records()
        return
    for item in records:
        if isinstance(item, RecordBatch):
            yield from item.records()
        elif isinstance(item, TypedRecord):
            yield item
        else:
            raise TypeError("construction requires typed records or batches")


def _identity(value, context):
    if value is Absent or value is None or isinstance(value, bool):
        raise ValueError(f"{context} requires a declared, present identity")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{context} requires a hashable identity") from exc
    return value


def _index(value, context):
    if isinstance(value, Fraction) and value.denominator == 1:
        value = value.numerator
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{context} requires an exact integer")
    return int(value)


def construct(records, spec: RelationSpec, *, vertices: VertexTable | None = None,
              max_relations=10_000_000, max_participants=100_000_000):
    """Construct sparse declared relations without format specific interpretation.

    ``head`` names an exact head slot field. For long form its value is common
    to the whole relation, while ``share`` is scalar per incidence. Row shares
    are aligned sequences. Long form preserves first occurrence order and
    refuses inconsistent relation level declarations across incidence records.
    """
    if not isinstance(spec, RelationSpec):
        raise TypeError("construction requires a RelationSpec")
    if any(type(n) is not int or not 0 < n < 2**31 for n in (max_relations, max_participants)):
        raise ValueError("construction limits must fit the native sparse domain")
    vertex_index = {}
    if vertices is not None:
        if not isinstance(vertices, VertexTable):
            raise TypeError("construction vertices require a VertexTable")
        vertex_index = {v: i for i, v in enumerate(vertices.ids) if v is not Absent}
    supports, declarations, pointers, groups, shares = [], [], [], {}, []
    participant_count = 0

    def participant(value):
        value = _identity(value, "participant")
        if value not in vertex_index:
            if vertices is not None:
                raise ValueError("participant is outside the declared vertex domain")
            vertex_index[value] = len(vertex_index)
        return vertex_index[value]

    fields = set(spec.participants) | {v for v in (spec.head, spec.share, spec.weight, spec.sign, spec.identity, spec.type) if v is not None} | set(spec.attributes)
    for record in _records(records):
        if not fields <= record.fields.keys():
            raise ValueError("relation declaration names fields outside the record schema")
        values = record.fields
        declared = {"head": 0 if spec.head is None else _index(values[spec.head], "head slot"),
                    "weight": Absent if spec.weight is None else values[spec.weight],
                    "sign": 1 if spec.sign is None else _index(values[spec.sign], "relation sign"),
                    "identity": Absent if spec.identity is None else values[spec.identity],
                    "type": Absent if spec.type is None else values[spec.type],
                    "attributes": {name: values[name] for name in spec.attributes}}
        if spec.shape == "pair":
            row = [values[name] for name in spec.participants]
        elif spec.shape == "row":
            row = values[spec.participants[0]]
            if not isinstance(row, (tuple, list)) or not row:
                raise TypeError("row relation requires a nonempty declared participant sequence")
        else:
            row = [values[spec.participants[0]]]
        row = [participant(v) for v in row]
        share = [Absent]*len(row) if spec.share is None else values[spec.share]
        if spec.shape == "long":
            share = [share] if spec.share is not None else share
        elif spec.share is not None and (not isinstance(share, (tuple, list)) or len(share) != len(row)):
            raise ValueError("row shares must align with the participant sequence")
        if spec.shape == "long":
            identity = _identity(declared["identity"], "long-form relation")
            if identity in groups:
                index = groups[identity]
                if declarations[index] != declared:
                    raise ValueError("conflicting declarations within a long-form relation")
                supports[index].extend(row)
                pointers[index].append(record.pointer.as_record())
                shares[index].extend(share)
            else:
                groups[identity] = len(supports)
                supports.append(row)
                declarations.append(declared)
                pointers.append([record.pointer.as_record()])
                shares.append(list(share))
        else:
            supports.append(row)
            declarations.append(declared)
            pointers.append([record.pointer.as_record()])
            shares.append(list(share))
        participant_count += len(row)
        if len(supports) > max_relations or participant_count > max_participants:
            raise ValueError("declared construction exceeds its sparse domain limits")
    if vertices is None:
        keys = tuple(vertex_index)
        vertices = VertexTable(keys, tuple(v if type(v) is str else "" for v in keys))
    return Relations.from_supports(
        supports, vertices=vertices,
        heads=[d["head"] for d in declarations], weights=[d["weight"] for d in declarations],
        signs=[d["sign"] for d in declarations],
        shares=[v for row in (shares if supports else ()) for v in row],
        relation_ids=[d["identity"] for d in declarations], relation_types=[d["type"] for d in declarations],
        attributes={1: {i: d["attributes"] for i, d in enumerate(declarations) if d["attributes"]}},
        provenance={"source_pointers": pointers},
    )


__all__ = ["construct"]
