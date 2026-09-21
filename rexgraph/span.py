"""Exact span components and their local annotation attachments."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric, _identity
from rexgraph.graded_metric import _fraction
from rexgraph.type_accession import CoordinateSpace, TypeAccession

__all__ = ["SpanBlock", "SpanRefinement", "SpanAttachment"]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _qkey(value):
    return hex(value.numerator) + "/" + hex(value.denominator)


@dataclass(frozen=True)
class SpanBlock:
    """Named half open intervals in one exact coordinate system."""

    name: str
    axis: str
    unit: str
    components: tuple
    interpretation: str = "joint"

    def __post_init__(self):
        for key in ("name", "axis", "unit"):
            _text(getattr(self, key), key)
        if self.interpretation not in {"joint", "alternative"}:
            raise ValueError("span interpretation must be joint or alternative")
        components = []
        for key, start, stop in self.components:
            key = _text(key, "component key")
            start, stop = _fraction(start), _fraction(stop)
            if stop <= start:
                raise ValueError("interval components require start less than stop")
            components.append((key, start, stop))
        if len({c[0] for c in components}) != len(components):
            raise ValueError("span component keys must be unique")
        object.__setattr__(self, "components", tuple(components))

    @property
    def coordinates(self):
        name = _identity(("span-components-v1", self.axis, self.unit, self.name))
        return CoordinateSpace("span/" + name, tuple(c[0] for c in self.components))

    @property
    def coefficient_digest(self):
        return _identity(("span-block-v1", self.name, self.axis, self.unit, self.interpretation,
                          tuple((key, _qkey(a), _qkey(b)) for key, a, b in self.components)))


@dataclass(frozen=True)
class SpanRefinement:
    """Elementary intervals induced by explicitly retained rational endpoints."""

    axis: str
    unit: str
    points: tuple

    def __post_init__(self):
        _text(self.axis, "axis")
        _text(self.unit, "unit")
        points = tuple(_fraction(p) for p in self.points)
        if any(a >= b for a, b in zip(points[:-1], points[1:], strict=True)):
            raise ValueError("refinement endpoints must be strictly increasing")
        object.__setattr__(self, "points", points)

    @classmethod
    def from_blocks(cls, blocks):
        blocks = tuple(blocks)
        if not blocks or any(not isinstance(b, SpanBlock) for b in blocks):
            raise TypeError("refinement requires one or more span blocks")
        axis, unit = blocks[0].axis, blocks[0].unit
        if any(b.axis != axis or b.unit != unit for b in blocks):
            raise ValueError("refinement requires the same coordinate axis and unit")
        points = sorted({p for b in blocks for _, a, z in b.components for p in (a, z)})
        return cls(axis, unit, tuple(points))

    @property
    def intervals(self):
        return tuple(zip(self.points[:-1], self.points[1:], strict=True))

    @property
    def coefficient_digest(self):
        return _identity(("span-refinement-v1", self.axis, self.unit, tuple(_qkey(p) for p in self.points)))

    @property
    def space(self):
        return CoordinateSpace("interval/" + self.coefficient_digest,
                               tuple(_qkey(a) + ":" + _qkey(b) for a, b in self.intervals))

    @property
    def endpoint_space(self):
        return CoordinateSpace("endpoint/" + self.coefficient_digest,
                               tuple(_qkey(p) for p in self.points))

    @property
    def metric(self):
        return CoordinateMetric.diagonal(self.space, tuple(b-a for a, b in self.intervals))

    @property
    def boundary(self):
        entries = tuple(e for i in range(len(self.intervals)) for e in ((i, i, -1), (i+1, i, 1)))
        return CoordinateMap(self.space, self.endpoint_space, entries)

    def realization(self, block):
        if not isinstance(block, SpanBlock) or (block.axis, block.unit) != (self.axis, self.unit):
            raise ValueError("span and refinement must use the same axis and unit")
        index = {p: i for i, p in enumerate(self.points)}
        entries = []
        for j, (_, start, stop) in enumerate(block.components):
            if start not in index or stop not in index:
                raise ValueError("refinement must retain every component endpoint")
            entries.extend((i, j, 1) for i in range(index[start], index[stop]))
        return CoordinateMap(block.coordinates, self.space, tuple(entries))

    def coverage(self, block, *, mode):
        """Observe component multiplicity or the union without changing the block."""
        if block.interpretation != "joint":
            raise ValueError("alternative intervals require an explicit component choice")
        if mode not in {"sum", "union"}:
            raise ValueError("coverage mode must be sum or union")
        result = self.realization(block).apply([1] * len(block.components))
        if mode == "union":
            result = np.asarray([Fraction(int(v != 0)) for v in result], dtype=object)
        return result

    def refinement_map(self, target):
        if not isinstance(target, SpanRefinement) or (target.axis, target.unit) != (self.axis, self.unit):
            raise ValueError("refinement maps require the same coordinate system")
        if not set(self.points).issubset(target.points):
            raise ValueError("target refinement must retain every source endpoint")
        index = {p: i for i, p in enumerate(target.points)}
        entries = tuple((i, j, 1) for j, (a, b) in enumerate(self.intervals)
                        for i in range(index[a], index[b]))
        return CoordinateMap(self.space, target.space, entries)

    def endpoint_map(self, target):
        self.refinement_map(target)
        index = {p: i for i, p in enumerate(target.points)}
        return CoordinateMap(self.endpoint_space, target.endpoint_space,
                             tuple((index[p], j, 1) for j, p in enumerate(self.points)))

    def accession(self, source, grade, name, columns, *, mode, local=False):
        """Realize declared primary columns on shared or attachment local support."""
        if not isinstance(local, bool):
            raise TypeError("local must be a boolean")
        columns = tuple(columns)
        if any(isinstance(j, (bool, np.bool_)) or not isinstance(j, Integral) for j, _ in columns):
            raise TypeError("span columns require explicit primary cell indices")
        if len({int(j) for j, _ in columns}) != len(columns):
            raise ValueError("each primary column must have one declared span block")
        entries, keys = [], []
        n = len(self.intervals)
        for a, (column, block) in enumerate(columns):
            values = self.coverage(block, mode=mode)
            offset = a*n if local else 0
            if local:
                keys.extend(f"cell:{int(column)}:{key}" for key in self.space.keys)
            entries.extend((offset+i, int(column), v) for i, v in enumerate(values) if v)
        coords = (CoordinateSpace("local/"+self.coefficient_digest, tuple(keys)) if local else self.space)
        return TypeAccession(source, grade, name, tuple(entries), coordinates=coords)


@dataclass(frozen=True)
class SpanAttachment:
    """An identified event or entity attachment with text and time realizations."""

    annotation_id: str
    owner_id: str
    role: str
    source_id: str
    text: SpanBlock | None = None
    time: SpanBlock | None = None
    grounding: CoordinateMap | None = None
    qualifiers: tuple = ()

    def __post_init__(self):
        for key in ("annotation_id", "owner_id", "role", "source_id"):
            _text(getattr(self, key), key)
        if self.text is None and self.time is None:
            raise ValueError("an attachment requires a text or time realization")
        if any(b is not None and not isinstance(b, SpanBlock) for b in (self.text, self.time)):
            raise TypeError("attachment supports must be SpanBlocks")
        if self.grounding is not None:
            if (self.text is None or self.time is None or not isinstance(self.grounding, CoordinateMap)
                    or self.grounding.domain != self.text.coordinates
                    or self.grounding.codomain != self.time.coordinates):
                raise ValueError("grounding must map text components to time components")
        qualifiers = tuple(tuple(q) for q in self.qualifiers)
        if any(len(q) != 2 or not all(isinstance(v, str) for v in q) or not q[0] for q in qualifiers):
            raise ValueError("qualifiers must contain named string values")
        if len({q[0] for q in qualifiers}) != len(qualifiers):
            raise ValueError("qualifier keys must be unique")
        object.__setattr__(self, "qualifiers", qualifiers)

    @property
    def coefficient_digest(self):
        return _identity(("span-attachment-v1", self.annotation_id, self.owner_id, self.role, self.source_id,
                          None if self.text is None else self.text.coefficient_digest,
                          None if self.time is None else self.time.coefficient_digest,
                          None if self.grounding is None else self.grounding.coefficient_digest,
                          self.qualifiers))
