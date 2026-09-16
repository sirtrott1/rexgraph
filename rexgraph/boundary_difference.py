"""Exact primary boundary differences on explicit union coordinates."""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property

from rexgraph.native_rank import primary_columns, clear_column_denominators
from rexgraph.graded_boundary import _rank_integer_columns
from rexgraph.type_accession import _action


def vertex_alignment(reference, other, ref_labels=None, other_labels=None):
    """An injective endpoint map into the complete union, without label coercion."""
    if ref_labels is None and other_labels is None:
        keys = tuple(range(max(reference.nV, other.nV)))
        return keys, tuple(range(reference.nV)), tuple(range(other.nV))
    if ref_labels is None or other_labels is None:
        raise ValueError("give vertex labels for both endpoints or neither")
    labels = []
    for values, count in ((ref_labels, reference.nV), (other_labels, other.nV)):
        if isinstance(values, (str, bytes)):
            raise TypeError("vertex labels require an ordered sequence")
        values = tuple(values)
        if len(values) != count or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("vertex labels require one nonempty string per C0 cell")
        if len(set(values)) != len(values):
            raise ValueError("vertex labels must be unique at each endpoint")
        labels.append(values)
    keys = tuple(dict.fromkeys((*labels[0], *labels[1])))
    position = {key: i for i, key in enumerate(keys)}
    return keys, *(tuple(position[key] for key in values) for values in labels)


def _pairs(reference, other, left_map, right_map, matching):
    ids = reference.relation_ids, other.relation_ids
    if matching not in {"auto", "identity", "support"}:
        raise ValueError("difference matching must be auto, identity or support")
    if matching == "auto":
        if (ids[0] is None) != (ids[1] is None):
            raise ValueError("one endpoint lacks relation IDs; choose support matching explicitly")
        matching = "support" if ids[0] is None else "identity"
    if matching == "identity":
        if any(value is None for value in ids):
            raise ValueError("identity matching requires relation IDs on both endpoints")
        positions = [{int(key): i for i, key in enumerate(values)} for values in ids]
        keys = tuple(dict.fromkeys((*positions[0], *positions[1])))
        return matching, tuple((positions[0].get(key), positions[1].get(key)) for key in keys)
    groups = [defaultdict(list), defaultdict(list)]
    for rex, mapper, grouped in zip((reference, other), (left_map, right_map), groups, strict=True):
        for edge, support in enumerate(rex.relation_supports()):
            span = tuple(mapper[int(v)] for v in support)
            # Multiplicity distinguishes a cancelling loop from a witness.
            key = tuple(sorted(span))
            orientation = (span[0], tuple(sorted(span[1:]))) if span else (None, ())
            grouped[key].append((orientation, edge))
    pairs = []
    for support in dict.fromkeys((*groups[0], *groups[1])):
        available = defaultdict(deque)
        for orientation, edge in groups[0][support]:
            available[orientation].append(edge)
        unmatched = []
        for orientation, edge in groups[1][support]:
            if available[orientation]:
                pairs.append((available[orientation].popleft(), edge))
            else:
                unmatched.append(edge)
        remaining = deque(edge for edges in available.values() for edge in edges)
        pairs.extend((remaining.popleft() if remaining else None, edge) for edge in unmatched)
        pairs.extend((edge, None) for edge in remaining)
    return matching, tuple(pairs)


@dataclass(frozen=True)
class BoundaryDifference:
    """An owned Q operator on union axes, not a canonical RexGraph boundary."""

    reference: object = field(repr=False)
    other: object = field(repr=False)
    ref_labels: tuple | None = field(default=None, repr=False)
    other_labels: tuple | None = field(default=None, repr=False)
    matching: str = "auto"
    vertex_keys: tuple = field(init=False)
    relation_pairs: tuple = field(init=False)
    entries: tuple = field(init=False)
    source_digests: tuple[str, str] = field(init=False)

    def __post_init__(self):
        from rexgraph.graph import RexGraph
        from rexgraph.io.catalog import object_digest
        reference, other = self.reference, self.other
        if not isinstance(reference, RexGraph) or not isinstance(other, RexGraph):
            raise TypeError("boundary difference requires two native RexGraph endpoints")
        reference._ensure_clean()
        other._ensure_clean()
        labels = tuple(value if value is None or isinstance(value, (str, bytes)) else tuple(value)
                       for value in (self.ref_labels, self.other_labels))
        keys, left_map, right_map = vertex_alignment(reference, other, *labels)
        matching, pairs = _pairs(reference, other, left_map, right_map, self.matching)
        columns = [primary_columns(rex) for rex in (reference, other)]
        entries = []
        for j, (a, b) in enumerate(pairs):
            column = {}
            for endpoint, index, mapper, sign in ((0, a, left_map, -1), (1, b, right_map, 1)):
                if index is not None:
                    for row, value in columns[endpoint][index].items():
                        i = mapper[row]
                        column[i] = column.get(i, Fraction(0)) + sign * value
            entries.extend((i, j, value) for i, value in sorted(column.items()) if value)
        for name, value in (("vertex_keys", keys), ("relation_pairs", pairs), ("entries", tuple(entries)),
                            ("matching", matching), ("source_digests", (object_digest(reference), object_digest(other)))):
            object.__setattr__(self, name, value)
        if self.ref_labels is not None:
            object.__setattr__(self, "ref_labels", labels[0])
            object.__setattr__(self, "other_labels", labels[1])

    @property
    def shape(self):
        return len(self.vertex_keys), len(self.relation_pairs)

    @property
    def nnz(self):
        return len(self.entries)

    def check_state(self):
        from rexgraph.io.catalog import object_digest
        if self.source_digests != (object_digest(self.reference), object_digest(self.other)):
            raise ValueError("difference endpoint changed after capture")

    def apply(self, values, *, exact=True):
        self.check_state()
        if not isinstance(exact, bool):
            raise TypeError("exact must be a boolean")
        return _action(self.entries, self.shape, values, exact=exact)

    def transpose_apply(self, values, *, exact=True):
        self.check_state()
        if not isinstance(exact, bool):
            raise TypeError("exact must be a boolean")
        return _action(tuple((j, i, v) for i, j, v in self.entries), self.shape[::-1], values, exact=exact)

    def as_native(self):
        self.check_state()
        from rexgraph.core import _sparse
        from rexgraph.native_sparse import NativeSparse
        rows = [i for i, _, _ in self.entries]
        cols = [j for _, j, _ in self.entries]
        return NativeSparse(_sparse.dual_from_coo(rows, cols, [float(v) for _, _, v in self.entries], *self.shape))

    def as_scipy(self):
        """Explicit numerical compatibility export; exact coefficients remain here."""
        return self.as_native().as_scipy().tocsc()

    @cached_property
    def _readings(self):
        columns = [{} for _ in self.relation_pairs]
        for i, j, value in self.entries:
            columns[j][i] = value
        rank = _rank_integer_columns(self.shape, clear_column_denominators(columns))[0]
        nonzero = sum(bool(c) for c in columns)
        return {"shape": self.shape, "agree": self.shape[1] - nonzero, "disagree": nonzero,
                "rank": rank, "nullity": self.shape[1] - rank,
                "frobenius2": sum((v*v for _, _, v in self.entries), Fraction(0)),
                "max_column_sum": max((abs(sum(c.values(), Fraction(0))) for c in columns), default=Fraction(0)),
                "in_both": sum(a is not None and b is not None for a, b in self.relation_pairs),
                "only_in_reference": sum(b is None for a, b in self.relation_pairs),
                "only_in_input": sum(a is None for a, b in self.relation_pairs),
                "matched_identical": sum(not c and a is not None and b is not None for c, (a, b) in zip(columns, self.relation_pairs, strict=True)),
                "unmapped_relations": 0}

    @property
    def readings(self):
        self.check_state()
        return dict(self._readings)


def boundary_difference(reference, other, *, ref_labels=None, other_labels=None, matching="auto"):
    return BoundaryDifference(reference, other, ref_labels, other_labels, matching)
