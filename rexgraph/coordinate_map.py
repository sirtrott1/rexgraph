"""Exact actions and forms on explicitly named coordinate spaces."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json

import numpy as np

from rexgraph.chain_map import _columns, _exact_entries, _triples
from rexgraph.exact_green import ExactSparse
from rexgraph.graded_boundary import _exact_compose_columns
from rexgraph.graded_metric import _fraction
from rexgraph.type_accession import CoordinateSpace, _action

__all__ = ["CoordinateMap", "CoordinateWord", "CoordinateDifference", "CoordinateMetric"]


def _identity(parts):
    return sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _space_key(space):
    return (space.name, space.keys)


def _coefficient_key(entries):
    return tuple((i, j, hex(v.numerator), hex(v.denominator)) for i, j, v in entries)


def _exact_values(values, size):
    values = np.asarray(values, dtype=object)
    if values.ndim not in (1, 2) or values.shape[0] != size:
        raise ValueError("values must match the declared vector or block axis")
    return np.asarray([_fraction(v) for v in values.flat], dtype=object).reshape(values.shape)


@dataclass(frozen=True)
class CoordinateMap:
    """A rational map between named spaces, not an inferred cell boundary."""

    domain: CoordinateSpace
    codomain: CoordinateSpace
    entries: tuple

    def __post_init__(self):
        if not all(isinstance(s, CoordinateSpace) for s in (self.domain, self.codomain)):
            raise TypeError("coordinate maps require explicit endpoint spaces")
        object.__setattr__(self, "entries", _exact_entries(self.entries, self.shape))

    @property
    def shape(self):
        return (len(self.codomain.keys), len(self.domain.keys))

    @property
    def coefficient_digest(self):
        return _identity(("coordinate-map-v1", _space_key(self.domain),
                          _space_key(self.codomain), _coefficient_key(self.entries)))

    @classmethod
    def identity(cls, space):
        return cls(space, space, tuple((i, i, 1) for i in range(len(space.keys))))

    @property
    def T(self):
        return CoordinateMap(self.codomain, self.domain,
                             tuple((j, i, v) for i, j, v in self.entries))

    def apply(self, values):
        return _action(self.entries, self.shape, _exact_values(values, self.shape[1]), exact=True)

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))

    def compose(self, following):
        """Materialize the sparse entries of following after this map."""
        if not isinstance(following, CoordinateMap) or self.codomain != following.domain:
            raise ValueError("composition requires the same named middle space")
        columns = _exact_compose_columns(_columns(following.entries, following.shape[1]),
                                         _columns(self.entries, self.shape[1]))
        return CoordinateMap(self.domain, following.codomain, _triples(columns))

    def as_sparse(self):
        return ExactSparse(*self.shape, {(i, j): v for i, j, v in self.entries})


def _is_action(value):
    if isinstance(value, (CoordinateMap, CoordinateWord, CoordinateDifference)):
        return True
    from rexgraph.native_field import FieldAction, NativeAction, ActionSum, MetricAction, ProjectorAction, BilinearFormAction
    from rexgraph.attachment_field import IntervalAction, AttachmentAction
    return isinstance(value, (FieldAction, NativeAction, ActionSum, MetricAction,
                              ProjectorAction, IntervalAction, AttachmentAction, BilinearFormAction))


@dataclass(frozen=True)
class CoordinateWord:
    """Compatible factors in application order. Products remain unassembled."""

    factors: tuple

    def __post_init__(self):
        factors = tuple(self.factors)
        if not factors or any(not _is_action(v) for v in factors):
            raise TypeError("a coordinate word requires one or more declared actions")
        if any(a.codomain != b.domain for a, b in zip(factors[:-1], factors[1:], strict=True)):
            raise ValueError("word factors require matching named intermediate spaces")
        object.__setattr__(self, "factors", factors)

    @property
    def domain(self):
        return self.factors[0].domain

    @property
    def codomain(self):
        return self.factors[-1].codomain

    @property
    def shape(self):
        return (len(self.codomain.keys), len(self.domain.keys))

    @property
    def coefficient_digest(self):
        return _identity(("coordinate-word-v1", tuple(f.coefficient_digest for f in self.factors)))

    @property
    def T(self):
        return CoordinateWord(tuple(f.T for f in reversed(self.factors)))

    def apply(self, values):
        result = _exact_values(values, self.shape[1])
        for factor in self.factors:
            result = factor.apply(result)
        return result

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((*self.factors, following))


@dataclass(frozen=True)
class CoordinateDifference:
    """The difference of two actions with identical declared endpoints."""

    left: object
    right: object

    def __post_init__(self):
        if not all(_is_action(a) for a in (self.left, self.right)):
            raise TypeError("a coordinate difference requires declared actions")
        if self.left.domain != self.right.domain or self.left.codomain != self.right.codomain:
            raise ValueError("a difference requires the same named endpoint spaces")

    @property
    def domain(self):
        return self.left.domain

    @property
    def codomain(self):
        return self.left.codomain

    @property
    def shape(self):
        return self.left.shape

    @property
    def coefficient_digest(self):
        return _identity(("coordinate-difference-v1", self.left.coefficient_digest,
                          self.right.coefficient_digest))

    @property
    def T(self):
        return CoordinateDifference(self.left.T, self.right.T)

    def apply(self, values):
        return self.left.apply(values) - self.right.apply(values)

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))


@dataclass(frozen=True)
class CoordinateMetric:
    """A positive rational form on a named coordinate space."""

    space: CoordinateSpace
    entries: tuple

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("a coordinate metric requires a named space")
        n = len(self.space.keys)
        entries = _exact_entries(self.entries, (n, n))
        values = {(i, j): v for i, j, v in entries}
        if any(values.get((j, i), Fraction(0)) != v for (i, j), v in values.items()):
            raise ValueError("a coordinate metric must be symmetric")
        if all(i == j for i, j in values):
            if len(values) != n or any(v <= 0 for v in values.values()):
                raise ValueError("a coordinate metric must be positive definite")
            object.__setattr__(self, "entries", entries)
            return
        work = dict(values)
        for k in range(n):
            pivot = work.get((k, k), Fraction(0))
            if pivot <= 0:
                raise ValueError("a coordinate metric must be positive definite")
            tail = [(i, work[i, k]) for i in range(k + 1, n) if (i, k) in work]
            for i, a in tail:
                for j, b in tail:
                    value = work.get((i, j), Fraction(0)) - a * b / pivot
                    if value:
                        work[i, j] = value
                    else:
                        work.pop((i, j), None)
            for i, _ in tail:
                work.pop((i, k), None)
                work.pop((k, i), None)
            work.pop((k, k), None)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def identity(cls, space):
        return cls(space, tuple((i, i, 1) for i in range(len(space.keys))))

    @classmethod
    def diagonal(cls, space, weights):
        weights = tuple(_fraction(v) for v in weights)
        if len(weights) != len(space.keys):
            raise ValueError("metric weights must match the named space")
        return cls(space, tuple((i, i, w) for i, w in enumerate(weights)))

    @property
    def coefficient_digest(self):
        return _identity(("coordinate-metric-v1", _space_key(self.space),
                          _coefficient_key(self.entries)))

    def as_sparse(self):
        n = len(self.space.keys)
        return ExactSparse(n, n, {(i, j): v for i, j, v in self.entries})

    def apply(self, values):
        n = len(self.space.keys)
        return _action(self.entries, (n, n), _exact_values(values, n), exact=True)

    def solve(self, values):
        values = _exact_values(values, len(self.space.keys))
        if all(i == j for i, j, _ in self.entries):
            weights = {i: w for i, _, w in self.entries}
            result = values.copy()
            for i in range(len(self.space.keys)):
                result[i] = result[i] / weights[i]
            return result
        if values.ndim == 1:
            return np.asarray(self.as_sparse().solve(values), dtype=object)
        result = np.empty(values.shape, dtype=object)
        matrix = self.as_sparse()
        for j in range(values.shape[1]):
            result[:, j] = matrix.solve(values[:, j])
        return result

    def moment(self, left, right):
        left = _exact_values(left, len(self.space.keys))
        right = _exact_values(right, len(self.space.keys))
        if left.shape != right.shape:
            raise ValueError("a metric pairing requires matching vector or block shapes")
        return sum((a * b for a, b in zip(left.flat, self.apply(right).flat, strict=True)), Fraction(0))
