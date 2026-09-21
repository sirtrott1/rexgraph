"""Retained temporal actions and exact channel moments."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.coordinate_map import (
    CoordinateMap, CoordinateWord, CoordinateDifference, CoordinateMetric,
    _exact_values, _identity, _is_action,
)
from rexgraph.type_accession import CoordinateSpace

__all__ = ["TemporalMetrics", "TemporalOperation", "TemporalWord", "TemporalDelta",
           "MomentKernel", "ChannelMoments", "injection_delta"]


def _labels(names, count):
    names = tuple(names)
    if len(names) != count or len(set(names)) != count or any(
            not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("channel names must be unique nonempty strings")
    return names


@dataclass(frozen=True)
class TemporalMetrics:
    """Explicit positive metric towers for a declared pair of complexes."""

    domain: CoordinateComplex
    codomain: CoordinateComplex
    old: tuple
    new: tuple

    def __post_init__(self):
        if not all(isinstance(c, CoordinateComplex) for c in (self.domain, self.codomain)):
            raise TypeError("temporal metrics require declared endpoint complexes")
        for key, complex_ in (("old", self.domain), ("new", self.codomain)):
            metrics = tuple(getattr(self, key))
            if len(metrics) != len(complex_.spaces) or any(
                    not isinstance(m, CoordinateMetric) or m.space != s
                    for m, s in zip(metrics, complex_.spaces, strict=True)):
                raise ValueError("each endpoint metric must match its named grade space")
            object.__setattr__(self, key, metrics)

    @classmethod
    def identity(cls, mapping):
        mapping = getattr(mapping, "declaration", mapping)
        if not isinstance(mapping, GradedMap):
            raise TypeError("identity metrics require a graded correspondence")
        return cls(mapping.domain, mapping.codomain,
                   tuple(CoordinateMetric.identity(s) for s in mapping.domain.spaces),
                   tuple(CoordinateMetric.identity(s) for s in mapping.codomain.spaces))

    def check(self, mapping):
        mapping = getattr(mapping, "declaration", mapping)
        if mapping.domain is not self.domain or mapping.codomain is not self.codomain:
            raise ValueError("metric towers must belong to the actual correspondence endpoints")
        mapping.check_state()

    @property
    def coefficient_digest(self):
        return _identity(("temporal-metrics-v1", self.domain.coefficient_digest,
                          self.codomain.coefficient_digest,
                          tuple(m.coefficient_digest for m in self.old),
                          tuple(m.coefficient_digest for m in self.new)))


@dataclass(frozen=True)
class ChannelMoments:
    """Evaluated channel moments, including their interaction entries."""

    names: tuple[str, ...]
    values: np.ndarray
    kernel_digest: str | None = None

    def __post_init__(self):
        names = _labels(self.names, len(self.names))
        values = _exact_values(self.values, len(names))
        if values.shape != (len(names), len(names)):
            raise ValueError("moment axes must match the channel names")
        values = values.copy()
        values.flags.writeable = False
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    def contract(self, weights=None):
        w = (np.asarray([Fraction(1)] * len(self.names), dtype=object) if weights is None
             else _exact_values(weights, len(self.names)))
        if w.ndim != 1:
            raise ValueError("channel contraction requires one weight per channel")
        return sum((w[i] * self.values[i, j] * w[j] for i in range(len(w))
                    for j in range(len(w))), Fraction(0))

    def as_record(self):
        return {"names": self.names, "values": tuple(tuple(r) for r in self.values.tolist()),
                "shape": self.values.shape, "total": self.contract(),
                "kernel_digest": self.kernel_digest, "coefficient_domain": "Q"}


def _moments(names, fields, metric, digest=None):
    n = len(fields)
    values = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(i, n):
            values[i, j] = values[j, i] = metric.moment(fields[i], fields[j])
    return ChannelMoments(names, values, digest)


@dataclass(frozen=True)
class TemporalDelta:
    """Named contributions to a field change in one declared target space."""

    space: CoordinateSpace
    names: tuple[str, ...]
    fields: tuple
    operation_digest: str | None = None

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("a temporal delta requires its target coordinate space")
        fields = tuple(_exact_values(v, len(self.space.keys)).copy() for v in self.fields)
        if not fields or any(v.shape != fields[0].shape for v in fields):
            raise ValueError("delta channels require matching vector or block shapes")
        names = _labels(self.names, len(fields))
        for values in fields:
            values.flags.writeable = False
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "names", names)

    @property
    def values(self):
        return sum(self.fields, np.full(self.fields[0].shape, Fraction(0), dtype=object))

    def moments(self, metric=None):
        metric = CoordinateMetric.identity(self.space) if metric is None else metric
        if not isinstance(metric, CoordinateMetric) or metric.space != self.space:
            raise ValueError("delta moments require a metric on the target space")
        return _moments(self.names, self.fields, metric, self.operation_digest)

    def as_record(self, metric=None):
        return {"names": self.names, "space_name": self.space.name, "space_keys": self.space.keys,
                "fields": tuple(tuple(v.tolist()) for v in self.fields),
                "values": tuple(self.values.tolist()), "shape": self.values.shape,
                "moments": self.moments(metric).as_record(),
                "operation_digest": self.operation_digest, "coefficient_domain": "Q"}


@dataclass(frozen=True)
class TemporalOperation:
    """One operation at two times with independent input and output maps."""

    old: object
    new: object
    input_map: CoordinateMap
    output_map: CoordinateMap

    def __post_init__(self):
        if not all(_is_action(a) for a in (self.old, self.new)) or not all(
                isinstance(m, CoordinateMap) for m in (self.input_map, self.output_map)):
            raise TypeError("temporal operations require explicit actions and correspondences")
        if (self.input_map.domain != self.old.domain or self.input_map.codomain != self.new.domain
                or self.output_map.domain != self.old.codomain
                or self.output_map.codomain != self.new.codomain):
            raise ValueError("temporal correspondence spaces must match the operation endpoints")

    @property
    def defect(self):
        return CoordinateDifference(CoordinateWord((self.input_map, self.new)),
                                    CoordinateWord((self.old, self.output_map)))

    @property
    def coefficient_digest(self):
        return _identity(("temporal-operation-v1", self.old.coefficient_digest,
                          self.new.coefficient_digest, self.input_map.coefficient_digest,
                          self.output_map.coefficient_digest))

    def field_delta(self, old_values, new_values, *, names=("operation", "field")):
        old_values = _exact_values(old_values, len(self.old.domain.keys))
        new_values = _exact_values(new_values, len(self.new.domain.keys))
        if old_values.shape[1:] != new_values.shape[1:]:
            raise ValueError("temporal fields require matching block axes")
        innovation = new_values - self.input_map.apply(old_values)
        result = TemporalDelta(self.new.codomain, names,
                               (self.defect.apply(old_values), self.new.apply(innovation)),
                               self.coefficient_digest)
        expected = self.new.apply(new_values) - self.output_map.apply(self.old.apply(old_values))
        if not np.array_equal(result.values, expected):
            raise ArithmeticError("temporal field decomposition failed")
        return result


@dataclass(frozen=True)
class MomentKernel:
    """Retained channel actions paired through one positive target form."""

    names: tuple[str, ...]
    channels: tuple
    metric: CoordinateMetric

    def __post_init__(self):
        channels = tuple(self.channels)
        if not channels or any(not _is_action(c) for c in channels):
            raise TypeError("a moment kernel requires declared channel actions")
        if any(c.domain != channels[0].domain or c.codomain != channels[0].codomain for c in channels):
            raise ValueError("kernel channels must share their named source and target")
        if not isinstance(self.metric, CoordinateMetric) or self.metric.space != channels[0].codomain:
            raise ValueError("kernel pairing must belong to the channel target")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "names", _labels(self.names, len(channels)))

    @property
    def domain(self):
        return self.channels[0].domain

    @property
    def coefficient_digest(self):
        return _identity(("moment-kernel-v1", self.names,
                          tuple(c.coefficient_digest for c in self.channels),
                          self.metric.coefficient_digest))

    def evaluate(self, values):
        fields = tuple(c.apply(values) for c in self.channels)
        return _moments(self.names, fields, self.metric, self.coefficient_digest)


@dataclass(frozen=True)
class TemporalWord:
    """Two operation words and a correspondence at every intermediate space."""

    old: tuple
    new: tuple
    correspondences: tuple[CoordinateMap, ...]
    names: tuple[str, ...] | None = None

    def __post_init__(self):
        old, new, maps = tuple(self.old), tuple(self.new), tuple(self.correspondences)
        CoordinateWord(old)
        CoordinateWord(new)
        if len(old) != len(new) or len(maps) != len(old) + 1:
            raise ValueError("a temporal word requires matching lengths and every intermediate map")
        for i in range(len(old)):
            TemporalOperation(old[i], new[i], maps[i], maps[i+1])
        names = tuple(f"operation:{i}" for i in range(len(old))) if self.names is None else self.names
        object.__setattr__(self, "old", old)
        object.__setattr__(self, "new", new)
        object.__setattr__(self, "correspondences", maps)
        object.__setattr__(self, "names", _labels(names, len(old)))

    @property
    def domain(self):
        return self.old[0].domain

    @property
    def codomain(self):
        return self.new[-1].codomain

    @property
    def coefficient_digest(self):
        return _identity(("temporal-word-v1", self.names,
                          tuple(a.coefficient_digest for a in self.old),
                          tuple(a.coefficient_digest for a in self.new),
                          tuple(a.coefficient_digest for a in self.correspondences)))

    @property
    def channels(self):
        return tuple(CoordinateWord((*self.old[:i],
                     TemporalOperation(a, b, self.correspondences[i], self.correspondences[i+1]).defect,
                     *self.new[i+1:])) for i, (a, b) in enumerate(zip(self.old, self.new, strict=True)))

    def delta(self, values):
        result = TemporalDelta(self.codomain, self.names,
                               tuple(c.apply(values) for c in self.channels), self.coefficient_digest)
        direct = TemporalOperation(CoordinateWord(self.old), CoordinateWord(self.new),
                                   self.correspondences[0], self.correspondences[-1]).defect.apply(values)
        if not np.array_equal(result.values, direct):
            raise ArithmeticError("temporal word decomposition failed")
        return result

    def moment_kernel(self, metric=None):
        metric = CoordinateMetric.identity(self.codomain) if metric is None else metric
        return MomentKernel(self.names, self.channels, metric)


def injection_delta(old, new, input_map, output_map, old_values, new_values):
    """Separate realization change from aligned source amplitude innovation."""
    return TemporalOperation(old, new, input_map, output_map).field_delta(
        old_values, new_values, names=("injection", "amplitude"))
