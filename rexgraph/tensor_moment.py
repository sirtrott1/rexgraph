"""Exact moment kernels with retained support and field coordinate pairs."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from rexgraph.chain_map import _exact_entries
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric, _identity, _is_action
from rexgraph.graded_metric import _fraction
from rexgraph.tensor_field import TensorField, TensorChannels, FieldSource, apply_tensor
from rexgraph.type_accession import CoordinateSpace

__all__ = ["CoordinatePairing", "RealizedPairing", "MomentSpan", "TensorMomentKernel", "TensorMoments"]


@dataclass(frozen=True)
class CoordinatePairing:
    """An ordered rational cross form without assumed positivity or symmetry."""
    left: CoordinateSpace
    right: CoordinateSpace
    entries: tuple

    def __post_init__(self):
        if not isinstance(self.left, CoordinateSpace) or not isinstance(self.right, CoordinateSpace):
            raise TypeError("cross forms require two named coordinate spaces")
        object.__setattr__(self, "entries", _exact_entries(self.entries, self.shape))

    @property
    def shape(self):
        return len(self.left.keys), len(self.right.keys)

    @property
    def coefficient_digest(self):
        return _identity(("coordinate_pairing_v1", (self.left.name, self.left.keys),
                          (self.right.name, self.right.keys),
                          tuple((i, j, hex(v.numerator), hex(v.denominator)) for i, j, v in self.entries)))

    @classmethod
    def metric(cls, metric):
        if not isinstance(metric, CoordinateMetric):
            raise TypeError("metric pairing requires a CoordinateMetric")
        return cls(metric.space, metric.space, metric.entries)

    @classmethod
    def from_cross_metric(cls, metric, complex_):
        from rexgraph.type_accession import CrossMetric
        if not isinstance(metric, CrossMetric) or not metric.exact:
            raise TypeError("bridge requires an exact CrossMetric")
        a, b = metric.left, metric.right
        if a.source is not complex_.source or b.source is not complex_.source:
            raise ValueError("cross metric and complex must share their native source")
        if a.cell_keys is not None or b.cell_keys is not None:
            raise ValueError("bridge requires canonical ambient order")
        return cls(a.coordinates or complex_.spaces[a.grade], b.coordinates or complex_.spaces[b.grade], metric.entries)

    @property
    def T(self):
        return CoordinatePairing(self.right, self.left, tuple((j, i, v) for i, j, v in self.entries))

    def _coordinate_action(self):
        return CoordinateMap(self.right, self.left, self.entries)

    def form_action(self, variance="chain"):
        from rexgraph.native_field import BilinearFormAction
        return BilinearFormAction(self._coordinate_action(), variance)

    def realize(self, left, right):
        return self, left, right


@dataclass(frozen=True)
class RealizedPairing:
    """A cross form through physical realizations and an ambient metric."""
    left_action: object
    right_action: object
    metric: CoordinateMetric

    def __post_init__(self):
        if not _is_action(self.left_action) or not _is_action(self.right_action):
            raise TypeError("realized pairings require certified actions")
        if (not isinstance(self.metric, CoordinateMetric) or self.left_action.codomain != self.metric.space
                or self.right_action.codomain != self.metric.space):
            raise ValueError("physical realizations must share the metric target")

    @property
    def left(self):
        return self.left_action.domain

    @property
    def right(self):
        return self.right_action.domain

    @property
    def coefficient_digest(self):
        return _identity(("realized_pairing_v1", self.left_action.coefficient_digest,
                          self.right_action.coefficient_digest, self.metric.coefficient_digest))

    @property
    def T(self):
        return RealizedPairing(self.right_action, self.left_action, self.metric)

    def realize(self, left, right):
        return CoordinatePairing.metric(self.metric), apply_tensor(self.left_action, left), apply_tensor(self.right_action, right)

    def _coordinate_action(self):
        from rexgraph.native_field import MetricAction
        from rexgraph.coordinate_map import CoordinateWord
        return CoordinateWord((self.right_action, MetricAction(self.metric), self.left_action.T))

    def form_action(self, variance="chain"):
        from rexgraph.native_field import BilinearFormAction
        return BilinearFormAction(self._coordinate_action(), variance)


@dataclass(frozen=True)
class MomentSpan:
    """A retained pair of fields and the form measuring their support contributions."""
    left: TensorField
    right: TensorField
    pairing: object
    channel_keys: tuple = ()
    kernel_digest: str | None = None
    endpoint_sources: tuple[FieldSource, ...] = ()

    def __post_init__(self):
        if not isinstance(self.left, TensorField) or not isinstance(self.right, TensorField):
            raise TypeError("moment span requires two tensor fields")
        if not isinstance(self.pairing, (CoordinatePairing, RealizedPairing)):
            raise TypeError("moment span requires an explicit cross form")
        if self.left.space != self.pairing.left or self.right.space != self.pairing.right:
            raise ValueError("moment fields do not match the form endpoints")
        if self.left.variance != self.right.variance:
            raise ValueError("moment fields require matching declared variance")
        object.__setattr__(self, "channel_keys", tuple(self.channel_keys))
        references = tuple(self.endpoint_sources)
        if any(not isinstance(s, FieldSource) for s in references):
            raise TypeError("moment endpoints require native FieldSources")
        object.__setattr__(self, "endpoint_sources", references)
        if len(self.channel_keys) not in (0, 2):
            raise ValueError("moment channel keys must identify both channels")

    @property
    def coefficient_digest(self):
        return _identity(("moment_span_v1", self.left.coefficient_digest, self.right.coefficient_digest,
                          self.pairing.coefficient_digest, self.channel_keys, self.kernel_digest,
                          tuple(s.coefficient_digest for s in self.endpoint_sources)))

    @property
    def dependencies(self):
        refs = (*self.endpoint_sources, *self.left.dependencies, *self.right.dependencies,
                *((self.left.source,) if self.left.source is not None else ()),
                *((self.right.source,) if self.right.source is not None else ()))
        return tuple({s.coefficient_digest: s for s in refs}.values())

    def _fields(self):
        for reference in self.dependencies:
            reference.check()
        self.left.check_state()
        self.right.check_state()
        return self.pairing.realize(self.left, self.right)

    def support(self):
        """Retain form incidences and every pair of remaining field coordinates."""
        form, left, right = self._fields()
        diagonal = form.left == form.right and all(i == j for i, j, _ in form.entries)
        if diagonal:
            space = form.left
            positions = tuple((i, i, i, w) for i, _, w in form.entries)
        else:
            import json
            keys = tuple(json.dumps((form.left.keys[i], form.right.keys[j]), ensure_ascii=False)
                         for i, j, _ in form.entries)
            space = CoordinateSpace("form_support/"+form.coefficient_digest, keys)
            positions = tuple((p, i, j, w) for p, (i, j, w) in enumerate(form.entries))
        axes = (tuple(CoordinateSpace("left/"+a.name, a.keys) for a in left.axes)
                + tuple(CoordinateSpace("right/"+a.name, a.keys) for a in right.axes))
        values = np.full((len(space.keys), *left.values.shape[1:], *right.values.shape[1:]), Fraction(0), dtype=object)
        for p, i, j, w in positions:
            values[p] += w*np.multiply.outer(left.values[i], right.values[j])
        return TensorField(space, values, axes, provenance=(self.coefficient_digest,
                           left.coefficient_digest, right.coefficient_digest), dependencies=self.dependencies)

    def contract_support(self):
        """Contract only the form support and retain all field coordinate pairs."""
        form, left, right = self._fields()
        result = np.full((*left.values.shape[1:], *right.values.shape[1:]), Fraction(0), dtype=object)
        for i, j, w in form.entries:
            result += w*np.multiply.outer(left.values[i], right.values[j])
        result.flags.writeable = False
        return result

    def contracted_field(self):
        axes = (tuple(CoordinateSpace("left/"+a.name, a.keys) for a in self.left.axes)
                + tuple(CoordinateSpace("right/"+a.name, a.keys) for a in self.right.axes))
        values = self.contract_support()
        return TensorField(CoordinateSpace("moment_total", ("total",)), values.reshape((1, *values.shape)),
                           axes, provenance=(self.coefficient_digest,), dependencies=self.dependencies)

    def scalar(self):
        if self.left.axes or self.right.axes:
            raise ValueError("scalar output requires explicit contraction of the retained field axes")
        return self.contract_support().item()

    def rate(self, interval):
        """Divide the moment contributions, not both of their input fields."""
        interval = _fraction(interval)
        if not interval:
            raise ValueError("a finite rate requires a nonzero interval")
        if isinstance(self.pairing, CoordinatePairing):
            pairing = CoordinatePairing(self.pairing.left, self.pairing.right,
                                        tuple((i,j,v/interval) for i,j,v in self.pairing.entries))
        else:
            from rexgraph.native_field import ActionSum
            pairing = RealizedPairing(self.pairing.left_action,
                                      ActionSum(((1/interval,self.pairing.right_action),)),
                                      self.pairing.metric)
        return MomentSpan(self.left,self.right,pairing,self.channel_keys,
                          self.kernel_digest,self.endpoint_sources)

    def realized(self):
        form, left, right = self._fields()
        return MomentSpan(left, right, form, self.channel_keys, self.kernel_digest, self.endpoint_sources)

    def as_record(self):
        return {"left": self.left.as_record(), "right": self.right.as_record(),
                "pairing_digest": self.pairing.coefficient_digest, "channel_keys": self.channel_keys,
                "kernel_digest": self.kernel_digest, "endpoint_sources": tuple(s.as_record() for s in self.endpoint_sources),
                "coefficient_domain": "Q"}


@dataclass(frozen=True)
class TensorMomentKernel:
    """Retained channel actions and declared cross forms, not a scalar summary."""
    names: tuple[str, ...]
    channels: tuple
    pairings: tuple = ()
    common_metric: CoordinateMetric | None = None
    endpoint_sources: tuple[FieldSource, ...] = ()

    def __post_init__(self):
        names, channels = tuple(self.names), tuple(self.channels)
        if (not channels or len(names) != len(channels) or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n for n in names)
                or any(not _is_action(c) for c in channels)):
            raise ValueError("kernel requires named exact channel actions")
        pairs = tuple((a, b, m) for a, b, m in self.pairings)
        if len({(a,b) for a,b,_ in pairs}) != len(pairs):
            raise ValueError("cross forms require unique ordered channel pairs")
        for a,b,m in pairs:
            if a not in names or b not in names or not isinstance(m, (CoordinatePairing, RealizedPairing)):
                raise ValueError("cross forms require declared channel names")
            if m.left != channels[names.index(a)].codomain or m.right != channels[names.index(b)].codomain:
                raise ValueError("cross form endpoints differ from their channel targets")
        if self.common_metric is not None and (not isinstance(self.common_metric, CoordinateMetric)
                or any(c.codomain != self.common_metric.space for c in channels)):
            raise ValueError("common metric must belong to every channel target")
        if not pairs and self.common_metric is None:
            raise ValueError("kernel requires explicit measurement forms")
        refs = tuple(self.endpoint_sources)
        if any(not isinstance(s, FieldSource) for s in refs):
            raise TypeError("kernel endpoints require native FieldSources")
        object.__setattr__(self, "endpoint_sources", refs)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "pairings", pairs)

    @property
    def coefficient_digest(self):
        return _identity(("tensor_moment_kernel_v1", self.names, tuple(c.coefficient_digest for c in self.channels),
                          tuple((a,b,m.coefficient_digest) for a,b,m in self.pairings),
                          None if self.common_metric is None else self.common_metric.coefficient_digest,
                          tuple(s.coefficient_digest for s in self.endpoint_sources)))

    @classmethod
    def from_family_metric(cls, family, complex_, *, source=None):
        """Retain the existing accession and physical realization factors."""
        from rexgraph.type_accession import FamilyMetric
        from rexgraph.native_field import bridge_action
        from rexgraph.coordinate_map import CoordinateWord
        if not isinstance(family, FamilyMetric):
            raise TypeError("native family bridge requires a FamilyMetric")
        family.check_state(exact=True)
        metric = family.metric
        if metric.source is not complex_.source or metric.cell_keys is not None:
            raise ValueError("family and complex require the same canonical native source")
        pairing = CoordinateMetric.diagonal(complex_.spaces[metric.grade], metric.weights)
        names = tuple(e.accession.name for e in family.realizations)
        channels = tuple(CoordinateWord((bridge_action(e.accession, complex_), bridge_action(e, complex_)))
                         for e in family.realizations)
        if source is not None and (not isinstance(source, FieldSource) or source.source is not complex_.source):
            raise ValueError("family source must identify the actual native object")
        return cls(names, channels, common_metric=pairing,
                   endpoint_sources=() if source is None else (source,))

    def select(self, names):
        """Select channels before evaluating their fields or any moment table."""
        names = tuple(names)
        if any(name not in self.names for name in names):
            raise KeyError("unknown selected channel")
        return TensorMomentKernel(names, tuple(self.channels[self.names.index(name)] for name in names),
            tuple((a,b,m) for a,b,m in self.pairings if a in names and b in names),
            self.common_metric, self.endpoint_sources)

    def evaluate_pair(self, left, right, left_field, right_field=None):
        """Evaluate one requested form block without the other channel fields."""
        for source in self.endpoint_sources:
            source.check()
        right_field = left_field if right_field is None else right_field
        a, b = self.channels[self.names.index(left)], self.channels[self.names.index(right)]
        return MomentSpan(apply_tensor(a, left_field), apply_tensor(b, right_field),
                          self.pairing(left, right), (left, right), self.coefficient_digest, self.endpoint_sources)

    def pairing(self, left, right):
        if left not in self.names or right not in self.names:
            raise KeyError((left,right))
        for a,b,m in self.pairings:
            if (a,b) == (left,right):
                return m
        if self.common_metric is not None:
            return CoordinatePairing.metric(self.common_metric)
        raise KeyError("no form was declared for this channel pair")

    def evaluate(self, fields, other=None):
        for source in self.endpoint_sources:
            source.check()
        def side(values):
            values = (values,)*len(self.names) if isinstance(values, TensorField) else tuple(values)
            if len(values) != len(self.names):
                raise ValueError("one input tensor is required per channel")
            return TensorChannels(self.names, tuple(apply_tensor(a,x) for a,x in zip(self.channels,values,strict=True)),
                                  self.coefficient_digest, self.endpoint_sources)
        left = side(fields)
        return TensorMoments(self, left, left if other is None else side(other))

    def form_action(self, left, right, variance="chain"):
        from rexgraph.coordinate_map import CoordinateWord
        from rexgraph.native_field import BilinearFormAction
        a, b = self.channels[self.names.index(left)], self.channels[self.names.index(right)]
        return BilinearFormAction(CoordinateWord((b, self.pairing(left,right)._coordinate_action(), a.T)), variance)

    def contract(self, field, weights):
        """Contract channels before measuring through a common target metric."""
        if self.common_metric is None:
            raise ValueError("direct channel contraction requires a common target metric")
        weights = tuple(_fraction(w) for w in weights)
        if len(weights) != len(self.channels):
            raise ValueError("explicit weights must match the channels")
        from rexgraph.native_field import ActionSum
        result = apply_tensor(ActionSum(tuple(zip(weights,self.channels,strict=True))),field)
        return MomentSpan(result,result,CoordinatePairing.metric(self.common_metric),kernel_digest=self.coefficient_digest, endpoint_sources=self.endpoint_sources)


@dataclass(frozen=True)
class TensorMoments:
    """Evaluated channel fields with lazy support moments for selected pairs."""
    kernel: TensorMomentKernel
    left: TensorChannels
    right: TensorChannels

    def __post_init__(self):
        if (not isinstance(self.kernel, TensorMomentKernel) or not isinstance(self.left,TensorChannels)
                or not isinstance(self.right,TensorChannels) or self.left.names != self.kernel.names
                or self.right.names != self.kernel.names):
            raise ValueError("evaluated moments must retain their original channel declarations")

    def pair(self,left,right):
        return MomentSpan(self.left.field(left),self.right.field(right),self.kernel.pairing(left,right),
                          (left,right),self.kernel.coefficient_digest,self.kernel.endpoint_sources)

    def as_record(self):
        return {"kernel_digest":self.kernel.coefficient_digest,"left":self.left.as_record(),
                "right":self.right.as_record(),"coefficient_domain":"Q"}
