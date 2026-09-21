"""Exact field actions on declared relational boundaries and metrics."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property
from numbers import Integral
import numpy as np

from rexgraph.chain_map import CoordinateComplex, _chain_residual
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric, CoordinateWord, _exact_values, _identity
from rexgraph.exact_green import (ExactSparse, exact_hodge_apply, exact_green_apply,
                                 exact_projector_apply, exact_hodge_frames)
from rexgraph.graded_metric import _fraction
from rexgraph.type_accession import CoordinateSpace

__all__ = ["NativeFieldCalculus", "FieldAction", "MetricAction", "ProjectorAction",
           "ActionSum", "NativeAction", "bridge_action", "BilinearFormAction"]


def _block(function, values, rows):
    if values.ndim == 1:
        return np.asarray(function(tuple(values)), dtype=object)
    result = np.empty((rows, values.shape[1]), dtype=object)
    for j in range(values.shape[1]):
        result[:, j] = function(tuple(values[:, j]))
    return result


def _stable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Fraction):
        return (hex(value.numerator), hex(value.denominator))
    if isinstance(value, (tuple, list)):
        return tuple(_stable(v) for v in value)
    digest = getattr(value, "coefficient_digest", None)
    if isinstance(digest, str):
        return (type(value).__name__, digest)
    raise TypeError("an exact action dependency requires a stable coefficient identity")


class _ExtendedComplex(CoordinateComplex):
    """Explicit empty upper grades beside an unchanged native source tower."""

    def __init__(self, base, top_grade):
        spaces = (*base.spaces, *(CoordinateSpace(f"C{k}", ())
                                  for k in range(len(base.spaces), top_grade + 1)))
        boundaries = (*base.boundaries, *( () for _ in range(len(base.spaces), top_grade + 1)))
        super().__init__(spaces, boundaries)
        object.__setattr__(self, "source", base.source)
        object.__setattr__(self, "_primary", base._primary)
        object.__setattr__(self, "_relation_ids", base._relation_ids)
        object.__setattr__(self, "_base", base)

    def check_state(self):
        self._base.check_state()


@dataclass(frozen=True, eq=False)
class NativeFieldCalculus:
    """A verified boundary tower and its declared rational metric tower."""

    complex: CoordinateComplex
    metrics: tuple

    def __post_init__(self):
        if not isinstance(self.complex, CoordinateComplex):
            raise TypeError("field calculus requires a CoordinateComplex")
        self.complex.check_state()
        if _chain_residual(self.complex):
            raise ValueError("boundary tower does not satisfy the chain condition")
        metrics = tuple(self.metrics)
        if len(metrics) != len(self.complex.spaces) or any(
                not isinstance(m, CoordinateMetric) or m.space != s
                for m, s in zip(metrics, self.complex.spaces, strict=True)):
            raise ValueError("each grade requires a positive rational metric on its named space")
        object.__setattr__(self, "metrics", metrics)

    @classmethod
    def from_rex(cls, source, metrics=None, *, top_grade=None):
        complex_ = CoordinateComplex.from_rex(source)
        if top_grade is not None:
            if (isinstance(top_grade, bool) or not isinstance(top_grade, Integral)
                    or top_grade < len(complex_.spaces) - 1):
                raise ValueError("top grade must retain the complete native tower")
            if top_grade >= len(complex_.spaces):
                complex_ = _ExtendedComplex(complex_, int(top_grade))
        if metrics is None:
            weights = getattr(source, "edge_metric_exact", None)
            metrics = tuple(CoordinateMetric.diagonal(s, weights)
                            if k == 1 and weights is not None else CoordinateMetric.identity(s)
                            for k, s in enumerate(complex_.spaces))
        return cls(complex_, metrics)

    @property
    def source(self):
        return self.complex.source

    @property
    def coefficient_digest(self):
        return _identity(("native_field_v1", self.complex.coefficient_digest,
                          tuple(m.coefficient_digest for m in self.metrics)))

    def check_state(self):
        self.complex.check_state()

    def _grade(self, grade):
        if isinstance(grade, bool) or not isinstance(grade, Integral) or not 0 <= grade < len(self.metrics):
            raise ValueError("grade is outside the declared tower")
        return int(grade)

    def adjacent(self, grade):
        k = self._grade(grade)
        n = self.complex.sizes[k]
        lower = (CoordinateMap(self.complex.spaces[k], self.complex.spaces[k-1],
                               self.complex.boundaries[k-1]).as_sparse() if k else ExactSparse(0, n, {}))
        upper = (CoordinateMap(self.complex.spaces[k+1], self.complex.spaces[k],
                               self.complex.boundaries[k]).as_sparse()
                 if k+1 < len(self.metrics) else ExactSparse(n, 0, {}))
        return (lower, upper, self.metrics[k-1].as_sparse() if k else ExactSparse.identity(0),
                self.metrics[k].as_sparse(), self.metrics[k+1].as_sparse()
                if k+1 < len(self.metrics) else ExactSparse.identity(0))

    def boundary(self, grade):
        return FieldAction(self, "boundary", self._grade(grade))

    def hodge(self, grade):
        return FieldAction(self, "hodge", self._grade(grade))

    def green(self, grade, parameter=1):
        return FieldAction(self, "green", self._grade(grade), _fraction(parameter))

    def sector(self, grade, sector):
        return FieldAction(self, sector, self._grade(grade))

    def split(self, grade, values):
        parts = tuple(self.sector(grade, s).apply(values) for s in ("gradient", "curl", "harmonic"))
        if not np.array_equal(sum(parts), _exact_values(values, self.complex.sizes[grade])):
            raise ArithmeticError("sector fields do not reconstruct their source")
        return parts


@dataclass(frozen=True, eq=False)
class FieldAction:
    """A boundary, Hodge, Green or sector action without an assembled operator."""

    owner: NativeFieldCalculus
    operation: str
    grade: int
    parameter: Fraction = Fraction(1)
    transposed: bool = False

    def __post_init__(self):
        if not isinstance(self.owner, NativeFieldCalculus):
            raise TypeError("field action requires NativeFieldCalculus")
        object.__setattr__(self, "grade", self.owner._grade(self.grade))
        if self.operation not in {"boundary", "hodge", "green", "gradient", "curl", "harmonic"}:
            raise ValueError("unknown exact field action")
        if self.operation == "boundary" and self.grade == 0:
            raise ValueError("grade zero has no boundary in this tower")
        parameter = _fraction(self.parameter)
        if self.operation == "green" and parameter < 0:
            raise ValueError("positive Green action requires a nonnegative parameter")
        if not isinstance(self.transposed, bool):
            raise TypeError("transpose flag must be boolean")
        object.__setattr__(self, "parameter", parameter)

    @property
    def domain_variance(self):
        return "cochain" if self.transposed else "chain"

    codomain_variance = domain_variance

    @property
    def domain_grade(self):
        return self.grade-1 if self.operation == "boundary" and self.transposed else self.grade

    @property
    def codomain_grade(self):
        return self.grade-1 if self.operation == "boundary" and not self.transposed else self.grade

    @property
    def domain(self):
        return self.owner.complex.spaces[self.domain_grade]

    @property
    def codomain(self):
        return self.owner.complex.spaces[self.codomain_grade]

    @property
    def shape(self):
        return len(self.codomain.keys), len(self.domain.keys)

    @property
    def coefficient_digest(self):
        return _identity(("field_action_v1", self.owner.coefficient_digest, self.operation,
                          self.grade, _stable(self.parameter), self.transposed))

    @property
    def T(self):
        return replace(self, transposed=not self.transposed)

    @cached_property
    def _frames(self):
        return exact_hodge_frames(*self.owner.adjacent(self.grade))

    def check_state(self):
        self.owner.check_state()

    def apply(self, values):
        self.check_state()
        values = _exact_values(values, self.shape[1])
        lower, upper, ml, metric, mu = self.owner.adjacent(self.grade)
        if self.operation == "boundary":
            matrix = lower.T if self.transposed else lower
            return _block(matrix.apply, values, self.shape[0])
        def forward(x):
            if self.operation == "hodge":
                return exact_hodge_apply(lower, upper, ml, metric, mu, x)
            if self.operation == "green":
                return exact_green_apply(lower, upper, ml, metric, mu, self.parameter, x)
            frame = self._frames[("gradient", "curl", "harmonic").index(self.operation)]
            return exact_projector_apply(frame, metric, x)
        def action(x):
            return metric.apply(forward(metric.solve(x))) if self.transposed else forward(x)
        return _block(action, values, self.shape[0])

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))


@dataclass(frozen=True)
class MetricAction:
    """Apply or solve a positive rational metric."""
    metric: CoordinateMetric
    inverse: bool = False

    def __post_init__(self):
        if not isinstance(self.metric, CoordinateMetric) or not isinstance(self.inverse, bool):
            raise TypeError("metric action requires a metric and boolean solve flag")

    @property
    def domain(self):
        return self.metric.space

    codomain = domain

    @property
    def shape(self):
        return (len(self.domain.keys),) * 2

    @property
    def coefficient_digest(self):
        return _identity(("metric_action_v1", self.metric.coefficient_digest, self.inverse))

    @property
    def T(self):
        return self

    def apply(self, values):
        return self.metric.solve(values) if self.inverse else self.metric.apply(values)

    transpose_apply = apply

    def then(self, following):
        return CoordinateWord((self, following))


@dataclass(frozen=True)
class ProjectorAction:
    """Projection through an independent frame without assembling the projector."""
    frame: CoordinateMap
    metric: CoordinateMetric
    transposed: bool = False

    def __post_init__(self):
        if (not isinstance(self.frame, CoordinateMap) or not isinstance(self.metric, CoordinateMetric)
                or self.frame.codomain != self.metric.space):
            raise ValueError("projector frame and metric must share a named target")
        if len(self.frame.as_sparse().rref()[1]) != self.frame.shape[1]:
            raise ValueError("projector frame must be independent")
        if not isinstance(self.transposed, bool):
            raise TypeError("transpose flag must be boolean")

    @property
    def domain(self):
        return self.metric.space

    codomain = domain

    @property
    def shape(self):
        return (len(self.domain.keys),) * 2

    @property
    def coefficient_digest(self):
        return _identity(("projector_action_v1", self.frame.coefficient_digest,
                          self.metric.coefficient_digest, self.transposed))

    @property
    def T(self):
        return replace(self, transposed=not self.transposed)

    def apply(self, values):
        values = _exact_values(values, self.shape[1])
        frame, metric = self.frame.as_sparse(), self.metric.as_sparse()
        def action(x):
            if self.transposed:
                return metric.apply(exact_projector_apply(frame, metric, metric.solve(x)))
            return exact_projector_apply(frame, metric, x)
        return _block(action, values, self.shape[0])

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))


@dataclass(frozen=True)
class ActionSum:
    """A retained rational sum on one named source and target."""
    terms: tuple

    def __post_init__(self):
        from rexgraph.coordinate_map import _is_action
        terms = tuple((_fraction(w), a) for w, a in self.terms)
        if not terms or any(not _is_action(a) for _, a in terms):
            raise TypeError("an action sum requires declared rational terms")
        first = terms[0][1]
        if any(a.domain != first.domain or a.codomain != first.codomain for _, a in terms):
            raise ValueError("sum terms require identical named endpoints")
        object.__setattr__(self, "terms", terms)

    @property
    def domain(self):
        return self.terms[0][1].domain

    @property
    def codomain(self):
        return self.terms[0][1].codomain

    @property
    def shape(self):
        return len(self.codomain.keys), len(self.domain.keys)

    @property
    def coefficient_digest(self):
        return _identity(("action_sum_v1", tuple((_stable(w), a.coefficient_digest) for w, a in self.terms)))

    @property
    def T(self):
        return ActionSum(tuple((w, a.T) for w, a in self.terms))

    def apply(self, values):
        values = _exact_values(values, self.shape[1])
        result = np.full((self.shape[0], *values.shape[1:]), Fraction(0), dtype=object)
        for weight, action in self.terms:
            result += weight * action.apply(values)
        return result

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))


@dataclass(frozen=True, eq=False)
class NativeAction:
    """A checked bridge for an existing exact native operator or column factor."""
    owner: object
    complex: CoordinateComplex
    domain: CoordinateSpace
    codomain: CoordinateSpace
    transposed: bool = False
    _kind: str = field(init=False, repr=False)
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        from rexgraph.column_expansion import ColumnLegs, PrimaryColumnLift
        from rexgraph.linear_operator import RexOperator
        from rexgraph.type_accession import TypeAccession, TypeRealization
        if not isinstance(self.complex, CoordinateComplex) or self.complex.source is None:
            raise TypeError("native bridge requires a source bound coordinate complex")
        if not isinstance(self.transposed, bool):
            raise TypeError("transpose flag must be boolean")
        owner, spaces = self.owner, self.complex.spaces
        if isinstance(owner, RexOperator):
            if (owner.source is not self.complex.source or owner.exact_matvec is None
                    or owner.exact_transpose_matvec is None or owner.construction == "external-operator"):
                raise ValueError("native operator lacks the required exact source contract")
            left, right = spaces[owner.domain_grade], spaces[owner.codomain_grade]
            kind = "operator"
            identity = (owner.name, owner.construction, _stable(owner.parameters), _stable(owner.boundary_entries))
        elif isinstance(owner, (ColumnLegs, PrimaryColumnLift)):
            if owner.source is not self.complex.source:
                raise ValueError("column factor belongs to another primary source")
            owner.expansion.check_state()
            legs = CoordinateSpace("legs/"+owner.expansion.coefficient_digest,
                                   tuple("leg:"+str(i) for i in range(owner.expansion.legs.shape[1])))
            left, right = ((legs, spaces[owner.grade-1]) if isinstance(owner, ColumnLegs)
                           else (spaces[owner.grade], legs))
            kind, identity = "factor", (owner.expansion.coefficient_digest, type(owner).__name__)
        elif isinstance(owner, (TypeAccession, TypeRealization)):
            accession = owner if isinstance(owner, TypeAccession) else owner.accession
            if accession.source is not self.complex.source or not owner.exact or accession.cell_keys is not None:
                raise ValueError("bridge requires exact accessions in canonical source order")
            target = accession.coordinates or spaces[accession.grade]
            left, right = ((spaces[accession.grade], target) if isinstance(owner, TypeAccession)
                           else (target, spaces[accession.grade]))
            kind, identity = "entries", (owner.coefficient_digest, accession.name)
        else:
            raise TypeError("native owner has no supported exact bridge")
        if self.transposed:
            left, right = right, left
        if self.domain != left or self.codomain != right:
            raise ValueError("bridge endpoints must be the original named coordinates")
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_digest", _identity(("native_action_v1", self.complex.coefficient_digest,
                         identity, self.transposed, (left.name, left.keys), (right.name, right.keys))))
        self.check_state()

    @property
    def shape(self):
        return len(self.codomain.keys), len(self.domain.keys)

    @property
    def coefficient_digest(self):
        return self._digest

    @property
    def T(self):
        return NativeAction(self.owner, self.complex, self.codomain, self.domain, not self.transposed)

    def check_state(self):
        self.complex.check_state()
        expansion = getattr(self.owner, "expansion", None)
        if expansion is not None:
            expansion.check_state()

    def apply(self, values):
        self.check_state()
        values = _exact_values(values, self.shape[1])
        if self._kind == "operator":
            function = self.owner.exact_transpose_matvec if self.transposed else self.owner.exact_matvec
            result = function(values)
        elif self._kind == "factor":
            result = self.owner.apply(values, exact=True, transpose=self.transposed)
        else:
            left, right = ((self.codomain, self.domain) if self.transposed else (self.domain, self.codomain))
            action = CoordinateMap(left, right, self.owner.entries)
            result = (action.T if self.transposed else action).apply(values)
        return _exact_values(result, self.shape[0])

    def transpose_apply(self, values):
        return self.T.apply(values)

    def then(self, following):
        return CoordinateWord((self, following))


def bridge_action(owner, complex_):
    """Expose exact native actions while retaining their original source ownership."""
    from rexgraph.column_expansion import ColumnLegs, PrimaryColumnLift
    from rexgraph.linear_operator import RexOperator
    from rexgraph.type_accession import TypeAccession, TypeRealization
    if isinstance(owner, RexOperator):
        left, right = complex_.spaces[owner.domain_grade], complex_.spaces[owner.codomain_grade]
    elif isinstance(owner, (ColumnLegs, PrimaryColumnLift)):
        legs = CoordinateSpace("legs/"+owner.expansion.coefficient_digest,
                               tuple("leg:"+str(i) for i in range(owner.expansion.legs.shape[1])))
        left, right = ((legs, complex_.spaces[owner.grade-1]) if isinstance(owner, ColumnLegs)
                       else (complex_.spaces[owner.grade], legs))
    elif isinstance(owner, (TypeAccession, TypeRealization)):
        a = owner if isinstance(owner, TypeAccession) else owner.accession
        target = a.coordinates or complex_.spaces[a.grade]
        left, right = ((complex_.spaces[a.grade], target) if isinstance(owner, TypeAccession)
                       else (target, complex_.spaces[a.grade]))
    else:
        raise TypeError("unsupported native action owner")
    return NativeAction(owner, complex_, left, right)


@dataclass(frozen=True)
class BilinearFormAction:
    """A form action with an explicit vector to covector variance change."""
    action: object
    domain_variance: str = "chain"

    def __post_init__(self):
        from rexgraph.coordinate_map import _is_action
        if not _is_action(self.action) or self.domain_variance not in {"chain","cochain"}:
            raise ValueError("form actions require a certified action and explicit input variance")

    @property
    def codomain_variance(self):
        return "cochain" if self.domain_variance=="chain" else "chain"

    @property
    def domain(self):
        return self.action.domain

    @property
    def codomain(self):
        return self.action.codomain

    @property
    def shape(self):
        return self.action.shape

    @property
    def coefficient_digest(self):
        return _identity(("bilinear_form_action_v1",self.action.coefficient_digest,self.domain_variance))

    @property
    def T(self):
        return BilinearFormAction(self.action.T,self.domain_variance)

    def apply(self,values):
        return self.action.apply(values)

    def transpose_apply(self,values):
        return self.T.apply(values)

    def then(self,following):
        return CoordinateWord((self,following))
