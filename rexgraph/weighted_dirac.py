"""Full tower Chain Dirac calculus with positive diagonal grade metrics.

D = boundary + metric adjoint; A = boundary - metric adjoint.
Grade blocks stay explicit. Neither a block matrix nor square root metric
coordinates are constructed. The Euclidean propagation API is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from rexgraph.cochain import Chain
from rexgraph.graded_metric import DiagonalMetric, diagonal_metric
from rexgraph.linear_operator import (
    _boundaries,
    _exact_array,
    _grade_sizes,
    _numeric_array,
    boundary_operator,
    metric_adjoint,
)


@dataclass(frozen=True, init=False)
class GradedChain:
    """Canonical direct sum of Chains, one component per carried grade.

    Missing seed grades are explicit zero; duplicates are errors, not addition.
    Components share one vector/block shape and coefficient domain. Exact seeds
    remain Q; any numerical seed selects real/complex arithmetic for the state.
    Inputs are copied and arrays made read only. Keep the bound source unchanged.
    """

    source: object
    components: tuple[Chain, ...]

    def __init__(self, source, components=()):
        if not isinstance(components, (tuple, list)):
            raise TypeError("graded Chain components must be a list or tuple of Chains")
        sizes = tuple(_grade_sizes(_boundaries(source)))
        supplied = {}
        trailing = None
        exact = True
        complex_ = False
        for c in components:
            if not isinstance(c, Chain) or c.source is not source:
                raise TypeError("graded Chain components require Chains bound to the same source")
            if c.grade in supplied:
                raise ValueError("duplicate graded Chain component")
            if c.cell_keys is not None or not 0 <= c.grade < len(sizes):
                raise ValueError("graded Chain requires canonical bases at carried grades")
            a = np.asarray(c.values)
            if a.ndim not in (1, 2) or a.shape[0] != sizes[c.grade]:
                raise ValueError("graded Chain component has the wrong cell or block axes")
            if trailing is not None and trailing != a.shape[1:]:
                raise ValueError("graded Chain components require matching vector/block shapes")
            trailing = a.shape[1:]
            is_q = all(isinstance(v, (Integral, Fraction)) and not isinstance(v, (bool, np.bool_))
                       for v in a.flat) and a.dtype.kind in "iuO"
            if not is_q:
                a = _numeric_array(a, operation="graded Chain")
            exact &= is_q
            complex_ |= np.iscomplexobj(a)
            supplied[c.grade] = a
        trailing = () if trailing is None else trailing
        result = []
        for k, n in enumerate(sizes):
            a = supplied.get(k)
            if a is None:
                a = np.full((n, *trailing), Fraction(0), dtype=object) if exact else np.zeros(
                    (n, *trailing), dtype=complex if complex_ else float)
            elif exact:
                a = _exact_array(a)
            else:
                a = _numeric_array(a, operation="graded Chain").astype(complex if complex_ else float, copy=True)
            a.setflags(write=False)
            result.append(Chain(k, a, source=source))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "components", tuple(result))

    @property
    def sizes(self):
        return tuple(c.n_cells for c in self.components)

    @property
    def exact(self):
        return all(c.values.dtype.kind == "O" and all(isinstance(v, Fraction) for v in c.values.flat)
                   for c in self.components)

    def component(self, grade):
        if isinstance(grade, (bool, np.bool_)) or not isinstance(grade, Integral):
            raise TypeError("component grade must be an integer")
        if not 0 <= grade < len(self.components):
            raise ValueError("component grade is not carried")
        return self.components[grade]


@dataclass(frozen=True)
class WeightedDiracOperator:
    """Full tower factored action, not a falsely single grade RexOperator.

    Metric self-adjointness/skew-adjointness do not certify Euclidean symmetry
    or PSD. D squared and A squared are Hodge sums with opposite signs only
    when the source chain law holds. Construction does not check that law.
    """

    source: object
    sizes: tuple[int, ...]
    grade_metrics: tuple[DiagonalMetric, ...]
    anti: bool
    _factors: tuple

    @property
    def name(self):
        return "ANTI_DIRAC" if self.anti else "DIRAC"

    @property
    def shape(self):
        n = sum(self.sizes)
        return n, n

    @property
    def active_boundaries(self):
        return tuple(k for k, _, _ in self._factors)

    @property
    def exact(self):
        return all(b.exact_matvec is not None and a.exact_matvec is not None for _, b, a in self._factors)

    @property
    def metric_self_adjoint(self):
        return not self.anti

    @property
    def metric_skew_adjoint(self):
        return self.anti

    def apply(self, state, *, exact=False):
        return self._apply(state, exact=exact, transpose=False)

    def transpose_apply(self, state, *, exact=False):
        """Hermitian coordinate transpose, distinct from the metric adjoint."""
        return self._apply(state, exact=exact, transpose=True)

    def _apply(self, state, *, exact, transpose):
        if not isinstance(exact, (bool, np.bool_)):
            raise TypeError("exact must be a boolean")
        if not isinstance(state, GradedChain) or state.source is not self.source:
            raise TypeError("Dirac requires a GradedChain bound to its source")
        if state.sizes != self.sizes or tuple(_grade_sizes(_boundaries(self.source))) != self.sizes:
            raise ValueError("Dirac source populations changed; bind fresh state and operator")
        if exact and not self.exact:
            raise TypeError("Dirac has no certified exact action for these boundaries and metrics")
        values = [_exact_array(c.values) if exact else _numeric_array(c.values, operation="Dirac")
                  for c in state.components]
        output = [np.full(a.shape, Fraction(0), dtype=object) if exact else np.zeros_like(a) for a in values]
        sign = -1 if self.anti else 1
        with np.errstate(over="ignore", invalid="ignore"):
            for k, b, adj in self._factors:
                if transpose:
                    output[k] += b.transpose_apply(values[k-1], exact=exact)
                    output[k-1] += sign * adj.transpose_apply(values[k], exact=exact)
                else:
                    output[k-1] += b.apply(values[k], exact=exact)
                    output[k] += sign * adj.apply(values[k-1], exact=exact)
        if not exact and any(not np.all(np.isfinite(a)) for a in output):
            raise FloatingPointError("Dirac result is outside numerical range")
        return GradedChain(self.source, tuple(Chain(k, a, source=self.source) for k, a in enumerate(output)))


def weighted_dirac(rex, *, metrics=None, anti=False):
    """D or A with explicitly supplied per grade positive diagonals.

    Supply any subset of metrics, at most one per carried grade; the others
    are identity. No relation/channel weighting or type restriction is inferred.
    Zero incidence factors require no metric arithmetic, preserving exact zero.
    """
    if not isinstance(anti, (bool, np.bool_)):
        raise TypeError("anti must be a boolean")
    if metrics is not None and not isinstance(metrics, (tuple, list)):
        raise TypeError("Dirac metrics must be a list or tuple")
    boundaries = _boundaries(rex)
    sizes = tuple(_grade_sizes(boundaries))
    selected = {}
    for m in (() if metrics is None else metrics):
        if not isinstance(m, DiagonalMetric) or m.source is not rex:
            raise TypeError("Dirac metrics must be bound positive diagonal metrics")
        if m.grade in selected:
            raise ValueError("duplicate Dirac metric grade")
        if m.cell_keys is not None or not 0 <= m.grade < len(sizes) or len(m.weights) != sizes[m.grade]:
            raise ValueError("Dirac metric requires canonical basis, carried grade and matching population")
        selected[m.grade] = m
    grade_metrics = tuple(selected[k] if k in selected else diagonal_metric(rex, k) for k in range(len(sizes)))
    factors = []
    for k, matrix in enumerate(boundaries, 1):
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Dirac requires finite boundary coefficients")
        if matrix.nnz:
            b = boundary_operator(rex, k)
            factors.append((k, b, metric_adjoint(b, grade_metrics[k], grade_metrics[k-1])))
    return WeightedDiracOperator(rex, sizes, grade_metrics, bool(anti), tuple(factors))
