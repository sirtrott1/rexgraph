"""Factored operator brackets on one cell space or a complete Chain tower.

[A,B] = AB BA and {A,B} = AB+BA. These are ordinary, not Koszul graded,
brackets. The rightmost action runs first; no product matrix is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.cochain import Chain
from rexgraph.linear_operator import RexOperator, _boundaries, _grade_sizes, _numeric_array
from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator
from rexgraph.weighted_hodge import WeightedHodgeOperator


@dataclass(frozen=True)
class OperatorBracket(RexOperator):
    left: RexOperator | None = None
    right: RexOperator | None = None
    anti: bool = False
    grade_metrics: tuple = ()
    metric_self_adjoint: bool | None = None
    metric_skew_adjoint: bool | None = None
    euclidean_skew_adjoint: bool | None = None


@dataclass(frozen=True)
class GradedOperatorBracket:
    source: object
    sizes: tuple[int, ...]
    left: object
    right: object
    anti: bool
    grade_metrics: tuple = ()
    metric_self_adjoint: bool | None = None
    metric_skew_adjoint: bool | None = None

    @property
    def name(self):
        return "ANTICOMMUTATOR" if self.anti else "COMMUTATOR"

    @property
    def shape(self):
        return (sum(self.sizes),)*2

    @property
    def exact(self):
        return self.left.exact and self.right.exact

    @property
    def active_boundaries(self):
        return tuple(sorted(set(self.left.active_boundaries) | set(self.right.active_boundaries)))

    def apply(self, state, *, exact=False):
        return self._apply(state, exact=exact, transpose=False)

    def transpose_apply(self, state, *, exact=False):
        return self._apply(state, exact=exact, transpose=True)

    def _apply(self, state, *, exact, transpose):
        if not isinstance(exact, (bool, np.bool_)):
            raise TypeError("exact must be a boolean")
        if not isinstance(state, GradedChain) or state.source is not self.source:
            raise TypeError("graded bracket requires a GradedChain bound to its source")
        if state.sizes != self.sizes or tuple(_grade_sizes(_boundaries(self.source))) != self.sizes:
            raise ValueError("graded bracket source populations changed; bind fresh state and operators")
        if exact and not self.exact:
            raise TypeError("graded bracket has no certified exact action")
        a, b = _products(self.left, self.right, state, exact=exact, transpose=transpose)
        return GradedChain(self.source, [Chain(k, _sum(x.values, y.values, self.anti, exact), source=self.source)
            for k, (x, y) in enumerate(zip(a.components, b.components, strict=True))])


def _products(left, right, values, *, exact, transpose):
    if transpose:
        # (AB)* = B* A*, including complex conjugation in each native action.
        return (right.transpose_apply(left.transpose_apply(values, exact=exact), exact=exact),
                left.transpose_apply(right.transpose_apply(values, exact=exact), exact=exact))
    return (left.apply(right.apply(values, exact=exact), exact=exact),
            right.apply(left.apply(values, exact=exact), exact=exact))


def _sum(a, b, anti, exact):
    if np.shape(a) != np.shape(b):
        raise ValueError("bracket factors returned incompatible coefficient axes")
    with np.errstate(over="ignore", invalid="ignore"):
        result = a + b if anti else a - b
    if not exact and not np.all(np.isfinite(result)):
        raise FloatingPointError("operator bracket result is outside numerical range")
    return result


def _form(op):
    """Known adjoint sign in a declared form; None means no certificate."""
    if isinstance(op, WeightedDiracOperator):
        return op.grade_metrics, -1 if op.anti else 1
    if isinstance(op, WeightedHodgeOperator) and op.metric_self_adjoint:
        return (op.grade_metric,), 1
    if isinstance(op, (OperatorBracket, GradedOperatorBracket)):
        if op.metric_self_adjoint or op.metric_skew_adjoint:
            return op.grade_metrics, 1 if op.metric_self_adjoint else -1
        if isinstance(op, OperatorBracket) and op.euclidean_skew_adjoint:
            return (), -1
    if isinstance(op, RexOperator) and op.symmetric:
        return (), 1
    return None


def _certificate(left, right, anti):
    a, b = _form(left), _form(right)
    if a is None or b is None or len(a[0]) != len(b[0]):
        return (), None
    if any((x.grade, x.cell_keys, x.coefficient_digest) != (y.grade, y.cell_keys, y.coefficient_digest)
           for x, y in zip(a[0], b[0], strict=True)):
        return (), None
    return a[0], (1 if anti else -1)*a[1]*b[1]


def operator_bracket(left, right, *, anti=False):
    """Build a source bound AB +/- BA action, exact only when both factors are.

    Operands must be endomorphisms of the same canonical grade/variance or the
    same full Chain tower. Different metrics do not prevent composition, but no
    common metric adjointness is then inferred. Neither bracket is promised PSD.
    Keep the source stable: these handles do not create transaction snapshots.
    """
    if not isinstance(anti, (bool, np.bool_)):
        raise TypeError("anti must be a boolean")
    graded = (WeightedDiracOperator, GradedOperatorBracket)
    if isinstance(left, graded) and isinstance(right, graded):
        if left.source is not right.source or left.sizes != right.sizes:
            raise ValueError("graded bracket requires the same source and full graded spaces")
        if tuple(_grade_sizes(_boundaries(left.source))) != left.sizes:
            raise ValueError("graded bracket source populations changed")
        metrics, sign = _certificate(left, right, anti)
        return GradedOperatorBracket(left.source, left.sizes, left, right, bool(anti), metrics,
                                     True if sign == 1 else None, True if sign == -1 else None)
    if not isinstance(left, RexOperator) or not isinstance(right, RexOperator):
        raise TypeError("bracket requires two single-grade operators or two full-tower operators")
    if (left.source is None or left.source is not right.source or left.variance != right.variance
            or left.domain_grade != left.codomain_grade or right.domain_grade != right.codomain_grade
            or left.domain_grade != right.domain_grade or left.shape != right.shape
            or left.shape[0] != left.shape[1]):
        raise ValueError("bracket requires endomorphisms of the same source, grade, variance and cell axes")
    n = cell_count(left.source, left.domain_grade, allow_empty_upper=True)
    if left.shape != (n, n):
        raise ValueError("bracket operator population differs from its source")

    def action(v, *, exact=False, transpose=False):
        if cell_count(left.source, left.domain_grade, allow_empty_upper=True) != n:
            raise ValueError("bracket source population changed; bind fresh operators")
        if not exact:
            v = _numeric_array(v, operation="operator bracket")
        a, b = _products(left, right, v, exact=exact, transpose=transpose)
        result = _sum(a, b, anti, exact)
        if result.shape != v.shape:
            raise ValueError("bracket factors changed the vector/block shape")
        return result

    metrics, sign = _certificate(left, right, anti)
    return OperatorBracket(
        "ANTICOMMUTATOR" if anti else "COMMUTATOR", left.shape, left.domain_grade, left.codomain_grade,
        action, source=left.source, construction="operator-bracket", variance=left.variance,
        symmetric=not metrics and sign == 1, psd=False,
        transpose_matvec=(lambda v: action(v, transpose=True)) if left.has_transpose and right.has_transpose else None,
        exact_matvec=(lambda v: action(v, exact=True)) if left.exact_matvec and right.exact_matvec else None,
        exact_transpose_matvec=(lambda v: action(v, exact=True, transpose=True))
            if left.exact_transpose_matvec and right.exact_transpose_matvec else None,
        parameters=(("bracket", "anticommutator" if anti else "commutator"),),
        left=left, right=right, anti=bool(anti), grade_metrics=metrics,
        metric_self_adjoint=True if metrics and sign == 1 else None,
        metric_skew_adjoint=True if metrics and sign == -1 else None,
        euclidean_skew_adjoint=True if not metrics and sign == -1 else None,
    )
