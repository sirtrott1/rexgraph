"""Positive diagonal Hodge calculus on Chains in canonical cell coordinates.

L_down = B_k^dagger B_k; L_up = B_(k+1) B_(k+1)^dagger.
The sum is metric PSD; the difference is metric self adjoint, not generally PSD.
Neither is implicitly Euclidean symmetric. No channel G is an upper sector.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.graded_metric import DiagonalMetric, diagonal_metric
from rexgraph.linear_operator import (
    RexOperator,
    _boundaries,
    _numeric_array,
    boundary_operator,
    metric_adjoint,
)

HODGE_SECTORS = frozenset({"down", "up", "sum", "difference"})


@dataclass(frozen=True)
class WeightedHodgeOperator(RexOperator):
    """Factored Chain endomorphism; metric certificates are not Euclidean ones.

    Keep the source unchanged while reusing the handle, like other Rex operators.
    """

    grade_metric: DiagonalMetric | None = None
    lower_metric: DiagonalMetric | None = None
    upper_metric: DiagonalMetric | None = None
    sector: str = "sum"
    active_sectors: tuple[str, ...] = ()
    metric_self_adjoint: bool = True
    metric_psd: bool | None = True


def weighted_hodge(rex, grade, *, sector="sum", metric=None, lower_metric=None, upper_metric=None):
    """Native down/up/sum/difference with explicit endpoint metric coefficients.

    Exact application uses Q incidence/metric factors, never reconstructed floats.
    Missing or zero incidence sectors are exact zero, independent of the metric;
    supplied metrics and all input coefficients are still validated. Explicitly
    irrelevant metrics are refused. Metric defaults are identity, not channels or
    relation weights. No Gram, eigensystem, or matrix inverse is assembled.
    """
    if isinstance(grade, (bool, np.bool_)) or not isinstance(grade, Integral):
        raise TypeError("Hodge grade must be an integer")
    grade = int(grade)
    if not isinstance(sector, str) or sector not in HODGE_SECTORS:
        raise ValueError("Hodge sector must be down, up, sum or difference")
    n = cell_count(rex, grade)  # The acted on grade must actually be carried.
    if (grade == 0 or sector == "up") and lower_metric is not None:
        raise ValueError("this Hodge action has no lower metric input")
    if sector == "down" and upper_metric is not None:
        raise ValueError("the down Hodge action has no upper metric input")

    def selected(supplied, k):
        m = diagonal_metric(rex, k) if supplied is None else supplied
        if (not isinstance(m, DiagonalMetric) or m.source is not rex or m.grade != k
                or len(m.weights) != cell_count(rex, k, allow_empty_upper=True)):
            raise ValueError("Hodge metric must match its source, grade and population")
        if m.cell_keys is not None:
            raise ValueError("Hodge metrics require the canonical ordered basis")
        return m

    center = selected(metric, grade)
    lower = selected(lower_metric, grade-1) if grade > 0 and sector != "up" else None
    upper = selected(upper_metric, grade+1) if sector != "down" else None
    boundaries = _boundaries(rex)
    factors = []
    active = []
    if lower is not None and boundaries[grade-1].nnz:
        if not np.all(np.isfinite(boundaries[grade-1].data)):
            raise ValueError("Hodge requires finite boundary coefficients")
        b = boundary_operator(rex, grade)
        adj = metric_adjoint(b, center, lower)
        factors.append((b, adj, 1))  # adj @ b
        active.append("down")
    if upper is not None and grade < len(boundaries) and boundaries[grade].nnz:
        if not np.all(np.isfinite(boundaries[grade].data)):
            raise ValueError("Hodge requires finite boundary coefficients")
        b = boundary_operator(rex, grade+1)
        adj = metric_adjoint(b, upper, center)
        factors.append((adj, b, -1 if sector == "difference" else 1))
        active.append("up")

    def action(values, *, exact=False, transpose=False):
        if not exact:
            values = _numeric_array(values, operation="weighted Hodge")
        out = (np.full(values.shape, Fraction(0), dtype=object) if exact else np.zeros_like(values))
        with np.errstate(over="ignore", invalid="ignore"):
            for first, second, sign in factors:
                if transpose:
                    contribution = first.transpose_apply(second.transpose_apply(values, exact=exact), exact=exact)
                else:
                    contribution = second.apply(first.apply(values, exact=exact), exact=exact)
                out += sign * contribution
        if not exact and not np.all(np.isfinite(out)):
            raise FloatingPointError("weighted Hodge result is outside numerical range")
        return out

    exact = all(a.exact_matvec is not None and b.exact_matvec is not None for a, b, _ in factors)
    exact_transpose = all(a.exact_transpose_matvec is not None and b.exact_transpose_matvec is not None
                          for a, b, _ in factors)
    return WeightedHodgeOperator(
        "HODGE_" + sector.upper(), (n, n), grade, grade, action,
        source=rex, construction="weighted-hodge", variance="chain",
        transpose_matvec=lambda v: action(v, transpose=True),
        exact_matvec=(lambda v: action(v, exact=True)) if exact else None,
        exact_transpose_matvec=(lambda v: action(v, exact=True, transpose=True)) if exact_transpose else None,
        parameters=(("sector", sector),), grade_metric=center, lower_metric=lower, upper_metric=upper,
        sector=sector, active_sectors=tuple(active), metric_psd=True if sector != "difference" else None,
    )
