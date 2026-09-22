"""The resolvent rank: the PageRank fixed point as a Green response at any grade.

Personalized PageRank solves (I - alpha T) pi = (1 - alpha) v for a column stochastic
transition T. With L = I - T and lam = alpha / (1 - alpha) this is

    pi = (I + lam L)^-1 v,

the Green resolvent of L applied to the personalization. For loop free pair relations
with positive conductances W, the walk metrics M_0 = D^-1 and M_1 = W^-1, where
D = diag(B_1 W B_1^T), make the Hodge operator L_0 = B_1 B_1^dagger equal to I - A D^-1,
so at grade 0 the resolvent rank under the walk metric is exactly personalized PageRank.

The resolvent is defined at every grade in any positive metrics. The same response
therefore ranks relations and higher cells, reads a face through B_(k+1), and keeps each
harmonic direction: a hole is never damped, and a branching relation keeps the
differences between its tails. Where such directions exist the response conserves mass
but is signed rather than a probability. It is evaluated by the exact Green action of the
native field calculus; no transition matrix, operator or inverse is assembled.

Metric policies, each a declared observation rather than a default of the mathematics:

    declared   the source's declared metrics; a field calculus declares every grade, and
               a RexGraph declares its edge metric with the identity elsewhere
    walk       the grade 0 metric derived from the declared grade 1 metric, read as
               conductances W = M_1^-1: M_0 = diag(B_1 W B_1^T)^-1
    degree     every grade below the top derived the same way, from the top grade down:
               M_j = diag(B_(j+1) M_(j+1)^-1 B_(j+1)^T)^-1
    completed  the ranked grade's metric replaced by the completed field metric
               M_k (L_k + Pi^h) that the complete field coordinates induce

Under walk and degree a cell with no coface keeps its declared metric; it has no
participation to normalize. Metrics given explicitly per grade replace the policy's at
those grades, which is how a metric read from a field enters.
"""
from __future__ import annotations

from fractions import Fraction as Q

import numpy as np

__all__ = ["POLICIES", "rank_calculus", "resolvent_rank"]

POLICIES = ("declared", "walk", "degree", "completed")


def _diagonal(metric, grade: int, policy: str):
    entries = metric.as_sparse().entries
    if any(i != j for i, j in entries):
        raise ValueError(f"the {policy} metric derives from a diagonal grade {grade} metric")
    return [entries[i, i] for i in range(len(metric.space.keys))]


def _derived(complex_, grade: int, upper, declared):
    """diag(B_(j+1) M_(j+1)^-1 B_(j+1)^T)^-1, keeping the declared value where a cell has no coface."""
    participation = [Q(0)] * complex_.sizes[grade]
    for i, j, value in complex_.boundaries[grade]:
        participation[i] += Q(value) * Q(value) / upper[j]
    return [1 / p if p else d for p, d in zip(participation, declared, strict=True)]


def _explicit(metric, space, grade: int):
    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.graded_metric import DiagonalMetric
    if isinstance(metric, CoordinateMetric):
        if metric.space != space:
            raise ValueError(f"the grade {grade} metric must be declared on that grade's space")
        return metric
    if isinstance(metric, DiagonalMetric):
        if metric.grade != grade or metric.cell_keys is not None:
            raise ValueError(f"the grade {grade} metric must be on that grade's canonical basis")
        metric = metric.weights
    return CoordinateMetric.diagonal(space, list(metric))


def rank_calculus(source, grade: int, metric: str = "declared", metrics=None):
    """The native field calculus whose Green action the resolvent rank evaluates."""
    from numbers import Integral

    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.native_field import NativeFieldCalculus
    if metric not in POLICIES:
        raise ValueError(f"metric must be one of {', '.join(POLICIES)}; got {metric!r}")
    if isinstance(grade, bool) or not isinstance(grade, Integral):
        raise TypeError("grade must be an integer")
    if isinstance(source, NativeFieldCalculus):
        source.check_state()
        base = source
    else:
        base = NativeFieldCalculus.from_rex(source)
    complex_ = base.complex
    top = len(complex_.sizes) - 1
    if not 0 <= grade <= top:
        raise ValueError(f"grade {grade} is not present; the complex carries grades 0 to {top}")
    spaces = complex_.spaces
    chosen = list(base.metrics)
    if metrics is not None:
        metrics = list(metrics)
        if len(metrics) > len(chosen):
            raise ValueError("explicit metrics exceed the carried grades")
        for g, value in enumerate(metrics):
            if value is not None:
                chosen[g] = _explicit(value, spaces[g], g)
    if metric in ("walk", "degree"):
        for g in range(top - 1, -1, -1) if metric == "degree" else ((0,) if top else ()):
            upper = _diagonal(chosen[g + 1], g + 1, metric)
            declared = _diagonal(chosen[g], g, metric)
            chosen[g] = CoordinateMetric.diagonal(spaces[g], _derived(complex_, g, upper, declared))
    if metric == "completed":
        calculus = NativeFieldCalculus(complex_, tuple(chosen))
        n = complex_.sizes[grade]
        eye = np.eye(n, dtype=int)
        form = chosen[grade].apply(calculus.hodge(grade).apply(eye) + calculus.sector(grade, "harmonic").apply(eye))
        chosen[grade] = CoordinateMetric(spaces[grade], tuple(
            (i, j, form[i, j]) for i in range(n) for j in range(n) if form[i, j]))
    return NativeFieldCalculus(complex_, tuple(chosen))


def resolvent_rank(source, grade: int = 0, seed=None, damping=Q(17, 20), metric: str = "declared",
                   *, metrics=None, report: bool = False):
    """(I + lam L_grade)^-1 seed with lam = damping / (1 - damping), exactly.

    `source` is a RexGraph or a NativeFieldCalculus. `seed` is a nonnegative exact vector
    over the grade's cells, normalized to unit mass, and uniform when omitted, as PageRank
    teleports. `damping` is a rational in [0, 1). `metric` names a policy and `metrics`
    gives explicit per grade metrics (None keeps the policy's). Returns the exact response
    as an object array, with a record of the method, mass and sign when `report` is set.
    """
    from rexgraph.ranking_response import _damping, _seed
    if not isinstance(report, bool):
        raise TypeError("report must be a boolean")
    alpha = _damping(damping)
    calculus = rank_calculus(source, grade, metric, metrics)
    n = calculus.complex.sizes[int(grade)]
    if n == 0:
        raise ValueError("the resolvent rank requires a nonempty grade")
    lam = alpha / (1 - alpha)
    result = calculus.green(int(grade), lam).apply(_seed(seed, n))
    if not report:
        return result
    return result, {"method": "exact-graded-green-resolvent", "grade": int(grade), "metric": metric,
                    "damping": alpha, "resolvent_parameter": lam, "mass": sum(result, Q(0)),
                    "signed": any(v < 0 for v in result), "metric_digest": calculus.coefficient_digest}
